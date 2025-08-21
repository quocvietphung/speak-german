# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# --- (Tùy chọn) nếu vẫn OOM MPS, cân nhắc bật 1 dòng dưới (cẩn trọng) ---
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # nới giới hạn bộ nhớ MPS   [oai_citation:4‡PyTorch Forums](https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879?utm_source=chatgpt.com)

# =========================
# 1) Model & Processor
# =========================
MODEL_ID = "openai/whisper-tiny"   # ✅ dùng tiny cho nhẹ RAM
LANG = "de"
TASK = "transcribe"

processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANG, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    attn_implementation="eager",  # tránh SDPA ngốn RAM trên MPS
)
model.config.use_cache = False
# tiết kiệm RAM activations (non-reentrant cho ổn định trên MPS)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# =========================
# 2) Dataset (lọc <=15s cho nhẹ)
# =========================
common_voice = DatasetDict({
    "train": load_dataset("mozilla-foundation/common_voice_13_0", "de", split="train[:1%]"),
    "validation": load_dataset("mozilla-foundation/common_voice_13_0", "de", split="validation[:1%]"),
})
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

MAX_SECONDS = 15
SR = 16000
def _dur_ok(ex):
    return len(ex["audio"]["array"]) <= SR * MAX_SECONDS

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].filter(_dur_ok)

# =========================
# 3) Preprocess (tách AUDIO/TEXT)
# =========================
def preprocess(batch):
    audio = batch["audio"]

    # AUDIO -> features (KHÔNG pad 3000 ở đây)
    fe = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=False
    )
    batch["input_features"] = fe.input_features[0]  # [80, T]

    # TEXT -> labels (giới hạn 448 token cho decoder)
    tok = processor.tokenizer(
        batch["sentence"],
        max_length=448,
        truncation=True
    )
    batch["labels"] = tok.input_ids  # list[int]
    return batch

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].map(
        preprocess,
        remove_columns=common_voice[split].column_names,
        num_proc=1
    )

# =========================
# 4) Collator ép AUDIO -> [80,3000] & LABELS -> 448
# =========================
@dataclass
class DataCollatorWhisper:
    processor: WhisperProcessor
    max_input_length: int = 3000   # Whisper encoder yêu cầu 30s = 3000 frames   [oai_citation:5‡GitHub](https://github.com/guillaumekln/faster-whisper/issues/171?utm_source=chatgpt.com)
    max_label_length: int = 448    # decoder limit ~448 token   [oai_citation:6‡GitHub](https://github.com/huggingface/transformers/issues/27445?utm_source=chatgpt.com)

    def __call__(self, features):
        # Pad/trim AUDIO về [80, 3000]
        feats = []
        for f in features:
            feat = f["input_features"]
            if not torch.is_tensor(feat):
                feat = torch.tensor(feat)
            T = feat.shape[-1]
            if T > self.max_input_length:
                feat = feat[:, : self.max_input_length]
            elif T < self.max_input_length:
                pad_T = self.max_input_length - T
                pad = torch.zeros(feat.size(0), pad_T, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=-1)
            feats.append(feat)
        input_features = torch.stack(feats, dim=0)  # [B,80,3000]

        # Pad LABELS -> 448 và mask -100
        label_dicts = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_dicts,
            padding="max_length",
            max_length=self.max_label_length,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Bỏ BOS nếu cần
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and (labels[:, 0] == bos_id).all().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels.long()}

data_collator = DataCollatorWhisper(processor=processor)

# =========================
# 5) (Tùy chọn) Metrics WER — bật khi bạn chạy evaluate sau train
# =========================
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids = torch.where(
        torch.tensor(label_ids) == -100,
        torch.tensor(processor.tokenizer.pad_token_id),
        torch.tensor(label_ids)
    ).numpy()
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# =========================
# 6) TrainingArguments (tiết kiệm RAM MPS)
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-de-test",
    per_device_train_batch_size=1,   # batch nhỏ
    gradient_accumulation_steps=8,   # bù qua accumulate
    learning_rate=1.25e-5,
    warmup_steps=10,
    max_steps=100,                   # chạy thử ngắn
    # Mặc định KHÔNG evaluate trong lúc train (tránh OOM). Bật lại sau khi ổn.
    predict_with_generate=False,     # generate trong eval rất tốn RAM
    optim="adafactor",               # optimizer tiết kiệm RAM
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    save_total_limit=1,
    logging_steps=10,
    report_to="none",
    fp16=False,                      # MPS không dùng fp16
)

# =========================
# 7) Trainer (chỉ train, không eval để tiết kiệm RAM)
# =========================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=None,               # không dùng validation lúc train
    data_collator=data_collator,
)

# =========================
# 8) Train
# =========================
trainer.train()

# =========================
# 9) Evaluate (tạo lại trainer với validation)
# =========================
eval_trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=common_voice["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

metrics = eval_trainer.evaluate()
print("Eval metrics:", metrics)

# =========================
# 10) Test trên 1 mẫu audio
# =========================
sample = common_voice["validation"][0]
input_features = sample["input_features"].unsqueeze(0).to(model.device)

with torch.no_grad():
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print("Reference:", processor.decode(sample["labels"], skip_special_tokens=True))
print("Prediction:", transcription)