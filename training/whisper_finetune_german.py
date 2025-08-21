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

# (TÙY CHỌN) Nếu vẫn OOM MPS, cân nhắc mở 1–2 dòng dưới (ưu tiên giảm batch/độ dài trước):
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 1) Model & Processor
MODEL_ID = "openai/whisper-small"
LANG = "de"
TASK = "transcribe"

processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANG, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    attn_implementation="eager"  # tránh SDPA ngốn RAM trên MPS
)
model.config.use_cache = False     # giảm RAM khi train

# 2) Dataset (lọc file <= 15s để nhẹ RAM I/O; vẫn pad lên 30s=3000 frame theo chuẩn Whisper)
common_voice = DatasetDict({
    "train": load_dataset("mozilla-foundation/common_voice_13_0", "de", split="train[:1%]"),
    "validation": load_dataset("mozilla-foundation/common_voice_13_0", "de", split="validation[:1%]"),
})
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

MAX_SECONDS = 15
SR = 16000
def _dur_ok(example):
    return len(example["audio"]["array"]) <= SR * MAX_SECONDS

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].filter(_dur_ok)

# 3) Tiền xử lý: tạo input_features + labels (để collator pad/truncate về 3000)
def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["sentence"],
        return_tensors="pt",
        padding=False,
    )
    batch["input_features"] = inputs.input_features[0]  # [80, T]
    batch["labels"] = inputs.labels[0]                  # token ids
    return batch

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].map(
        preprocess,
        remove_columns=common_voice[split].column_names,
        num_proc=1
    )

# 4) Collator: pad/truncate về đúng chuẩn Whisper [80, 3000]
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    max_input_length: int = 3000   # Whisper yêu cầu 3000 frame (=30s)
    max_label_length: int = 448

    def __call__(self, features):
        feats = []
        for f in features:
            feat = f["input_features"]
            if isinstance(feat, list):
                feat = torch.tensor(feat)
            elif not torch.is_tensor(feat):
                feat = torch.tensor(feat)

            T = feat.shape[-1]
            if T > self.max_input_length:
                feat = feat[:, : self.max_input_length]
            elif T < self.max_input_length:
                pad_T = self.max_input_length - T
                pad = torch.zeros(feat.size(0), pad_T, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=-1)

            feats.append(feat)

        # [B, 80, 3000]
        input_features = torch.stack(feats, dim=0)

        # Pad labels tới max_label_length, rồi mask -100
        label_dicts = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_dicts,
            padding="max_length",
            max_length=self.max_label_length,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Nếu tất cả hàng bắt đầu bằng BOS thì bỏ BOS để tránh tính loss
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and (labels[:, 0] == bos_id).all().item():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,   # float tensor [B,80,3000]
            "labels": labels.long(),
        }

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5) Metrics
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

# 6) TrainingArguments — TẮT gradient checkpointing để tránh lỗi backward re-entrant
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-de-finetuned",
    per_device_train_batch_size=1,     # ↓ RAM MPS
    gradient_accumulation_steps=16,    # giữ effective batch lớn
    learning_rate=1.25e-5,
    warmup_steps=200,
    max_steps=2000,

    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    predict_with_generate=True,

    gradient_checkpointing=False,      # <-- tắt để tránh lỗi backward lần 2
    dataloader_pin_memory=False,       # MPS không hỗ trợ
    dataloader_num_workers=0,
    save_total_limit=2,
    report_to="none",
)

# 7) Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["validation"],
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

# 8) Train
trainer.train()

# -------- PHƯƠNG ÁN B (nếu muốn checkpointing mà vẫn an toàn) ----------
# BẬT lại GC theo kiểu non-reentrant:
# model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
# và đổi training_args.gradient_checkpointing=False (giữ nguyên như trên, KHÔNG bật ở args)
# Tham khảo PyTorch docs & HF issues về use_reentrant=False.  # noqa
