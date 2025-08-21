# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np

# =========================
# 0) (Tuỳ chọn) cấu hình MPS cho Mac
# =========================
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Cẩn trọng! Có thể gây thiếu ổn định. (PyTorch MPS)
# https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879

# =========================
# 1) Config
# =========================
MODEL_ID   = "openai/whisper-tiny"
OUT_DIR   = "./models"
CKPT_PATH = os.path.join(OUT_DIR, "whisper_tiny_de_finetuned")
LANG       = "de"
TASK       = "transcribe"
SR         = 16000
MAX_SECONDS= 15

device = "mps" if torch.backends.mps.is_available() else "cpu"

# =========================
# 2) Load model + processor
#    - Nếu checkpoint đã có đủ file (model + processor) → load từ đó
#    - Ngược lại → load từ MODEL_ID
# =========================
def _has_full_processor(path: str) -> bool:
    needed = ["preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "pytorch_model.bin"]
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, n)) for n in needed)

load_path = CKPT_PATH if _has_full_processor(CKPT_PATH) else MODEL_ID

processor = WhisperProcessor.from_pretrained(load_path, language=LANG, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(
    load_path,
    attn_implementation="eager",  # tránh SDPA ngốn RAM trên MPS
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(device)

# (Khuyến nghị) ép prompt decoder cho đúng ngôn ngữ / task
forced_ids = processor.get_decoder_prompt_ids(language=LANG, task=TASK)
if forced_ids is not None:
    model.generation_config.forced_decoder_ids = forced_ids

# =========================
# 3) Dataset (lọc <=15s)
# =========================
common_voice = DatasetDict({
    "train": load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "de",
        split="train[:1%]",
        trust_remote_code=True
    ),
    "validation": load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "de",
        split="validation[:1%]",
        trust_remote_code=True
    ),
})
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=SR))

def _dur_ok(ex):
    return len(ex["audio"]["array"]) <= SR * MAX_SECONDS

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].filter(_dur_ok)

# =========================
# 4) Preprocess
#   - KHÔNG tokenize nhãn ở đây (để collator làm bằng tokenizer(...))
#   - Lưu text gốc vào labels_text
# =========================
def preprocess(batch):
    audio = batch["audio"]
    fe = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=False
    )
    batch["input_features"] = fe.input_features[0]   # [80, T]
    batch["labels_text"]    = batch["sentence"]      # giữ text gốc
    return batch

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].map(
        preprocess,
        remove_columns=common_voice[split].column_names,  # giữ lại chỉ các cột mới
        num_proc=1
    )

# =========================
# 5) Collator: pad AUDIO -> [80,3000], tokenize labels bằng __call__
# =========================
@dataclass
class DataCollatorWhisper:
    processor: WhisperProcessor
    max_input_length: int = 3000     # Whisper encoder bắt buộc 3000 mel frames (~30s)
    max_label_length: int = 448

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- AUDIO ---
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

        # --- LABELS (dùng tokenizer(...) thay vì encode+pad) ---
        texts = [f["labels_text"] for f in features]
        labels_batch = self.processor.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Bỏ BOS nếu toàn batch bắt đầu bằng BOS
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and (labels[:, 0] == bos_id).all().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels.long()}

data_collator = DataCollatorWhisper(processor=processor)

# =========================
# 6) (Tuỳ chọn) Metrics WER cho lúc bạn muốn Trainer.evaluate (không dùng ở đây)
# =========================
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids = torch.where(
        torch.tensor(label_ids) == -100,
        torch.tensor(processor.tokenizer.pad_token_id),
        torch.tensor(label_ids)
    ).numpy()
    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# =========================
# 11) Eval thủ công (tránh OOM do Trainer gom predictions)
# =========================
def eval_streaming(model, dataset, processor, wer_metric, max_new_tokens=64):
    model.eval()
    wers = []
    for i in range(len(dataset)):
        sample = dataset[i]
        feats = sample["input_features"]
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(model.device)
        with torch.no_grad():
            pred_ids = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False
            )
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        ref_str  = sample["labels_text"]
        wer_val  = wer_metric.compute(predictions=[pred_str], references=[ref_str])
        wers.append(wer_val)

        del x, pred_ids
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return {"wer": float(np.mean(wers))}

# =========================
# Check if final checkpoint exists and load it to skip training
# =========================
if os.path.exists(CKPT_PATH) and os.path.isfile(os.path.join(CKPT_PATH, "pytorch_model.bin")):
    model = WhisperForConditionalGeneration.from_pretrained(CKPT_PATH).to(device)
    processor = WhisperProcessor.from_pretrained(CKPT_PATH)
    print("✅ Loaded final checkpoint, skip training.")
    metrics = eval_streaming(model, common_voice["validation"], processor, wer_metric)
    print("Eval metrics:", metrics)
    exit()

# =========================
# 7) Training args
#   - QUAN TRỌNG: remove_unused_columns=False để giữ labels_text tới collator
#   - Vô hiệu hoá checkpoint trung gian: chỉ lưu whisper_tiny_de_finetuned duy nhất
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1.25e-5,
    warmup_steps=10,
    max_steps=100,
    predict_with_generate=False,
    optim="adafactor",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    save_strategy="no",  # không lưu checkpoint trung gian tự động
    logging_steps=10,
    report_to="none",
    fp16=False,
    remove_unused_columns=False,   # giữ labels_text, tránh KeyError
)

# =========================
# 8) Trainer (train-only để tiết kiệm RAM)
# =========================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=None,
    data_collator=data_collator,
)

# =========================
# 9) Kiểm tra nếu whisper_tiny_de_finetuned đã có → bỏ qua training, chạy eval luôn
#    Nếu chưa có → train rồi lưu whisper_tiny_de_finetuned
# =========================
if _has_full_processor(CKPT_PATH):
    print(f"Checkpoint-final found at {CKPT_PATH}, skipping training.")
else:
    trainer.train()
    # Lưu duy nhất whisper_tiny_de_finetuned sau khi train xong
    os.makedirs(CKPT_PATH, exist_ok=True)
    trainer.save_model(CKPT_PATH)
    model.save_pretrained(CKPT_PATH)
    processor.save_pretrained(CKPT_PATH)

# =========================
# 11) Eval thủ công (tránh OOM do Trainer gom predictions)
# =========================
metrics = eval_streaming(model, common_voice["validation"], processor, wer_metric, max_new_tokens=64)
print("Eval metrics:", metrics)

# =========================
# 12) Test nhanh 1 mẫu
# =========================
sample = common_voice["validation"][0]
feats = sample["input_features"]
inp = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(model.device)
with torch.no_grad():
    gen_ids = model.generate(inp, max_new_tokens=64, num_beams=1, do_sample=False)
    transcription = processor.batch_decode(gen_ids, skip_special_tokens=True)
print("Reference:", sample["labels_text"])
print("Prediction:", transcription)