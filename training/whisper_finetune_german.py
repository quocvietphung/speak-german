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
import numpy as np

# =========================
# 1) Config
# =========================
MODEL_ID = "openai/whisper-tiny"
CKPT_PATH = "./whisper-tiny-de-test/checkpoint-final"
LANG = "de"
TASK = "transcribe"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# =========================
# 2) Load model + processor
# =========================
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANG, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    attn_implementation="eager"
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(device)

# =========================
# 3) Dataset (lọc <=15s)
# =========================
common_voice = DatasetDict({
    "train": load_dataset("mozilla-foundation/common_voice_13_0", "de", split="train[:1%]"),
    "validation": load_dataset("mozilla-foundation/common_voice_13_0", "de", split="validation[:1%]"),
})
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

MAX_SECONDS, SR = 15, 16000
def _dur_ok(ex):
    return len(ex["audio"]["array"]) <= SR * MAX_SECONDS

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].filter(_dur_ok)

# =========================
# 4) Preprocess
# =========================
def preprocess(batch):
    audio = batch["audio"]
    fe = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=False
    )
    batch["input_features"] = fe.input_features[0]

    batch["labels_text"] = batch["sentence"]
    return batch

for split in ["train", "validation"]:
    common_voice[split] = common_voice[split].map(
        preprocess,
        remove_columns=common_voice[split].column_names,
        num_proc=1
    )

# =========================
# 5) Collator
# =========================
@dataclass
class DataCollatorWhisper:
    processor: WhisperProcessor
    max_input_length: int = 3000
    max_label_length: int = 448

    def __call__(self, features):
        feats = []
        for f in features:
            feat = torch.tensor(f["input_features"]) if not torch.is_tensor(f["input_features"]) else f["input_features"]
            T = feat.shape[-1]
            if T > self.max_input_length:
                feat = feat[:, : self.max_input_length]
            elif T < self.max_input_length:
                pad = torch.zeros(feat.size(0), self.max_input_length - T, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=-1)
            feats.append(feat)
        input_features = torch.stack(feats, dim=0)

        texts = [f["labels_text"] for f in features]
        labels_batch = self.processor.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and (labels[:, 0] == bos_id).all().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels.long()}

data_collator = DataCollatorWhisper(processor=processor)

# =========================
# 6) Metrics (WER)
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
# 7) Training args
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-de-test",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1.25e-5,
    warmup_steps=10,
    max_steps=100,
    predict_with_generate=False,
    optim="adafactor",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    save_total_limit=1,
    logging_steps=10,
    report_to="none",
    fp16=False,
)

# =========================
# 8) Trainer
# =========================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=None,
    data_collator=data_collator,
)

# =========================
# 9) Train
# =========================
trainer.train()

# ✅ Save model + processor đầy đủ để load lại không lỗi
trainer.save_model(CKPT_PATH)
processor.save_pretrained(CKPT_PATH)

# =========================
# 10) Eval thủ công
# =========================
def eval_streaming(model, dataset, processor, wer_metric, max_new_tokens=64):
    model.eval()
    wers = []
    for i in range(len(dataset)):
        sample = dataset[i]
        x = sample["input_features"].unsqueeze(0).to(model.device)
        with torch.no_grad():
            pred_ids = model.generate(x, max_new_tokens=max_new_tokens)

        # 🔹 Prediction
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        # 🔹 Reference (lấy text gốc)
        ref_str = sample["labels_text"]

        wer_val = wer_metric.compute(predictions=[pred_str], references=[ref_str])
        wers.append(wer_val)

        # cleanup
        del x, pred_ids
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return {"wer": float(np.mean(wers))}

# chạy
metrics = eval_streaming(model, common_voice["validation"], processor, wer_metric)
print("Eval metrics:", metrics)

# =========================
# 11) Test trên 1 mẫu
# =========================
sample = common_voice["validation"][0]
inp = sample["input_features"].unsqueeze(0).to(model.device)
with torch.no_grad():
    gen_ids = model.generate(inp, max_new_tokens=64)
    transcription = processor.batch_decode(gen_ids, skip_special_tokens=True)

print("Reference:", processor.decode(sample["labels"], skip_special_tokens=True))
print("Prediction:", transcription)