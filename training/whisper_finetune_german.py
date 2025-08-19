# training/whisper_finetune_german.py

# 🤖 1. Import thư viện
import os
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List
import matplotlib.pyplot as plt

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)

# 🤖 2. Cấu hình ban đầu
MODEL_ID = "openai/whisper-large-v3"
OUTPUT_DIR = "./training/models/whisper_de_finetune"

# ⚡ Chỉ định nơi lưu dataset (cache HuggingFace)
os.environ["HF_HOME"] = "./training/datasets"

# Load dataset Common Voice 13.0 (tiếng Đức) với subset nhỏ để thử nghiệm
common_voice_train = load_dataset(
    "mozilla-foundation/common_voice_13_0", "de",
    split="train[:5000]+validation[:1000]"
)
common_voice_eval = load_dataset(
    "mozilla-foundation/common_voice_13_0", "de",
    split="test[:1000]"
)

# Chuyển audio sampling_rate về 16kHz cho phù hợp với Whisper
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16000))
common_voice_eval = common_voice_eval.cast_column("audio", Audio(sampling_rate=16000))

# 🤖 3. Load processor và model
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="de", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Đóng băng encoder để giảm số tham số phải train
model.freeze_encoder()

# 🤖 4. Hàm preprocess dữ liệu
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Map preprocess
train_dataset = common_voice_train.map(
    prepare_dataset,
    remove_columns=common_voice_train.column_names,
    num_proc=2
)
eval_dataset = common_voice_eval.map(
    prepare_dataset,
    remove_columns=common_voice_eval.column_names,
    num_proc=2
)

# 🤖 5. Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding bằng -100 để không tính vào loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 🤖 6. Metric WER
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer * 100}

# 🤖 7. Training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    learning_rate=1e-5,
    warmup_steps=200,
    num_train_epochs=3,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    max_grad_norm=1.0,
    logging_steps=50,
    report_to=["tensorboard"],  # bật TensorBoard
    eval_accumulation_steps=1,
    predict_with_generate=True,
    push_to_hub=False,
    dataloader_num_workers=2
)

# 🤖 8. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 🤖 9. Train
trainer.train()

# 🤖 10. Save model
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# 🤖 11. Trực quan hóa Loss, WER & Learning Rate
history = trainer.state.log_history

train_steps = [e["step"] for e in history if "loss" in e and "step" in e]
train_loss = [e["loss"] for e in history if "loss" in e]
eval_steps = [e["step"] for e in history if "eval_loss" in e]
eval_loss = [e["eval_loss"] for e in history if "eval_loss" in e]
eval_wer = [e["eval_wer"] for e in history if "eval_wer" in e]
lr_steps = [e["step"] for e in history if "learning_rate" in e]
lr_values = [e["learning_rate"] for e in history if "learning_rate" in e]

# Train vs Val Loss
plt.figure(figsize=(6,4))
plt.plot(train_steps, train_loss, label="Train loss")
plt.plot(eval_steps, eval_loss, label="Validation loss")
plt.xlabel("Step"); plt.ylabel("Loss")
plt.title("Training & Validation Loss"); plt.legend(); plt.show()

# Validation WER
plt.figure(figsize=(6,4))
plt.plot(eval_steps, eval_wer, marker='o', color='orange')
plt.xlabel("Step"); plt.ylabel("WER (%)")
plt.title("Validation WER"); plt.show()

# Learning Rate schedule
plt.figure(figsize=(6,4))
plt.plot(lr_steps, lr_values, label="Learning Rate", color="green")
plt.xlabel("Step"); plt.ylabel("LR")
plt.title("Learning Rate Schedule"); plt.legend(); plt.show()

print("✅ Training finished. Logs available in TensorBoard.")

# 🤖 12. Chạy TensorBoard (sau khi train xong, trong terminal gõ):
# tensorboard --logdir ./training/models/whisper_de_finetune/runs