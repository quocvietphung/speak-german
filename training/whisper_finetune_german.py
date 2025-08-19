# 🤖 1. Import libraries
import os
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# 🤖 2. Config
MODEL_ID = "openai/whisper-large-v3"
OUTPUT_DIR = "./whisper_de_finetune"

# Dataset: Common Voice 13.0 (German)
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "de", split="train+validation")
eval_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "de", split="test")

# Cast audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

# 🤖 3. Load processor + model
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="de", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Optional: freeze encoder để train nhanh hơn
model.freeze_encoder()

# 🤖 4. Preprocess function
def prepare_dataset(batch):
    audio = batch["audio"]

    # Extract features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]

    # Encode target text
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names, num_proc=4)

# 🤖 5. Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

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

    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# 🤖 7. Training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=5,
    logging_steps=50,
    eval_steps=200,
    predict_with_generate=True,
    generation_max_length=225,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    report_to=["tensorboard"]
)

# 🤖 8. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
)

# 🤖 9. Train
trainer.train()

# 🤖 10. Save
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)