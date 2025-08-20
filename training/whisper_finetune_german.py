# training/whisper_finetune_german.py

# 1. Imports
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
from peft import get_peft_model, LoraConfig, TaskType

# 2. Config
MODEL_ID = "openai/whisper-large-v3"
OUTPUT_DIR = "./models/whisper_de_finetune"
os.environ["HF_HOME"] = "./datasets"

# 3. Load dataset with cache reuse
common_voice_train = load_dataset(
    "mozilla-foundation/common_voice_13_0", "de",
    split="train[:5000]+validation[:1000]",
    trust_remote_code=True,
    download_mode="reuse_dataset_if_exists"
)
common_voice_eval = load_dataset(
    "mozilla-foundation/common_voice_13_0", "de",
    split="test[:1000]",
    trust_remote_code=True,
    download_mode="reuse_dataset_if_exists"
)

common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16000))
common_voice_eval = common_voice_eval.cast_column("audio", Audio(sampling_rate=16000))

# 4. Processor & Model
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="de", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.freeze_encoder()

# 5. LoRA Configuration with target_modules to avoid errors
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # necessary to specify modules  [oai_citation:3‡Kaggle](https://www.kaggle.com/code/imtiazprio/fast-whisper-large-v2-fine-tuning-with-lora?utm_source=chatgpt.com) [oai_citation:4‡assemblyai.com](https://assemblyai.com/blog/how-to-evaluate-speech-recognition-models?utm_source=chatgpt.com) [oai_citation:5‡NVIDIA Docs](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/tts-evaluate.html?utm_source=chatgpt.com) [oai_citation:6‡Hugging Face](https://huggingface.co/spaces/openai/whisper/discussions/6?utm_source=chatgpt.com) [oai_citation:7‡Amazon Web Services, Inc.](https://aws.amazon.com/blogs/machine-learning/fine-tune-whisper-models-on-amazon-sagemaker-with-lora/?utm_source=chatgpt.com) [oai_citation:8‡GitHub](https://github.com/huggingface/peft?utm_source=chatgpt.com) [oai_citation:9‡Google Colab](https://colab.research.google.com/github/Vaibhavs10/fast-whisper-finetuning/blob/main/Whisper_w_PEFT.ipynb?utm_source=chatgpt.com)
)
model = get_peft_model(model, peft_config)

# 6. Preprocessing
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

train_dataset = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, num_proc=2)
eval_dataset = common_voice_eval.map(prepare_dataset, remove_columns=common_voice_eval.column_names, num_proc=2)

# 7. DataCollator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_feats = [{"input_features": f["input_features"]} for f in features]
        label_feats = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 8. Metrics: WER, CER, SER, BERTScore
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
ser_metric = evaluate.load("ser")
bert_score = evaluate.load("bertscore")

def compute_metrics(pred):
    pred_str = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str) * 100,
        "cer": cer_metric.compute(predictions=pred_str, references=label_str) * 100,
        "ser": ser_metric.compute(predictions=pred_str, references=label_str) * 100,
        "bertscore_f1": bert_score.compute(predictions=pred_str, references=label_str, lang="de")["f1"]
    }

# 9. TrainingArgs
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
    warmup_ratio=0.1,
    num_train_epochs=5,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    max_grad_norm=1.0,
    logging_steps=50,
    report_to=["tensorboard"],
    predict_with_generate=True,
    dataloader_num_workers=2
)

# 10. Trainer Setup
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 11. Train, Save, Evaluate
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

best = trainer.evaluate(eval_dataset)
print("===== Final metrics on best model =====")
print(f"WER: {best['eval_wer']:.2f}% | CER: {best['eval_cer']:.2f}% | "
      f"SER: {best['eval_ser']:.2f}% | BERTScore-F1: {best['eval_bertscore_f1']:.3f}")

# 12. Visualization
history = trainer.state.log_history
train_steps = [e["step"] for e in history if "loss" in e]
train_loss = [e["loss"] for e in history if "loss" in e]
eval_steps = [e["step"] for e in history if "eval_loss" in e]
eval_loss = [e["eval_loss"] for e in history if "eval_loss" in e]
eval_wer = [e["eval_wer"] for e in history if "eval_wer" in e]
eval_cer = [e["eval_cer"] for e in history if "eval_cer" in e]
eval_ser = [e["eval_ser"] for e in history if "eval_ser" in e]
eval_bert = [e["eval_bertscore_f1"] for e in history if "eval_bertscore_f1" in e]
lr_steps = [e["step"] for e in history if "learning_rate" in e]
lr_values = [e["learning_rate"] for e in history if "learning_rate" in e]

# Plotting
plt.figure(figsize=(6,4))
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Val Loss")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend(); plt.show()

plt.figure(figsize=(6,4))
plt.plot(eval_steps, eval_wer, marker='o', label="WER (%)", color='orange')
plt.plot(eval_steps, eval_cer, marker='x', label="CER (%)", color='red')
plt.plot(eval_steps, eval_ser, marker='s', label="SER (%)", color='purple')
plt.xlabel("Step"); plt.ylabel("%"); plt.title("Error Rates"); plt.legend(); plt.show()

plt.figure(figsize=(6,4))
plt.plot(eval_steps, eval_bert, marker='^', label="BERTScore F1", color='green')
plt.xlabel("Step"); plt.ylabel("F1"); plt.title("Semantic Accuracy"); plt.legend(); plt.show()

plt.figure(figsize=(6,4))
plt.plot(lr_steps, lr_values, label="LR Schedule", color='blue')
plt.xlabel("Step"); plt.ylabel("Learning Rate"); plt.title("LR"); plt.legend(); plt.show()

print("✅ Training complete. Logs & metrics available in TensorBoard.")