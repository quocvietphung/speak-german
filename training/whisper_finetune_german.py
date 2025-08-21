from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

model_id = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_id, language="de", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# Load và preprocess dữ liệu (ví dụ Common Voice tiếng Đức)
common_voice = load_dataset("mozilla-foundation/common_voice_13_0", "de")
common_voice = common_voice.map(lambda b: {
    "input_features": processor.feature_extractor(b["audio"]["array"], sampling_rate=b["audio"]["sampling_rate"]).input_features[0],
    "labels": processor.tokenizer(b["sentence"]).input_ids
}, remove_columns=common_voice.column_names)

# Data collator và metrics
from dataclasses import dataclass
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-de-finetuned",
    per_device_train_batch_size=8,
    learning_rate=1.25e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()