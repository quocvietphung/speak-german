# 🤖 1. Import thư viện
import os
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback  # thêm EarlyStopping
)

# 🤖 2. Cấu hình ban đầu
MODEL_ID = "openai/whisper-large-v3"
OUTPUT_DIR = "./training/models/whisper_de_finetune"

# Load dataset Common Voice 13.0 (tiếng Đức)
common_voice_train = load_dataset("mozilla-foundation/common_voice_13_0", "de", split="train+validation")
common_voice_eval = load_dataset("mozilla-foundation/common_voice_13_0", "de", split="test")

# Chuyển audio sampling_rate về 16kHz cho phù hợp với Whisper
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16000))
common_voice_eval = common_voice_eval.cast_column("audio", Audio(sampling_rate=16000))

# 🤖 3. Load processor và model
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="de", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Đóng băng encoder để giảm số tham số phải train (tăng tốc và tránh overfitting)
model.freeze_encoder()

# 🤖 4. Hàm preprocess dữ liệu (chuẩn bị input_features và labels)
def prepare_dataset(batch):
    audio = batch["audio"]
    # Tính feature từ audio
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    # Mã hóa câu thoại (sentence) thành token id để làm nhãn
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Map hàm preprocess vào dataset
train_dataset = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, num_proc=4)
eval_dataset = common_voice_eval.map(prepare_dataset, remove_columns=common_voice_eval.column_names, num_proc=4)

# 🤖 5. Data collator tùy biến cho Seq2Seq (padding cho audio và text)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Tách input_features và labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Padding cho input_features (audio) và chuyển sang tensor
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # Padding cho labels (text) và chuyển sang tensor
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Thay token padding bằng -100 để không tính vào loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 🤖 6. Định nghĩa metric WER (Word Error Rate)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Giải mã chuỗi dự đoán và nhãn thật về text
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # bỏ qua -100
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Tính WER bằng thư viện evaluate
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer * 100}  # nhân 100 để biểu thị dưới dạng %
    # (WER càng thấp càng tốt, 0% là hoàn hảo)

# 🤖 7. Thiết lập tham số huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    # Batch size (có thể giảm nếu OOM, Apple M3 không có GPU CUDA nên chạy trên CPU/MPS)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # tích lũy gradient cho batch hiệu dụng = 16
    # Chiến lược đánh giá & lưu mô hình
    evaluation_strategy="epoch",       # đánh giá mỗi epoch
    save_strategy="epoch",             # lưu checkpoint mỗi epoch
    load_best_model_at_end=True,       # tự động load model tốt nhất cuối training
    metric_for_best_model="wer",       # dùng WER trên eval để chọn model tốt nhất
    greater_is_better=False,           # WER càng thấp càng tốt
    save_total_limit=2,                # chỉ lưu tối đa 2 checkpoint (tiết kiệm dung lượng)
    # Thiết lập optimizer và lịch học
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=5,               # train tối đa 5 epoch (có thể dừng sớm nếu không cải thiện)
    # Kỹ thuật hỗ trợ training
    gradient_checkpointing=True,      # dùng gradient checkpointing để tiết kiệm bộ nhớ
    fp16=torch.cuda.is_available(),   # dùng FP16 nếu có GPU CUDA (MPS chưa hỗ trợ FP16 tốt)
    max_grad_norm=1.0,                # gradient clipping với norm tối đa 1.0
    # Logging
    logging_steps=50,
    report_to=["tensorboard"],        # log ra tensorboard để trực quan nếu cần
    eval_accumulation_steps=1,        # tính metric trên toàn bộ tập eval một lượt
    predict_with_generate=True,       # generate text trong eval để tính WER
    push_to_hub=False                 # không push lên Hugging Face Hub trong quá trình train
)

# 🤖 8. Khởi tạo Trainer với các thành phần cần thiết
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)]
    # EarlyStopping: nếu WER không cải thiện sau 3 lần eval (3 epoch liên tiếp) thì dừng
)

# 🤖 9. Bắt đầu huấn luyện
trainer.train()

# 🤖 10. Lưu model và tokenizer đã fine-tune
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# 🤖 11. Trực quan hóa kết quả huấn luyện (vẽ đồ thị Loss và WER)
import matplotlib.pyplot as plt

history = trainer.state.log_history  # lịch sử log

# Lấy các giá trị loss và WER từ lịch sử log
train_steps = [entry["step"] for entry in history if "loss" in entry and "step" in entry]
train_loss = [entry["loss"] for entry in history if "loss" in entry]
eval_steps = [entry["step"] for entry in history if "eval_loss" in entry]
eval_loss = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
eval_wer = [entry["eval_wer"] for entry in history if "eval_wer" in entry]

# Vẽ biểu đồ Train vs Val Loss
plt.figure(figsize=(6,4))
plt.plot(train_steps, train_loss, label="Train loss")
plt.plot(eval_steps, eval_loss, label="Validation loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()

# Vẽ biểu đồ WER trên tập Validation
plt.figure(figsize=(6,4))
plt.plot(eval_steps, eval_wer, marker='o', color='orange')
plt.xlabel("Step")
plt.ylabel("WER (%)")
plt.title("Validation WER")
plt.show()
