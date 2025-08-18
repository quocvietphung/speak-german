import os
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from datasets import load_dataset
from transformers import (
    WhisperConfig,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
import evaluate
import torch

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.steps = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self.steps.append(state.global_step)
        if "loss" in logs:
            self.train_loss.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_loss.append(logs["eval_loss"])

def compute_metrics(pred, processor, wer_metric):
    pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    acc = sum(p.strip() == l.strip() for p, l in zip(pred_str, label_str)) / len(pred_str)
    return {"wer": wer, "accuracy": acc}

def main():
    ds = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "de",
        split="train+validation",
        trust_remote_code=True
    )
    ds = ds.map(lambda x: {"text": x["sentence"]}, batched=False)
    ds = ds.train_test_split(test_size=0.1)

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="German", task="transcribe"
    )
    config = WhisperConfig.from_pretrained("openai/whisper-small")
    config.dropout = 0.1
    config.attention_dropout = 0.1
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", config=config)


    def preprocess_fn(x):
        audio = x["audio"]

        if audio["sampling_rate"] != 16000:
            audio_array = librosa.resample(
                audio["array"], orig_sr=audio["sampling_rate"], target_sr=16000
            )
            sampling_rate = 16000
        else:
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]

        # Extract features
        inputs = processor.feature_extractor(audio_array, sampling_rate=sampling_rate)
        return {
            "input_features": inputs.input_features[0],
            "labels": processor.tokenizer(x["text"]).input_ids
        }

    ds = ds.map(preprocess_fn, remove_columns=ds["train"].column_names, num_proc=2)

    wer_metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir="whisper-finetuned",
        per_device_train_batch_size=4 ,
        learning_rate=1e-5,
        optim="adamw_hf",
        weight_decay=0.01,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=500,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        logging_dir="runs"
    )

    loss_cb = LossHistoryCallback()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda p: compute_metrics(p, processor, wer_metric),
        callbacks=[loss_cb]
    )

    trainer.train()
    eval_res = trainer.evaluate()
    print("Evaluation results:", eval_res)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_cb.steps, loss_cb.train_loss, label="Train Loss")
    plt.plot(loss_cb.steps[:len(loss_cb.eval_loss)], loss_cb.eval_loss, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")
    plt.show()

    df = pd.DataFrame(trainer.state.log_history)
    print(df[['step', 'loss', 'eval_loss', 'wer', 'accuracy']])

if __name__ == "__main__":
    main()