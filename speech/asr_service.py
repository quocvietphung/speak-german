import os
import torch
from io import BytesIO
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer, cer

DEFAULT_MODEL = "primeline/whisper-large-v3-turbo-german"


# ===============================
#  Load model 1 lần dùng cho API
# ===============================
def load_model(model_id=DEFAULT_MODEL, model_dir="./models/whisper_de", offline=False):
    """
    Tải và load model Whisper để nhận dạng giọng nói.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not os.listdir(model_dir):
        print(f"[+] Downloading model to: {model_dir}")
        snapshot_download(repo_id=model_id, local_dir=model_dir, local_dir_use_symlinks=False)

    if torch.cuda.is_available():
        device_idx, torch_dtype = 0, torch.float16
        device_for_model = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_idx, torch_dtype = -1, torch.float32
        device_for_model = "mps"
    else:
        device_idx, torch_dtype = -1, torch.float32
        device_for_model = "cpu"

    print(f"[+] Loading model from: {model_dir}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    if device_for_model != "cpu":
        model = model.to(device_for_model)

    processor = AutoProcessor.from_pretrained(model_dir)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device_idx
    )


# ===============================
#  Transcribe
# ===============================
def transcribe(asr_pipeline, audio_input, language="de", timestamps=False):
    """
    Nhận diện giọng nói từ audio.
    - audio_input: đường dẫn file hoặc BytesIO
    """
    return asr_pipeline(
        audio_input,
        generate_kwargs={"language": language, "task": "transcribe"},
        chunk_length_s=30,
        stride_length_s=(4, 2),
        return_timestamps=timestamps,
    )


# ===============================
#  Scoring pronunciation
# ===============================
def score_pronunciation(ref_text, hyp_text):
    """
    So sánh câu chuẩn và câu nhận dạng, tính WER/CER và điểm.
    """
    w = wer(ref_text.strip(), hyp_text.strip())
    c = cer(ref_text.strip(), hyp_text.strip())
    return {
        "reference": ref_text.strip(),
        "hypothesis": hyp_text.strip(),
        "WER": w,
        "CER": c,
        "PronunciationScore": round((1 - w) * 100, 2)
    }


# ===============================
#  CLI runner (optional)
# ===============================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ASR German + Pronunciation scoring")
    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--ref_text", required=True, help="Reference German sentence")
    args = ap.parse_args()

    asr = load_model()
    result = transcribe(asr, args.audio)
    hyp_text = result["text"]

    metrics = score_pronunciation(args.ref_text, hyp_text)

    print("\n=== TRANSCRIPT ===")
    print(hyp_text)
    print("\n=== PRONUNCIATION EVAL ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")