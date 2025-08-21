import os
import torch
import difflib
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer, cer

# ===============================
# Model paths
# ===============================
HF_REPO_ID = "openai/whisper-tiny"
MODEL_BASE = "./models/whisper_tiny_de"
MODEL_FT   = "./models/whisper_tiny_de_finetuned"

# ===============================
# Device selection
# ===============================
if torch.cuda.is_available():
    device_idx, torch_dtype = 0, torch.float16
    device_for_model = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device_idx, torch_dtype = 0, torch.float32
    device_for_model = "mps"
else:
    device_idx, torch_dtype = -1, torch.float32
    device_for_model = "cpu"

print(f"[+] Using device: {device_for_model}")

# ===============================
# Ensure base model exists
# ===============================
def ensure_base_model():
    if not os.path.exists(MODEL_BASE) or not os.listdir(MODEL_BASE):
        print(f"[+] Downloading base model from HF repo: {HF_REPO_ID}")
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=MODEL_BASE,
            local_dir_use_symlinks=False
        )

# ===============================
# Build ASR pipeline
# ===============================
def build_asr_pipeline(model_dir: str):
    print(f"[+] Loading model from: {model_dir}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device_for_model)
    processor = AutoProcessor.from_pretrained(model_dir)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device_idx,
        chunk_length_s=10,
        stride_length_s=(4, 2),
    )

# ===============================
# Transcription + Scoring
# ===============================
def transcribe(asr_pipeline, audio_input, language="de", timestamps=False):
    return asr_pipeline(
        audio_input,
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=timestamps,
    )

def score_pronunciation(ref_text, hyp_text):
    w = wer(ref_text.strip(), hyp_text.strip())
    c = cer(ref_text.strip(), hyp_text.strip())
    mistakes = []
    seq = difflib.SequenceMatcher(None, ref_text.split(), hyp_text.split())
    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if tag != "equal":
            wrong = hyp_text.split()[j1:j2] or ["(missing)"]
            correct = ref_text.split()[i1:i2] or ["(extra)"]
            mistakes.append({
                "word": " ".join(wrong),
                "suggestion": " ".join(correct),
                "tip": f"Pronounce '{' '.join(correct)}' more clearly."
            })

    return {
        "reference": ref_text.strip(),
        "hypothesis": hyp_text.strip(),
        "WER": w,
        "CER": c,
        "PronunciationScore": round((1 - w) * 100, 2),
        "mistake_words": mistakes
    }

# ===============================
# PRELOAD MODELS (important!)
# ===============================
ensure_base_model()
asr_base = build_asr_pipeline(MODEL_BASE)
asr_fine_tuned = build_asr_pipeline(MODEL_FT)

print("[+] ASR models preloaded successfully!")