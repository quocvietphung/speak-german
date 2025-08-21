import torch
import difflib
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer, cer

# ===============================
# Model paths
# ===============================
MODEL_BASE = "./models/whisper_tiny_de"
MODEL_FT   = "./models/whisper_tiny_de_finetuned"

# ===============================
# Device selection (GPU -> MPS -> CPU)
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
# Helper: build pipeline
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
# Load both models
# ===============================
asr_base = build_asr_pipeline(MODEL_BASE)
asr_fine_tuned = build_asr_pipeline(MODEL_FT)

# ===============================
# Transcription
# ===============================
def transcribe(asr_pipeline, audio_input, language="de", timestamps=False):
    return asr_pipeline(
        audio_input,
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=timestamps,
    )

# ===============================
# Pronunciation Scoring
# ===============================
def score_pronunciation(ref_text, hyp_text):
    w = wer(ref_text.strip(), hyp_text.strip())
    c = cer(ref_text.strip(), hyp_text.strip())

    ref_words = ref_text.strip().replace("!", "").replace(",", "").split()
    hyp_words = hyp_text.strip().replace("!", "").replace(",", "").split()

    mistakes = []
    seq = difflib.SequenceMatcher(None, ref_words, hyp_words)
    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if tag != 'equal':
            wrong_segment = hyp_words[j1:j2] or ["(missing)"]
            correct_segment = ref_words[i1:i2] or ["(extra word)"]
            for wrong_word in wrong_segment:
                mistakes.append({
                    "word": wrong_word,
                    "suggestion": " ".join(correct_segment),
                    "tip": f"Pronounce '{' '.join(correct_segment)}' more clearly."
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
# CLI Runner
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR German + Pronunciation Scoring (2 models)")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--ref_text", required=True, help="Reference German sentence")
    args = parser.parse_args()

    # Run with both models
    for name, asr in [("BASE", asr_base), ("FINE-TUNED", asr_ft)]:
        print(f"\n=== Running {name} Model ===")
        result = transcribe(asr, args.audio)
        hyp_text = result["text"]

        metrics = score_pronunciation(args.ref_text, hyp_text)

        print("\n--- TRANSCRIPT ---")
        print(hyp_text)
        print("\n--- PRONUNCIATION EVAL ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")