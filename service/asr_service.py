import os
import torch
import difflib
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer, cer

# ===============================
# Default model (German Whisper fine-tuned)
# ===============================
DEFAULT_MODEL = "./models/whisper_tiny_de"

# ===============================
# Load Whisper ASR Model
# ===============================
def load_model(model_id=None, model_dir=None):
    """
    Download (if needed) and load a Whisper model for German ASR.

    Args:
        model_id (str): Hugging Face repo ID (e.g., "openai/whisper-tiny-de")
                       or local path to model directory.
        model_dir (str): Local path where the model is stored or will be saved.
                         Defaults to model_id if not provided.

    Returns:
        pipeline: Hugging Face ASR pipeline ready for transcription.
    """
    # Use default model if nothing is provided
    if model_id is None:
        model_id = DEFAULT_MODEL
    if model_dir is None:
        model_dir = model_id

    # Ensure local model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # If model_dir is empty and model_id is a HF repo, download the model
    if not os.listdir(model_dir) and not os.path.isdir(model_id):
        print(f"[+] Downloading model {model_id} to: {model_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )

    # Device selection (GPU -> MPS -> CPU)
    if torch.cuda.is_available():
        device_idx, torch_dtype = 0, torch.float16
        device_for_model = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_idx, torch_dtype = 0, torch.float32
        device_for_model = "mps"
    else:
        device_idx, torch_dtype = -1, torch.float32
        device_for_model = "cpu"

    print(f"[+] Loading model from: {model_dir}")

    # Load model into memory
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    # Move to GPU or MPS if available
    if device_for_model != "cpu":
        model = model.to(device_for_model)

    # Load processor (tokenizer + feature extractor)
    processor = AutoProcessor.from_pretrained(model_dir)

    # Return Hugging Face ASR pipeline
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device_idx,
        chunk_length_s=10,    # split audio into chunks of 10s
        stride_length_s=(4, 2),  # overlapping stride for smoother transcription
    )

# ===============================
# Transcription
# ===============================
def transcribe(asr_pipeline, audio_input, language="de", timestamps=False):
    """
    Run service-to-text transcription.
    audio_input: numpy array hoặc path đến file âm thanh
    """
    return asr_pipeline(
        audio_input,
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=timestamps,
    )


# ===============================
# Pronunciation Scoring
# ===============================
def score_pronunciation(ref_text, hyp_text):
    """
    Compare reference and hypothesis text,
    calculate WER, CER, pronunciation score,
    and list mistaken words.
    """
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
# CLI Runner (test nhanh)
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR German + Pronunciation Scoring")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--ref_text", required=True, help="Reference German sentence")
    args = parser.parse_args()

    asr = load_model()
    result = transcribe(asr, args.audio)
    hyp_text = result["text"]

    metrics = score_pronunciation(args.ref_text, hyp_text)

    print("\n=== TRANSCRIPT ===")
    print(hyp_text)
    print("\n=== PRONUNCIATION EVAL ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")