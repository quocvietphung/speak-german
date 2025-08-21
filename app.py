from flask import Flask, request, jsonify
from flask_cors import CORS
from service.asr_service import asr_base, asr_fine_tuned, transcribe, score_pronunciation
import soundfile as sf
import traceback
import tempfile
import subprocess
import os

# ===============================
# Initialize Flask App
# ===============================
app = Flask(__name__)
CORS(app)

print("[+] ASR models (base + fine-tuned) preloaded!")

# ===============================
# Test route (simple health check)
# ===============================
@app.route("/api/hello", methods=["GET"])
def hello_api():
    return jsonify({"message": "Hello from Flask API!"}), 200


# ===============================
# Helper: Convert uploaded WebM audio → WAV (16kHz, mono)
# ===============================
def convert_webm_to_wav(audio_bytes):
    # Write temporary WebM and WAV files
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:

        tmp_in.write(audio_bytes)
        tmp_in.flush()

        # Use ffmpeg to convert WebM → WAV
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_in.name,
            "-ar", "16000", "-ac", "1", tmp_out.name
        ], check=True)

        # Load WAV audio into numpy array
        data, samplerate = sf.read(tmp_out.name, dtype="float32")

        # Clean up temp files
        os.remove(tmp_in.name)
        os.remove(tmp_out.name)

        return data, samplerate


# ===============================
# Speech-to-Text + Pronunciation Scoring API
# ===============================
@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    try:
        # Check if audio file is provided
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        # Read audio and reference text from request
        audio_bytes = request.files["audio"].read()
        ref_text = request.form.get("target_text", "").strip()
        model_id = request.form.get("model_id", "fine_tuned")  # default = fine-tuned

        # Convert WebM → WAV
        data, samplerate = convert_webm_to_wav(audio_bytes)

        # Select which ASR model to use
        if model_id == "base":
            asr = asr_base
        elif model_id == "fine_tuned":
            asr = asr_fine_tuned
        else:
            return jsonify({"error": f"Invalid model_id '{model_id}'"}), 400

        # Run transcription
        result = transcribe(asr, data, language="de")
        hyp_text = result.get("text", "").strip()

        # Run scoring if reference text is provided
        score, mistakes, tip = None, [], ""
        if ref_text:
            metrics = score_pronunciation(ref_text, hyp_text)
            score = metrics.get("PronunciationScore")
            mistakes = [m["word"] for m in metrics.get("mistake_words", [])]
            tip = metrics.get("mistake_words", [{}])[0].get("tip", "") if metrics.get("mistake_words") else ""

        # Return JSON response
        return jsonify({
            "reference": ref_text,
            "hypothesis": hyp_text,
            "score": score,
            "mistakes": mistakes,
            "tip": tip,
            "model_used": model_id
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ===============================
# Run Flask Development Server
# ===============================
if __name__ == "__main__":
    app.run(debug=True, port=8000)