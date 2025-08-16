from flask import Flask, request, jsonify
from flask_cors import CORS
from speech.asr_service import load_model, transcribe, score_pronunciation
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

# ===============================
# Load ASR model once
# ===============================
print("[+] Initializing ASR model...")
asr = load_model()
print("[+] Model loaded and ready!")

# ===============================
# Test route
# ===============================
@app.route("/api/hello", methods=["GET"])
def hello_api():
    return jsonify({"message": "Hello from Flask API!"}), 200


# ===============================
# Helper: Convert webm → wav
# ===============================
def convert_webm_to_wav(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:

        tmp_in.write(audio_bytes)
        tmp_in.flush()

        # convert bằng ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_in.name,
            "-ar", "16000", "-ac", "1", tmp_out.name
        ], check=True)

        data, samplerate = sf.read(tmp_out.name, dtype="float32")

        # xóa file tạm
        os.remove(tmp_in.name)
        os.remove(tmp_out.name)

        return data, samplerate


# ===============================
# Speech-to-Text + Scoring API
# ===============================
@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_bytes = request.files["audio"].read()
        ref_text = request.form.get("target_text", "").strip()

        data, samplerate = convert_webm_to_wav(audio_bytes)

        result = transcribe(asr, data, language="de")
        hyp_text = result.get("text", "").strip()

        score, mistakes, tip = None, [], ""
        if ref_text:
            metrics = score_pronunciation(ref_text, hyp_text)
            score = metrics.get("PronunciationScore")
            mistakes = [m["word"] for m in metrics.get("mistake_words", [])]
            tip = metrics.get("mistake_words", [{}])[0].get("tip", "") if metrics.get("mistake_words") else ""

        return jsonify({
            "transcript": hyp_text,
            "score": score,
            "mistakes": mistakes,
            "tip": tip
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ===============================
# Run server
# ===============================
if __name__ == "__main__":
    app.run(debug=True, port=8000)