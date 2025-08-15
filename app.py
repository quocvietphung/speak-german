from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from speech.asr_service import load_model, transcribe, score_pronunciation
from gtts import gTTS
import soundfile as sf
import numpy as np

# ===============================
#  Initialize Flask App
# ===============================
app = Flask(__name__)
CORS(app)  # Enable CORS for requests from frontend (Next.js)

# ===============================
#  Load the ASR model once when server starts
# ===============================
print("[+] Initializing ASR model...")
asr = load_model()
print("[+] Model loaded and ready!")


# ===============================
#  Test route to check API connection
# ===============================
@app.route("/api/hello", methods=["GET"])
def hello_api():
    """
    Quick test route to verify API is running.
    """
    return jsonify({"message": "Hello from Flask API!"}), 200


# ===============================
#  API to evaluate pronunciation
# ===============================
@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    ref_text = request.form.get("ref_text", None)
    if not ref_text:
        return jsonify({"error": "No reference text provided"}), 400

    try:
        if "audio" in request.files:
            audio_bytes = BytesIO(request.files["audio"].read())
        else:
            tts = gTTS(text=ref_text, lang="de")
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

        data, samplerate = sf.read(audio_bytes)
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        result = transcribe(asr, data, language="de")
        hyp_text = result.get("text", "").strip()

        metrics = score_pronunciation(ref_text, hyp_text)

        return jsonify({
            "transcript": hyp_text,
            "metrics": metrics
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===============================
#  Run server
# ===============================
if __name__ == "__main__":
    app.run(debug=True, port=8000)