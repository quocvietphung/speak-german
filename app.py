from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from speech.asr_service import load_model, transcribe, score_pronunciation
from gtts import gTTS
import soundfile as sf
import numpy as np

# ===============================
#  Khởi tạo Flask App
# ===============================
app = Flask(__name__)
CORS(app)  # Cho phép CORS để gọi API từ frontend (Next.js)

# ===============================
#  Load model 1 lần khi server start
# ===============================
print("[+] Initializing ASR model...")
asr = load_model()
print("[+] Model loaded and ready!")

# ===============================
#  Route test kết nối
# ===============================
@app.route("/api/hello", methods=["GET"])
def hello_api():
    """
    Route test nhanh để kiểm tra API hoạt động.
    """
    return jsonify({"message": "Hello from Flask API!"}), 200

# ===============================
#  API Evaluate pronunciation
# ===============================
@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    """
    Nếu không có audio từ client, tạo audio giả từ ref_text để test.
    """
    ref_text = request.form.get("ref_text", None)
    if not ref_text:
        return jsonify({"error": "No reference text provided"}), 400

    try:
        if "audio" in request.files:
            # 🔹 Dùng file audio người dùng upload
            audio_bytes = BytesIO(request.files["audio"].read())
        else:
            # 🔹 Tạo audio giả từ ref_text để test
            tts = gTTS(text=ref_text, lang="de")
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

        # Đọc file audio thành numpy array cho Whisper
        data, samplerate = sf.read(audio_bytes)
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Nhận dạng
        result = transcribe(asr, data, language="de")
        hyp_text = result.get("text", "").strip()

        # Tính điểm phát âm
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