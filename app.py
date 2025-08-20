from flask import Flask, request, jsonify
from flask_cors import CORS
from speech.asr_service import load_model, transcribe, score_pronunciation
import soundfile as sf
import traceback
import tempfile
import subprocess
import os
import time
import ollama

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
# API: Generate German sentence
# ===============================
@app.route("/api/sentence", methods=["GET"])
def generate_sentence():
    try:
        prompt = "Gib mir einen kurzen einfachen deutschen Satz zum Üben."
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "Du bist ein Deutschlehrer."},
                {"role": "user", "content": prompt}
            ]
        )
        sentence = response["message"]["content"].strip()
        return jsonify({
            "sentence": sentence,
            "timestamp": int(time.time())
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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

        score, mistakes, tip, teacher_feedback = None, [], "", ""

        if ref_text:
            metrics = score_pronunciation(ref_text, hyp_text)
            score = metrics.get("PronunciationScore")
            mistakes = [m["word"] for m in metrics.get("mistake_words", [])]
            tip = metrics.get("mistake_words", [{}])[0].get("tip", "") if metrics.get("mistake_words") else ""

            # Generate teacher-like feedback via Ollama (phonetics-focused)
            feedback_prompt = f"""
            Du bist ein freundlicher deutscher Sprachlehrer mit Fokus auf Phonetik.
            Analysiere die Aussprache des Schülers und gib klares, motivierendes Feedback.

            Referenzsatz: {ref_text}
            Gesprochen: {hyp_text}
            Bewertung: {score}%
            Fehlerwörter: {', '.join(mistakes) if mistakes else 'Keine'}

            Anforderungen für jedes Fehlerwort:
            - Korrekte Version
            - IPA-Aussprache (inkl. Schwa oder zentrale Vokale)
            - Betonung (Hauptakzent)
            - Ob Vokal lang oder kurz
            - Beispiel zur Wiederholung (z.B. ein ähnliches deutsches Wort oder Mini-Satz)

            Danach fasse in 2-3 motivierenden Sätzen zusammen,
            was der Schüler gut gemacht hat und was er verbessern sollte.
            """
            try:
                response = ollama.chat(
                    model="llama3",
                    messages=[
                        {"role": "system", "content": "Du bist ein Deutschlehrer mit Phonetik-Expertise."},
                        {"role": "user", "content": feedback_prompt}
                    ]
                )
                teacher_feedback = response["message"]["content"].strip()
            except Exception as e:
                teacher_feedback = "Feedback konnte nicht generiert werden."

        # ==== DEBUG LOG ====
        print("\n=== DEBUG /api/evaluate ===")
        print("Reference:", ref_text)
        print("Hypothesis:", hyp_text)
        print("Score:", score)
        print("Mistakes:", mistakes)
        print("Tip:", tip)
        print("Teacher feedback:", teacher_feedback)
        print("============================\n")

        return jsonify({
            "reference": ref_text,
            "hypothesis": hyp_text,
            "score": score,
            "mistakes": mistakes,
            "tip": tip,
            "teacher_feedback": teacher_feedback
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ===============================
# Run server
# ===============================
if __name__ == "__main__":
    app.run(debug=True, port=8000)