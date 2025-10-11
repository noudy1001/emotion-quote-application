import os
import tempfile
import base64
import csv
import random
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
import requests

app = Flask(__name__)

# --------------------------------
# Gradio API Endpoint (update as needed)
# --------------------------------
GRADIO_API_URL = "https://5efe1650c0a5918059.gradio.live"

# --------------------------------
# Load quotes dataset
# --------------------------------
quotes_path = "quotes.csv"
quotes = []
if os.path.exists(quotes_path):
    try:
        with open(quotes_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                quote = row.get("quote", "").strip()
                author = row.get("author", "").strip()
                category = row.get("category", "").strip()
                if quote and author and category:
                    quotes.append({"quote": quote, "author": author, "category": category})
    except Exception as e:
        print(f"[ERROR] Reading quotes.csv: {e}")
df = pd.DataFrame(quotes)

# --------------------------------
# Mood → tag keywords
# --------------------------------
MOOD_TO_TAGS = {
    "angry": ["anger", "frustration", "mistakes", "hate", "evil", "despair"],
    "sad": ["sad", "loss", "death", "regret", "grieving"],
    "neutral": ["wisdom", "truth", "knowledge"],
    "happy": ["happiness", "joy", "love", "smile", "hope", "fun"]
}

# --------------------------------
# Normalize audio: 16k mono
# --------------------------------
def ensure_16k_mono(path):
    wav, sr = torchaudio.load(path)
    # If multiple channels, convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != 16000:
        wav = Resample(sr, 16000)(wav)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, wav, 16000)
    tmp.close()
    return tmp.name

# --------------------------------
# Pick quote by emotion
# --------------------------------
def get_quote_for_emotion(emotion_label):
    keywords = MOOD_TO_TAGS.get(emotion_label.lower(), ["life", "wisdom"])
    if df.empty:
        return {"text": "Could not load quotes.", "author": "", "tags": []}
    # Filter rows whose category contains any of the keywords
    filtered = df[df["category"].apply(lambda x: any(k.lower() in str(x).lower() for k in keywords))]
    if filtered.empty:
        # fallback to random
        row = random.choice(df[["quote", "author"]].values)
    else:
        row = random.choice(filtered[["quote", "author"]].values)
    return {"text": row[0], "author": row[1], "tags": keywords}

# --------------------------------
# Health check
# --------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🎧 Emotion Quote API is running successfully!"})

# --------------------------------
# Main endpoint: infer emotion from audio
# --------------------------------
@app.route("/infer/audio", methods=["POST"])
def infer_audio():
    tmp_in = None
    tmp_norm = None
    try:
        # Check file
        if "audio" not in request.files:
            return jsonify({"error": "No audio file in request"}), 400

        f = request.files["audio"]
        filename = secure_filename(f.filename)
        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.save(tmp_in.name)
        tmp_in.close()

        print(f"[INFO] Received file: {filename}")

        # Normalize
        try:
            tmp_norm = ensure_16k_mono(tmp_in.name)
        except Exception as e:
            print(f"[WARN] Audio normalization failed: {e}")
            tmp_norm = tmp_in.name  # fallback

        # Read file and convert to base64
        with open(tmp_norm, "rb") as af:
            audio_bytes = af.read()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Prepare payload for Gradio
        payload = {
            "data": [f"data:audio/wav;base64,{b64_audio}"]
        }

        gradio_url = f"{GRADIO_API_URL}"
        print(f"[INFO] Sending request to Gradio: {gradio_url}")
        print(f"[DEBUG] Payload keys: {payload.keys()}")

        response = requests.post(gradio_url, json=payload, timeout=30)
        print(f"[INFO] Gradio status code: {response.status_code}")
        print(f"[DEBUG] Gradio response text: {response.text}")

        if response.status_code != 200:
            return jsonify({
                "error": "Gradio API failure",
                "status_code": response.status_code,
                "response": response.text
            }), 500

        result = response.json()
        print(f"[DEBUG] Gradio JSON result: {result}")

        # Parse response
        if isinstance(result, dict) and "data" in result and isinstance(result["data"], list):
            prediction = result["data"][0]
            emotion = prediction.get("label") or prediction.get("emotion", "")
            # Some Gradio outputs use “label” field
            confidence = prediction.get("confidence", 0)
        else:
            return jsonify({
                "error": "Invalid response format from Gradio",
                "result": result
            }), 500

        quote_data = get_quote_for_emotion(emotion.lower())

        return jsonify({
            "emotion": emotion.lower(),
            "confidence": confidence,
            "quote": quote_data
        })

    except Exception as e:
        print(f"[ERROR] Exception in infer_audio: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    finally:
        for p in (tmp_in and [tmp_in.name] or []) + (tmp_norm and [tmp_norm] or []):
            try:
                os.remove(p)
            except Exception:
                pass

# --------------------------------
# Run the server
# --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
