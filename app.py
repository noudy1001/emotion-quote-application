import os
import tempfile
import torch
import torchaudio
from torchaudio.transforms import Resample
from flask import Flask, request, jsonify
import pandas as pd
import random
import csv
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -----------------------------
# Gradio API URL (update if needed)
# -----------------------------
GRADIO_API_URL = "https://5efe1650c0a5918059.gradio.live/infer"

# -----------------------------
# Load Quotes CSV
# -----------------------------
quotes_path = "quotes.csv"
quotes = []
if os.path.exists(quotes_path):
    try:
        with open(quotes_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    quote = row.get("quote", "").strip()
                    author = row.get("author", "").strip()
                    category = row.get("category", "").strip()
                    if quote and author and category:
                        quotes.append({"quote": quote, "author": author, "category": category})
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading quotes.csv: {e}")

df = pd.DataFrame(quotes)

# -----------------------------
# Mood to Tags Mapping
# -----------------------------
MOOD_TO_TAGS = {
    "angry": ["anger", "frustration", "mistakes", "hate", "evil", "despair"],
    "sad": ["sad", "loss", "death", "regret", "grieving"],
    "neutral": ["wisdom", "truth", "knowledge"],
    "happy": ["happiness", "joy", "love", "smile", "hope", "fun"]
}

# -----------------------------
# Normalize audio to 16k mono
# -----------------------------
def ensure_16k_mono(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = Resample(sr, 16000)(wav)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, wav, 16000)
    tmp.close()
    return tmp.name

# -----------------------------
# Get quote based on emotion
# -----------------------------
def get_quote_for_emotion(emotion_label):
    keywords = MOOD_TO_TAGS.get(emotion_label.lower(), ["life", "wisdom"])
    if df.empty:
        return {"text": "Could not load quotes.", "author": "", "tags": []}

    filtered = df[df["category"].apply(lambda x: any(k.lower() in str(x).lower() for k in keywords))]
    if filtered.empty:
        row = random.choice(df[["quote", "author"]].values)
    else:
        row = random.choice(filtered[["quote", "author"]].values)

    return {"text": row[0], "author": row[1], "tags": keywords}

# -----------------------------
# Root Route (Health Check)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🎧 Emotion Quote API is running successfully!"})

# -----------------------------
# Main Emotion Inference Endpoint
# -----------------------------
@app.route("/infer/audio", methods=["POST"])
def infer_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f = request.files["audio"]
    filename = secure_filename(f.filename)
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.save(tmp_in.name)
    tmp_in.close()

    try:
        tmp_norm = ensure_16k_mono(tmp_in.name)

        # Send audio to Gradio API
        with open(tmp_norm, "rb") as audio_file:
            files = {"audio": (filename, audio_file, "audio/wav")}
            response = requests.post(f"{GRADIO_API_URL}/audio", files=files)

        if response.status_code != 200:
            return jsonify({"error": "Failed to get response from Gradio API"}), 500

        data = response.json()
        emotion = data.get("emotion", "").lower()
        confidence = data.get("confidence", 0)

        quote_data = get_quote_for_emotion(emotion)

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "quote": quote_data
        })

    finally:
        for path in [tmp_in.name, tmp_norm]:
            try:
                os.remove(path)
            except:
                pass

# -----------------------------
# Run Server (Local)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
