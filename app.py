import os
import tempfile
import torch
import torchaudio
from torchaudio.transforms import Resample
from flask import Flask, request, jsonify
import pandas as pd
import random
import csv
from speechbrain.inference import EncoderClassifier  # updated path

app = Flask(__name__)

# -----------------------------
# Load model
# -----------------------------
print("Loading SpeechBrain model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir=None
)
print("Model loaded.")

# -----------------------------
# Load CSV quotes
# -----------------------------
quotes_path = "quotes.csv"
quotes = []
if os.path.exists(quotes_path):
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

df = pd.DataFrame(quotes)

# -----------------------------
# Expanded keyword mapping
# -----------------------------
MOOD_TO_TAGS = {
    "angry": [
        "anger", "frustration", "mistakes", "out-of-control", "worst", "hate", "do-wrong", "evil", "despair", "pain", "heartache"
    ],
    "sad": [
        "sad", "loss", "death", "despair", "hopeless", "heartbreak", "suffering", "waiting", "regret", "disappointment", "grieving"
    ],
    "neutral": [
        "wisdom", "balance", "thought", "philosophy", "truth", "mind", "reality", "knowledge", "inspirational", "learning"
    ],
    "happy": [
        "happiness", "joy", "love", "smile", "positivity", "best", "life", "fun", "friendship", "peace", "hope", "adventure", "optimism", "trust", "inspirational"
    ]
}

# -----------------------------
# Helper: normalize audio
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
# Helper: pick quote
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
# API endpoint
# -----------------------------
@app.route("/infer/audio", methods=["POST"])
def infer_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f = request.files["audio"]
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.save(tmp_in.name)
    tmp_in.close()

    try:
        tmp_norm = ensure_16k_mono(tmp_in.name)

        # Inference
        out_prob, score, index, text_lab = classifier.classify_file(tmp_norm)
        if not isinstance(out_prob, torch.Tensor):
            out_prob = torch.tensor(out_prob)
        probs = torch.softmax(out_prob.squeeze(0), dim=-1)
        confidence = float(probs.max().item())

        label = text_lab[0] if isinstance(text_lab, (list, tuple)) else text_lab
        label = str(label).lower()

        quote_data = get_quote_for_emotion(label)

        return jsonify({
            "emotion": label,
            "confidence": confidence,
            "quote": quote_data
        })

    finally:
        for path in [tmp_in.name, tmp_norm]:
            try: os.remove(path)
            except: pass

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
