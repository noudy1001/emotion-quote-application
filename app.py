import os
import tempfile
import torch
import torchaudio
from torchaudio.transforms import Resample
from flask import Flask, request, jsonify
import requests
from speechbrain.pretrained import EncoderClassifier

app = Flask(__name__)

print("Loading model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir=None
)
print("Model loaded.")


# Map emotions → tags for quotes
MOOD_TO_TAGS = {
    "angry": "patience|forgiveness",
    "happy": "joy|happiness|inspiration",
    "sad": "hope|perseverance|resilience",
    "neutral": "life|wisdom|motivation",
}

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

@app.route("/infer/audio", methods=["POST"])
def infer_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f = request.files["audio"]
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.save(tmp_in.name)
    tmp_in.close()

    try:
        # Convert to 16k mono
        tmp_norm = ensure_16k_mono(tmp_in.name)

        # Inference
        out_prob, score, index, text_lab = classifier.classify_file(tmp_norm)
        if not isinstance(out_prob, torch.Tensor):
            out_prob = torch.tensor(out_prob)
        probs = torch.softmax(out_prob.squeeze(0), dim=-1)
        confidence = float(probs.max().item())

        label = text_lab[0] if isinstance(text_lab, (list, tuple)) else text_lab
        label = str(label).lower()

        tags = MOOD_TO_TAGS.get(label, "life")

        # Get a quote
        qresp = requests.get("https://api.quotable.io/random", params={"tags": tags})
        qdata = qresp.json() if qresp.ok else {}

        return jsonify({
            "emotion": label,
            "confidence": confidence,
            "quote": {
                "text": qdata.get("content", ""),
                "author": qdata.get("author", ""),
                "tags": qdata.get("tags", [])
            }
        })

    finally:
        try: os.remove(tmp_in.name)
        except: pass
        try: os.remove(tmp_norm)
        except: pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

