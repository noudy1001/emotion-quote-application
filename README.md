# 🎙️ Emotion + Quote Flask API

This app:
1. Accepts an audio file (`POST /infer/audio` with `multipart/form-data`).
2. Runs [SpeechBrain wav2vec2 IEMOCAP model](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP).
3. Predicts emotion + confidence.
4. Fetches a motivational quote from [Quotable API](https://github.com/lukePeavey/quotable).

---

## 🚀 Run locally

```bash
# clone repo
git clone https://github.com/<your-username>/emotion-quote-app.git
cd emotion-quote-app

# setup venv
python -m venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt

# download model
python download_model.py

# run server
python app.py
