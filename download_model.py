from speechbrain.inference import EncoderClassifier

# Just load from HuggingFace cache (default path)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir=None  # Do not try to copy to local folder
)

print("✅ Model loaded from HuggingFace cache")
