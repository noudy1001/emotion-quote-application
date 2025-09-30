from speechbrain.inference import EncoderClassifier

print("Loading model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir=None
)
print("Model loaded.")
