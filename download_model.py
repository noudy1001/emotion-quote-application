from speechbrain.pretrained import EncoderClassifier

EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion"
)

print("✅ Model downloaded into pretrained_models/emotion")
