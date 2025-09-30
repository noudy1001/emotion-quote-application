from speechbrain.inference import EncoderClassifier
import os

# Force HuggingFace / SpeechBrain to avoid symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Create folder if it doesn't exist
os.makedirs("pretrained_models/emotion", exist_ok=True)

# Force copy strategy
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion",
    run_opts={"local_strategy": "copy"}  # THIS LINE FORCES COPY
)

print("✅ Model downloaded into pretrained_models/emotion")
