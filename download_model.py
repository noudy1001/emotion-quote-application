from speechbrain.inference import EncoderClassifier
import os

# Force HuggingFace / SpeechBrain to avoid symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY"] = "copy"

# Download and save the model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion"
)

print("✅ Model downloaded successfully!")
