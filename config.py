from pathlib import Path

# project dir
BASE_DIR: Path = Path("/content/drive/MyDrive/Creepy Data")
# where the data is
DATASET_DIR: Path = BASE_DIR / "folklores" / "raw data"

# used in S-BERT
SBERT_MODEL_NAME: str = "paraphrase-distilroberta-base-v1"
