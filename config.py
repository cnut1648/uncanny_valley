from pathlib import Path

# project dir
BASE_DIR: Path = Path(__file__)
# where the datamodules is
DATASET_DIR: Path = BASE_DIR / "folklores" / "raw datamodules"

# used in S-BERT
SBERT_MODEL_NAME: str = "paraphrase-distilroberta-base-v1"
