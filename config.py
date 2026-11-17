"""Shared paths and constants for LeafScope."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPO_DIR = DATA_DIR / "PlantVillage-Dataset"          # where prepare_data.py clones the dataset
DEFAULT_IMAGE_ROOT = REPO_DIR / "raw" / "color"       # un-augmented colour images, one folder per class
SPLIT_FILE = DATA_DIR / "splits.json"

OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
COMPARISON_JSON = OUT_DIR / "comparison.json"

MODEL_NAMES = ("resnet50", "mobilenet_v2")
MODEL_LABELS = {"resnet50": "ResNet-50", "mobilenet_v2": "MobileNetV2"}

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42

for _d in (DATA_DIR, OUT_DIR, CKPT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
