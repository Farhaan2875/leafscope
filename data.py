"""Dataset, augmentation and lighting-perturbation utilities (OpenCV + torchvision)."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import config


def load_splits():
    if not config.SPLIT_FILE.exists():
        raise SystemExit("data/splits.json not found - run `python prepare_data.py --download` first.")
    return json.loads(config.SPLIT_FILE.read_text())


def train_transform(size=config.IMG_SIZE):
    """Data augmentation aimed at real-world capture conditions: scale/crop, flips, rotation,
    strong brightness/contrast/saturation jitter (lighting) and occasional blur."""
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(25),
        v2.ColorJitter(brightness=0.45, contrast=0.35, saturation=0.30, hue=0.03),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform(size=config.IMG_SIZE):
    return v2.Compose([
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB -> uint8 3xHxW tensor."""
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()


class LeafDataset(Dataset):
    def __init__(self, root, items, transform, preprocess=None):
        self.root = Path(root)
        self.items = items
        self.transform = transform
        self.preprocess = preprocess          # optional fn(rgb uint8) -> rgb uint8 (lighting tests)

    def __len__(self):
        return len(self.items)

    def read_rgb(self, i):
        rel, _ = self.items[i]
        bgr = cv2.imread(str(self.root / rel), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {self.root / rel}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        rgb = self.read_rgb(i)
        if self.preprocess is not None:
            rgb = self.preprocess(rgb)
        return self.transform(rgb_to_tensor(rgb)), self.items[i][1]


def make_loaders(batch_size=64, workers=4, size=config.IMG_SIZE):
    sp = load_splits()
    root = sp["root"]
    pin = torch.cuda.is_available()
    kw = dict(num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    train = DataLoader(LeafDataset(root, sp["train"], train_transform(size)),
                       batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val = DataLoader(LeafDataset(root, sp["val"], eval_transform(size)),
                     batch_size=batch_size, shuffle=False, **kw)
    test = DataLoader(LeafDataset(root, sp["test"], eval_transform(size)),
                      batch_size=batch_size, shuffle=False, **kw)
    return train, val, test, sp["class_names"]


# --------------------------------------------------------------------------------------
# Lighting perturbations (OpenCV/NumPy) used by evaluate.py to probe field-like conditions
# --------------------------------------------------------------------------------------
def _gamma(rgb, g):
    lut = (255.0 * (np.arange(256) / 255.0) ** g).clip(0, 255).astype(np.uint8)
    return cv2.LUT(rgb, lut)


def dim_light(rgb):
    return (_gamma(rgb, 1.8).astype(np.float32) * 0.65).clip(0, 255).astype(np.uint8)


def overexposed(rgb):
    return (rgb.astype(np.float32) * 1.45 + 20).clip(0, 255).astype(np.uint8)


def hard_shadow(rgb):
    h, w = rgb.shape[:2]
    ramp = np.clip((np.arange(w, dtype=np.float32) - 0.35 * w) / (0.15 * w), 0, 1)
    gain = 1.0 - 0.55 * ramp                         # right side falls into shadow
    return (rgb.astype(np.float32) * gain[None, :, None]).clip(0, 255).astype(np.uint8)


def warm_cast(rgb):
    out = rgb.astype(np.float32)
    out[..., 0] *= 1.18
    out[..., 2] *= 0.82
    return out.clip(0, 255).astype(np.uint8)


LIGHTING_VARIANTS = {
    "original": None,
    "dim light": dim_light,
    "overexposed": overexposed,
    "hard shadow": hard_shadow,
    "warm cast": warm_cast,
}
