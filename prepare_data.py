"""Download PlantVillage and create reproducible stratified train/val/test splits.

Examples
    python prepare_data.py --download                      # git sparse-clone raw/color (~1 GB)
    python prepare_data.py --data-dir D:/datasets/color    # use a folder you already have
    python prepare_data.py --download --max-per-class 120  # tiny subset for a laptop CPU

The dataset is the public PlantVillage collection (Hughes & Salathe, 2015; Mohanty et al.,
2016): ~54,300 labelled leaf photos in 38 crop/disease classes.  `raw/color` holds the
original, non-augmented photographs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import config

REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset"
EXTS = {".jpg", ".jpeg", ".png"}


def download():
    if (config.DEFAULT_IMAGE_ROOT).exists() and any(config.DEFAULT_IMAGE_ROOT.iterdir()):
        print(f"Dataset already present at {config.DEFAULT_IMAGE_ROOT}")
        return config.DEFAULT_IMAGE_ROOT
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Cloning PlantVillage (only raw/color is checked out; this can take several minutes)...")
    cmds = [
        ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
         REPO_URL, str(config.REPO_DIR)],
        ["git", "-C", str(config.REPO_DIR), "sparse-checkout", "set", "--cone", "raw/color"],
        ["git", "-C", str(config.REPO_DIR), "checkout"],
    ]
    for c in cmds:
        r = subprocess.run(c)
        if r.returncode != 0:
            sys.exit("git failed. Install git, or download the dataset manually (see README) "
                     "and pass --data-dir.")
    return config.DEFAULT_IMAGE_ROOT


def scan(root: Path):
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    items = {}
    for c in classes:
        files = sorted(f.name for f in (root / c).iterdir() if f.suffix.lower() in EXTS)
        if files:
            items[c] = files
    if not items:
        sys.exit(f"No class folders with images found under {root}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="git sparse-clone PlantVillage raw/color")
    ap.add_argument("--data-dir", type=Path, help="folder that contains one sub-folder per class")
    ap.add_argument("--max-per-class", type=int, default=0, help="cap images per class (0 = all)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()

    if a.data_dir:
        root = a.data_dir.resolve()
    elif a.download or not config.SPLIT_FILE.exists():
        root = download()
    else:
        sys.exit("Nothing to do. Use --download or --data-dir.")

    items = scan(root)
    class_names = list(items.keys())
    rng = np.random.RandomState(a.seed)
    split = {"train": [], "val": [], "test": []}
    counts = {}
    for label, c in enumerate(class_names):
        files = np.array(items[c])
        rng.shuffle(files)
        if a.max_per_class:
            files = files[: a.max_per_class]
        n = len(files)
        n_test = max(1, int(round(n * a.test)))
        n_val = max(1, int(round(n * a.val)))
        parts = {"test": files[:n_test], "val": files[n_test:n_test + n_val],
                 "train": files[n_test + n_val:]}
        counts[c] = {k: len(v) for k, v in parts.items()}
        for k, v in parts.items():
            split[k] += [[f"{c}/{f}", label] for f in v]
    for k in split:
        rng.shuffle(split[k])

    out = {"root": str(root), "class_names": class_names, "seed": a.seed,
           "counts": counts, **split}
    config.SPLIT_FILE.write_text(json.dumps(out))

    print(f"\nImage root : {root}")
    print(f"Classes    : {len(class_names)}")
    print(f"Split      : train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    print(f"Saved      : {config.SPLIT_FILE}")
    print("\nPer-class (train/val/test):")
    for c in class_names:
        k = counts[c]
        print(f"  {c:58s} {k['train']:5d} {k['val']:4d} {k['test']:4d}")


if __name__ == "__main__":
    main()
