"""Classical-vision severity estimation (OpenCV + NumPy only).

PlantVillage has no severity annotations, so severity here is an *image-based estimate*:
    severity % = lesion pixels / leaf pixels
Pipeline
    1. segment the leaf from its background (colour distance to the border colour + Otsu,
       GrabCut as a fallback for cluttered backgrounds)
    2. estimate the leaf's own healthy-green reference colour in CIE-Lab
    3. mark pixels whose chroma departs strongly from that reference (browning,
       yellowing, necrosis) as lesion tissue
    4. map the percentage onto a five-step grade

The estimate is a screening aid, not a pathologist's score.
"""
from __future__ import annotations

import cv2
import numpy as np

GRADES = [
    (1.0, 0, "Trace"),         # < 1 %
    (10.0, 1, "Mild"),         # 1 - 10 %
    (25.0, 2, "Moderate"),     # 10 - 25 %
    (50.0, 3, "Severe"),       # 25 - 50 %
    (101.0, 4, "Critical"),    # > 50 %
]


def grade_for(percent: float):
    for upper, grade, label in GRADES:
        if percent < upper:
            return grade, label
    return 4, "Critical"


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = np.pad(mask, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    holes = (flood[1:-1, 1:-1] == 0)
    out = mask.copy()
    out[holes] = 1
    return out


def _grabcut(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mx, my = int(w * 0.04), int(h * 0.04)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def segment_leaf(bgr: np.ndarray) -> np.ndarray:
    """Binary (0/1) leaf mask the same size as `bgr`."""
    h, w = bgr.shape[:2]
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Soil, concrete, paper and cast shadows are nearly neutral (low chroma); leaf tissue,
    # even brown or yellow tissue, is not.  Chroma therefore separates leaf from shadow far
    # better than distance to the border colour does.
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    chroma8 = np.clip(chroma * 6, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(chroma8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dead tissue (brown, grey-brown) has little chroma, so the test above misses it. It does
    # differ from the background in hue, whereas a cast shadow keeps the background's hue and
    # only gets darker. Compare a*/b* (not L*) with the border colour to catch it.
    ring = max(3, int(min(h, w) * 0.04))
    border = np.concatenate([lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3),
                             lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3)])
    bg_ab = np.median(border[:, 1:], axis=0)
    dab = np.sqrt(((lab[..., 1:] - bg_ab) ** 2).sum(axis=2))
    dab8 = np.clip(dab * 8, 0, 255).astype(np.uint8)
    t_ab, _ = cv2.threshold(dab8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_ab = max(t_ab, 8 * 7.0)                                  # never below a 7-unit hue shift
    mask = np.maximum(mask, (dab8 > t_ab).astype(np.uint8))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = _fill_holes(_largest_component(mask))

    # A leaf is a compact blob. Grey necrotic tissue on its rim can look just like the background
    # and leave a notch; a large closing fills such notches so the dead area counts as leaf.
    kc = int(min(h, w) * 0.14) | 1
    padded = cv2.copyMakeBorder(mask, kc, kc, kc, kc, cv2.BORDER_CONSTANT, value=0)
    padded = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    mask = _fill_holes(_largest_component(padded[kc:-kc, kc:-kc].copy()))

    frac = mask.mean()
    if frac < 0.08 or frac > 0.97:          # chroma found no clean leaf: start from a frame rectangle
        gc = _fill_holes(_largest_component(_grabcut(bgr)))
        if 0.05 < gc.mean() < 0.97:
            mask = gc
        return mask

    return mask


def lesion_mask(bgr: np.ndarray, leaf: np.ndarray, threshold: float = 20.0):
    """Return (lesion_mask, reference_lab) restricted to the leaf."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    inner = cv2.erode(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if inner.sum() < 200:
        inner = leaf
    inside = inner.astype(bool)

    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    greenish = inside & (hue >= 30) & (hue <= 90) & (sat > 45) & (val > 40)
    if greenish.sum() < 0.05 * inside.sum():
        greenish = inside                    # no clear green: use the whole leaf as reference
    ref = np.median(lab[greenish], axis=0)

    dl = (lab[..., 0] - ref[0]) * 0.35       # down-weight lightness so veins/shade are not lesions
    da = lab[..., 1] - ref[1]
    db = lab[..., 2] - ref[2]
    delta = np.sqrt(dl ** 2 + da ** 2 + db ** 2)

    # dark necrotic tissue is lesion even if its chroma is low
    dark = val < 0.45 * np.median(val[greenish])
    less_green = da > 14.0                      # grey/brown tissue: a* climbs towards neutral/red
    les = ((delta > threshold) | dark | less_green) & leaf.astype(bool)
    les = les.astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    les = cv2.morphologyEx(les, cv2.MORPH_OPEN, k)
    les = cv2.morphologyEx(les, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # drop specks smaller than 0.15 % of the leaf (leaf texture, glare, mite stippling)
    min_area = max(6, int(0.0015 * leaf.sum()))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(les, connectivity=8)
    keep = np.zeros_like(les)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep, ref


def estimate_severity(bgr: np.ndarray) -> dict:
    leaf = segment_leaf(bgr)
    les, ref = lesion_mask(bgr, leaf)
    leaf_px = int(leaf.sum())
    les_px = int(les.sum())
    pct = 100.0 * les_px / max(leaf_px, 1)
    grade, label = grade_for(pct)
    return {
        "leaf_mask": leaf, "lesion_mask": les,
        "percent": round(pct, 2), "grade": grade, "label": label,
        "leaf_fraction_of_frame": round(100.0 * leaf_px / leaf.size, 1),
        "reference_lab": [float(x) for x in ref],
    }


def attention_split(cam: np.ndarray, leaf: np.ndarray, lesion: np.ndarray) -> dict:
    """Share of Grad-CAM mass on lesions, on healthy leaf tissue and on background (sums to 100)."""
    cam = cv2.resize(cam.astype(np.float32), (leaf.shape[1], leaf.shape[0]))
    total = float(cam.sum())
    if total <= 1e-8:
        return {"lesion": 0.0, "leaf": 0.0, "background": 100.0}
    les = leaf.astype(bool) & lesion.astype(bool)
    healthy = leaf.astype(bool) & ~les
    bg = ~leaf.astype(bool)
    return {
        "lesion": round(100.0 * float(cam[les].sum()) / total, 1),
        "leaf": round(100.0 * float(cam[healthy].sum()) / total, 1),
        "background": round(100.0 * float(cam[bg].sum()) / total, 1),
    }


def lesion_focus_ratio(cam: np.ndarray, leaf: np.ndarray, lesion: np.ndarray) -> float | None:
    """Mean CAM on lesion pixels / mean CAM on healthy leaf pixels. >1 means attention
    concentrates on lesions. None when there are no lesion or no healthy pixels."""
    cam = cv2.resize(cam.astype(np.float32), (leaf.shape[1], leaf.shape[0]))
    les = leaf.astype(bool) & lesion.astype(bool)
    healthy = leaf.astype(bool) & ~les
    if les.sum() < 30 or healthy.sum() < 30:
        return None
    denom = float(cam[healthy].mean())
    if denom < 1e-6:
        return None
    return float(cam[les].mean()) / denom


def render_overlay(bgr: np.ndarray, sev: dict) -> np.ndarray:
    """Leaf outline (thin, light) + lesion regions tinted and outlined (vermilion)."""
    out = bgr.copy()
    les = sev["lesion_mask"]
    tint = np.zeros_like(out)
    tint[...] = (40, 80, 230)                          # BGR vermilion
    m = les.astype(bool)
    out[m] = cv2.addWeighted(out, 0.55, tint, 0.45, 0)[m]
    cnt_l, _ = cv2.findContours(sev["leaf_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt_l, -1, (235, 235, 225), 1, cv2.LINE_AA)
    cnt_s, _ = cv2.findContours(les, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt_s, -1, (30, 60, 200), 1, cv2.LINE_AA)
    return out
