"""Evaluate both trained models on the held-out test split and write the comparison.

Outputs (outputs/):
    comparison.json / comparison.csv       headline table used by the app's "Trials" page
    figures/confusion_<model>.png          row-normalised confusion matrices
    figures/comparison.png                 accuracy / F1 / size / latency / robustness bars
    figures/gradcam_gallery_<model>.png    Grad-CAM on correct vs misclassified leaves
    per_class_<model>.csv                  precision / recall / F1 for each class

    python evaluate.py                     # all available checkpoints
    python evaluate.py --robust-n 600 --cam-n 150
"""
import argparse
import csv
import json
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import data
import gradcam
import inference
import metrics
import models
import severity


@torch.no_grad()
def predict_loader(pred, loader):
    ys, ps = [], []
    for x, y in loader:
        out = pred.model(x.to(pred.device))
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def make_loader(split, items, preprocess, size, workers, bs=64):
    ds = data.LeafDataset(split["root"], items, data.eval_transform(size), preprocess=preprocess)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)


def latency_ms(pred, runs=40):
    x = torch.randn(1, 3, pred.img_size, pred.img_size, device=pred.device)
    with torch.no_grad():
        for _ in range(5):
            pred.model(x)
        if pred.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            pred.model(x)
        if pred.device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def plot_confusion(cm, class_names, title, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short = [c.replace("___", " - ").replace("_", " ")[:34] for c in class_names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gradcam_study(pred, split, class_names, n, gallery_path):
    """Grad-CAM focus on lesions (vs healthy tissue / background) over diseased test leaves,
    plus a gallery of correct and misclassified examples."""
    rng = np.random.RandomState(config.SEED)
    items = [it for it in split["test"] if "healthy" not in class_names[it[1]].lower()]
    pick = rng.choice(len(items), size=min(n, len(items)), replace=False)
    ds = data.LeafDataset(split["root"], [items[i] for i in pick], pred.transform)
    ratios, on_lesion, on_leaf, on_bg, rows = [], [], [], [], []
    for i in range(len(ds)):
        rgb = ds.read_rgb(i)
        label = ds.items[i][1]
        r = pred.predict(rgb, topk=1, with_cam=True)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        sev = severity.estimate_severity(bgr)
        split_ = severity.attention_split(r["cam"], sev["leaf_mask"], sev["lesion_mask"])
        on_lesion.append(split_["lesion"]); on_leaf.append(split_["leaf"]); on_bg.append(split_["background"])
        fr = severity.lesion_focus_ratio(r["cam"], sev["leaf_mask"], sev["lesion_mask"])
        if fr is not None:
            ratios.append(fr)
        rows.append((bgr, r["cam"], label, r["top"][0]["index"]))

    summary = {
        "n": len(rows),
        "attention_on_lesion_pct": float(np.mean(on_lesion)),
        "attention_on_healthy_leaf_pct": float(np.mean(on_leaf)),
        "attention_on_background_pct": float(np.mean(on_bg)),
        "lesion_focus_ratio_median": float(np.median(ratios)) if ratios else None,
    }

    correct = [r for r in rows if r[2] == r[3]][:4]
    wrong = [r for r in rows if r[2] != r[3]][:4]
    sel = [("correct", r) for r in correct] + [("misclassified", r) for r in wrong]
    if sel:
        cols = 4
        nrows = int(np.ceil(len(sel) / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(3.4 * cols, 3.6 * nrows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (kind, (bgr, cam, y, p)) in zip(axes.ravel(), sel):
            ov = gradcam.overlay(cv2.resize(bgr, (pred.img_size, pred.img_size)), cam)
            ax.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
            t = class_names[y].replace("___", " / ").replace("_", " ")[:30]
            q = class_names[p].replace("___", " / ").replace("_", " ")[:30]
            ax.set_title(f"{kind}\ntrue: {t}\npred: {q}", fontsize=7,
                         color="#2d6a2d" if kind == "correct" else "#b03020")
        fig.suptitle(f"Grad-CAM - {config.MODEL_LABELS[pred.name]}")
        fig.tight_layout()
        fig.savefig(gallery_path, dpi=120)
        plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--robust-n", type=int, default=800, help="test images used for lighting tests")
    ap.add_argument("--cam-n", type=int, default=120, help="diseased test images for the Grad-CAM study")
    a = ap.parse_args()

    split = data.load_splits()
    class_names = split["class_names"]
    avail = [m for m in config.MODEL_NAMES if (config.CKPT_DIR / f"{m}.pt").exists()]
    if not avail:
        raise SystemExit("No checkpoints in outputs/checkpoints - run train.py first.")

    rng = np.random.RandomState(config.SEED)
    sub_idx = rng.choice(len(split["test"]), size=min(a.robust_n, len(split["test"])), replace=False)
    sub_items = [split["test"][i] for i in sub_idx]

    report = {"classes": len(class_names), "test_images": len(split["test"]), "models": {}}
    for name in avail:
        print(f"\n=== {config.MODEL_LABELS[name]} ===")
        pred = inference.Predictor(name)
        n_cls = len(class_names)

        y, p = predict_loader(pred, make_loader(split, split["test"], None, pred.img_size, a.workers))
        cm = metrics.confusion_matrix(y, p, n_cls)
        summ = metrics.summarize(cm)
        prec, rec, f1, sup = metrics.prf(cm)
        print(f"test accuracy {summ['accuracy']:.4f} | macro-F1 {summ['macro_f1']:.4f}")

        with open(config.OUT_DIR / f"per_class_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "precision", "recall", "f1", "support"])
            for i, c in enumerate(class_names):
                w.writerow([c, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])
        plot_confusion(cm, class_names, f"{config.MODEL_LABELS[name]} - test confusion matrix",
                       config.FIG_DIR / f"confusion_{name}.png")

        robust = {}
        for vname, fn in data.LIGHTING_VARIANTS.items():
            yy, pp = predict_loader(pred, make_loader(split, sub_items, fn, pred.img_size, a.workers))
            s = metrics.summarize(metrics.confusion_matrix(yy, pp, n_cls))
            robust[vname] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}
            print(f"  lighting [{vname:12s}] acc {s['accuracy']:.4f}  macro-F1 {s['macro_f1']:.4f}")

        print("  Grad-CAM study ...")
        cam_sum = gradcam_study(pred, split, class_names, a.cam_n,
                                config.FIG_DIR / f"gradcam_gallery_{name}.png")

        report["models"][name] = {
            "label": config.MODEL_LABELS[name],
            "params_million": models.count_params(pred.model) / 1e6,
            "size_mb": models.size_mb(pred.model),
            "latency_ms_batch1": latency_ms(pred),
            "latency_device": str(pred.device),
            "train_minutes": pred.meta["train_seconds"] / 60,
            "test": summ,
            "lighting": robust,
            "gradcam": cam_sum,
        }

    config.COMPARISON_JSON.write_text(json.dumps(report, indent=2))
    with open(config.OUT_DIR / "comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "accuracy", "macro_f1", "weighted_f1", "params_M", "size_MB",
                    "latency_ms", "train_min", "attn_on_lesion_pct", "attn_on_background_pct"])
        for name, m in report["models"].items():
            w.writerow([m["label"], f"{m['test']['accuracy']:.4f}", f"{m['test']['macro_f1']:.4f}",
                        f"{m['test']['weighted_f1']:.4f}", f"{m['params_million']:.2f}",
                        f"{m['size_mb']:.1f}", f"{m['latency_ms_batch1']:.1f}",
                        f"{m['train_minutes']:.1f}", f"{m['gradcam']['attention_on_lesion_pct']:.1f}",
                        f"{m['gradcam']['attention_on_background_pct']:.1f}"])

    # ----- comparison figure -----
    names = list(report["models"])
    labels = [report["models"][n]["label"] for n in names]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    x = np.arange(len(names))
    ax[0].bar(x, [report["models"][n]["test"]["macro_f1"] for n in names], color=["#3f5b3a", "#b5651d"][:len(names)])
    ax[0].set_title("Test macro-F1")
    lo = min(report["models"][n]["test"]["macro_f1"] for n in names)
    ax[0].set_ylim(max(0.0, min(0.8, lo - 0.05)), 1.0)
    ax[1].bar(x, [report["models"][n]["size_mb"] for n in names], color=["#3f5b3a", "#b5651d"][:len(names)])
    ax[1].set_title("Model size (MB)")
    ax[2].bar(x, [report["models"][n]["latency_ms_batch1"] for n in names], color=["#3f5b3a", "#b5651d"][:len(names)])
    ax[2].set_title("Latency, batch 1 (ms)")
    variants = list(data.LIGHTING_VARIANTS)
    width = 0.8 / len(names)
    for k, n in enumerate(names):
        ax[3].bar(np.arange(len(variants)) + k * width,
                  [report["models"][n]["lighting"][v]["macro_f1"] for v in variants], width,
                  label=report["models"][n]["label"], color=["#3f5b3a", "#b5651d"][k])
    ax[3].set_xticks(np.arange(len(variants)) + width * (len(names) - 1) / 2)
    ax[3].set_xticklabels(variants, rotation=20, fontsize=8)
    ax[3].set_title("Macro-F1 under lighting changes")
    ax[3].legend(fontsize=8)
    for a_ in ax[:3]:
        a_.set_xticks(x)
        a_.set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "comparison.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved {config.COMPARISON_JSON}")


if __name__ == "__main__":
    main()
