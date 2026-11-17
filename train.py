"""Two-phase transfer learning on PlantVillage.

Phase 1  backbone frozen, only the new classification head is trained
Phase 2  backbone unfrozen, layer-wise learning rates + one-cycle LR schedule
Early stopping watches validation macro-F1.  The best checkpoint is saved.

    python train.py --model resnet50
    python train.py --model mobilenet_v2
    python train.py --model resnet50 --quick          # laptop-CPU sanity run
"""
import argparse
import contextlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import config
import data
import metrics
import models


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, n_classes, criterion=None):
    model.eval()
    preds, labels, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * y.size(0)
        n += y.size(0)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    cm = metrics.confusion_matrix(np.concatenate(labels), np.concatenate(preds), n_classes)
    res = metrics.summarize(cm)
    res["loss"] = loss_sum / max(n, 1)
    return res


def run_epoch(model, name, loader, optimizer, scheduler, criterion, device, scaler, frozen):
    models.set_train_mode(model, name, backbone_frozen=frozen)
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        amp = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if scaler.is_enabled() else contextlib.nullcontext())
        with amp:
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def plot_history(history, name, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val")
    ax[1].set_title("Accuracy")
    ax[2].plot(ep, [h["val_macro_f1"] for h in history], color="tab:green", label="val macro-F1")
    ax[2].set_title("Validation macro-F1")
    boundary = next((h["epoch"] for h in history if h["phase"] == 2), None)
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        if boundary:
            a.axvline(boundary - 0.5, color="grey", ls="--", lw=1)
        a.legend()
    fig.suptitle(f"{config.MODEL_LABELS[name]} - dashed line = backbone unfrozen")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=config.MODEL_NAMES, required=True)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=2e-4, help="top backbone block; earlier blocks decay")
    ap.add_argument("--lr-decay", type=float, default=0.6)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="1 head epoch + 2 fine-tune epochs, 128 px")
    a = ap.parse_args()
    if a.quick:
        a.epochs_head, a.epochs_ft, a.img_size, a.batch_size = 1, 2, 128, 32

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = pick_device()
    print(f"Device: {device} | model: {a.model}")

    train_dl, val_dl, test_dl, class_names = data.make_loaders(a.batch_size, a.workers, a.img_size)
    n_classes = len(class_names)
    print(f"{n_classes} classes | train {len(train_dl.dataset)} | val {len(val_dl.dataset)} "
          f"| test {len(test_dl.dataset)}")

    model = models.build_model(a.model, n_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best_f1, best_state, bad = [], -1.0, None, 0
    t_start = time.time()
    epoch = 0

    # ---------------- Phase 1: frozen backbone, train the head ----------------
    models.set_backbone_trainable(model, a.model, False)
    opt = torch.optim.AdamW(models.head_module(model, a.model).parameters(), lr=a.lr_head, weight_decay=1e-4)
    for _ in range(a.epochs_head):
        epoch += 1
        t0 = time.time()
        tl, ta = run_epoch(model, a.model, train_dl, opt, None, criterion, device, scaler, frozen=True)
        v = evaluate(model, val_dl, device, n_classes, criterion)
        history.append({"epoch": epoch, "phase": 1, "train_loss": tl, "train_acc": ta,
                        "val_loss": v["loss"], "val_acc": v["accuracy"], "val_macro_f1": v["macro_f1"],
                        "seconds": time.time() - t0})
        print(f"[head {epoch:2d}] train {tl:.4f}/{ta:.4f} | val {v['loss']:.4f}/{v['accuracy']:.4f} "
              f"F1 {v['macro_f1']:.4f} | {time.time() - t0:.0f}s")
        if v["macro_f1"] > best_f1:
            best_f1 = v["macro_f1"]
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    # ------------- Phase 2: unfreeze, layer-wise LR + one-cycle schedule -------------
    models.set_backbone_trainable(model, a.model, True)
    groups = models.param_groups(model, a.model, a.lr_head * 0.2, a.lr_backbone, a.lr_decay)
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], epochs=a.epochs_ft,
        steps_per_epoch=len(train_dl), pct_start=0.15)
    for _ in range(a.epochs_ft):
        epoch += 1
        t0 = time.time()
        tl, ta = run_epoch(model, a.model, train_dl, opt, sched, criterion, device, scaler, frozen=False)
        v = evaluate(model, val_dl, device, n_classes, criterion)
        history.append({"epoch": epoch, "phase": 2, "train_loss": tl, "train_acc": ta,
                        "val_loss": v["loss"], "val_acc": v["accuracy"], "val_macro_f1": v["macro_f1"],
                        "seconds": time.time() - t0})
        print(f"[ft   {epoch:2d}] train {tl:.4f}/{ta:.4f} | val {v['loss']:.4f}/{v['accuracy']:.4f} "
              f"F1 {v['macro_f1']:.4f} | {time.time() - t0:.0f}s")
        if v["macro_f1"] > best_f1 + 1e-4:
            best_f1, bad = v["macro_f1"], 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping: no validation F1 gain for {a.patience} epochs.")
                break

    train_seconds = time.time() - t_start
    model.load_state_dict(best_state)
    test = evaluate(model, test_dl, device, n_classes)
    print(f"\nBest val macro-F1 {best_f1:.4f} | TEST accuracy {test['accuracy']:.4f} "
          f"macro-F1 {test['macro_f1']:.4f} | train time {train_seconds / 60:.1f} min")

    ckpt = {
        "model_name": a.model, "class_names": class_names, "img_size": a.img_size,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "history": history, "train_seconds": float(train_seconds),
        "best_val_macro_f1": float(best_f1),
        "test_summary": {k: float(v) for k, v in test.items()},
        "args": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(a).items()},
        "device": str(device),
    }
    torch.save(ckpt, config.CKPT_DIR / f"{a.model}.pt")
    (config.OUT_DIR / f"history_{a.model}.json").write_text(json.dumps(history, indent=2))
    plot_history(history, a.model, config.FIG_DIR / f"training_{a.model}.png")
    print(f"Saved {config.CKPT_DIR / (a.model + '.pt')}")


if __name__ == "__main__":
    main()
