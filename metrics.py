"""Classification metrics in plain NumPy (accuracy, confusion matrix, precision/recall/F1)."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def prf(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    support = cm.sum(axis=1)
    return precision, recall, f1, support


def summarize(cm):
    precision, recall, f1, support = prf(cm)
    present = support > 0
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1)),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
    }
