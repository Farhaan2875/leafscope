"""Load a trained checkpoint and diagnose a single RGB image (with Grad-CAM)."""
import time

import cv2
import numpy as np
import torch

import config
import data
import gradcam
import models


def best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Predictor:
    def __init__(self, name, device=None):
        path = config.CKPT_DIR / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        self.name = name
        self.device = device or best_device()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.class_names = ckpt["class_names"]
        self.img_size = ckpt["img_size"]
        self.meta = {"train_seconds": ckpt["train_seconds"], "test": ckpt["test_summary"]}
        self.model = models.build_model(name, len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.layer = models.gradcam_layer(self.model, name)
        self.transform = data.eval_transform(self.img_size)

    def _prepare(self, rgb):
        return self.transform(data.rgb_to_tensor(rgb)).unsqueeze(0).to(self.device)

    def predict(self, rgb: np.ndarray, topk=3, with_cam=True):
        """rgb: uint8 HxWx3.  Returns top-k predictions and a Grad-CAM map (img_size x img_size)."""
        x = self._prepare(rgb)
        t0 = time.perf_counter()
        if with_cam:
            with gradcam.GradCAM(self.model, self.layer) as cam:
                heat, logits = cam(x)
            heat = heat[0]
        else:
            with torch.no_grad():
                logits = self.model(x)
            heat = None
        ms = (time.perf_counter() - t0) * 1000
        probs = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
        order = np.argsort(-probs)[:topk]
        return {
            "top": [{"index": int(i), "class_name": self.class_names[i], "prob": float(probs[i])} for i in order],
            "cam": heat, "ms": ms,
        }
