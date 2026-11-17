"""Grad-CAM (Selvaraju et al., 2017) implemented from scratch with PyTorch hooks."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Use as a context manager so the hooks are always removed:

        with GradCAM(model, target_layer) as cam:
            heatmaps, logits = cam(batch)          # heatmaps: (B, H, W) in [0, 1]
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.acts = None
        self.grads = None
        self._handle = None

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._on_forward)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
        self.acts = self.grads = None

    def _on_forward(self, module, inputs, output):
        self.acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)     # guarantees a graph even if weights are frozen
            logits = self.model(x)
            if class_idx is None:
                idx = logits.argmax(dim=1)
            elif isinstance(class_idx, int):
                idx = torch.full((x.shape[0],), class_idx, dtype=torch.long, device=x.device)
            else:
                idx = class_idx
            score = logits.gather(1, idx.view(-1, 1)).sum()
            score.backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # global-average-pooled gradients
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # weighted sum of feature maps
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo = flat.min(dim=1)[0].view(-1, 1, 1)
        hi = flat.max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()


def colorize(cam01: np.ndarray, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """(H, W) float in [0,1] -> BGR heat image."""
    return cv2.applyColorMap((np.clip(cam01, 0, 1) * 255).astype(np.uint8), colormap)


def overlay(bgr: np.ndarray, cam01: np.ndarray, alpha=0.55, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Blend a heat map onto a BGR image; the map is resized to the image first."""
    cam = cv2.resize(cam01.astype(np.float32), (bgr.shape[1], bgr.shape[0]))
    heat = colorize(cam, colormap)
    weight = (alpha * np.clip(cam, 0, 1) ** 0.6)[..., None]   # weak attention stays see-through
    return (bgr * (1 - weight) + heat * weight).astype(np.uint8)
