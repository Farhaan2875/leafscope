"""ResNet-50 and MobileNetV2 with ImageNet weights, plus the helpers needed for
transfer learning: freezing / unfreezing and layer-wise learning rates."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif name == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.last_channel, num_classes))
    else:
        raise ValueError(f"unknown model {name}")
    return m


def head_module(model, name):
    return model.fc if name == "resnet50" else model.classifier


def backbone_groups(model, name):
    """Backbone split into blocks, ordered from the input side to the output side."""
    if name == "resnet50":
        return [nn.ModuleList([model.conv1, model.bn1]), model.layer1, model.layer2,
                model.layer3, model.layer4]
    f = model.features
    return [f[0:4], f[4:7], f[7:14], f[14:19]]


def set_backbone_trainable(model, name, trainable: bool):
    for g in backbone_groups(model, name):
        for p in g.parameters():
            p.requires_grad = trainable


def set_train_mode(model, name, backbone_frozen: bool):
    """model.train(), but keep the frozen backbone's BatchNorm statistics fixed."""
    model.train()
    if backbone_frozen:
        for g in backbone_groups(model, name):
            g.eval()


def param_groups(model, name, head_lr, backbone_lr, decay=0.6, weight_decay=1e-4):
    """Layer-wise learning rates: the head gets `head_lr`; the backbone block nearest the
    head gets `backbone_lr`; each earlier block is multiplied by `decay` again."""
    groups = [{"params": list(head_module(model, name).parameters()), "lr": head_lr,
               "weight_decay": weight_decay}]
    blocks = backbone_groups(model, name)
    for depth, g in enumerate(reversed(blocks)):
        groups.append({"params": list(g.parameters()), "lr": backbone_lr * (decay ** depth),
                       "weight_decay": weight_decay})
    return groups


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024


def gradcam_layer(model, name):
    """Last convolutional feature block, where spatial detail and class semantics meet."""
    return model.layer4[-1] if name == "resnet50" else model.features[-1]
