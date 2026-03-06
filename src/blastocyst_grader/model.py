from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms


class MultiHeadBlastocystNet(nn.Module):
    """Single backbone with three classification heads (Expansion/ICM/TE)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.expansion_head = nn.Linear(in_features, 6)
        self.icm_head = nn.Linear(in_features, 4)
        self.te_head = nn.Linear(in_features, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "expansion": self.expansion_head(features),
            "icm": self.icm_head(features),
            "te": self.te_head(features),
        }


def build_transforms(image_size: int = 224, train: bool = False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
