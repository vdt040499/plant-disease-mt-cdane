"""Backbone, classifier and conditional discriminator.

Construction order matters. `FeatureExtractor`, `Classifier` and `CDANDisc` all
draw from the global RNG when they initialise their weights, so both arms build
them in this order: feature extractor, classifier, discriminator. `GradRevLayer`
holds no parameters and can be created anywhere.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


class FeatureExtractor(nn.Module):
    """ResNet-18 with the classification head removed. Output: 512-d.

    Args:
        pretrained: Load ImageNet weights. Pass False when the weights are
            about to be overwritten (the teacher, and evaluation-only copies) —
            it avoids a download and, more importantly, consumes the same
            amount of RNG either way.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        self.net = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        return self.net(x).flatten(1)


class Classifier(nn.Module):
    """Linear label classifier: 512 -> num_classes."""

    def __init__(self, in_dim: int = 512, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CDANDisc(nn.Module):
    """Conditional domain discriminator.

    Its input is the multilinear map of features and predictions, so the input
    dimension is ``in_dim * num_classes`` (512 * 3 = 1536 here), not `in_dim`.
    """

    def __init__(self, in_dim: int = 512, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)
