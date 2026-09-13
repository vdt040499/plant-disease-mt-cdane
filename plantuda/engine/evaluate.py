"""Evaluation on the held-out field test set."""

from __future__ import annotations

import torch


@torch.no_grad()
def evaluate_accuracy(phi, g, loader, device) -> float:
    """Top-1 accuracy of ``g(phi(x))`` over `loader`.

    Args:
        phi: Feature extractor (student or teacher).
        g: Classifier matching `phi`.
        loader: Labelled loader — in this project, only the field test split.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    phi.eval()
    g.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (g(phi(images)).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total
