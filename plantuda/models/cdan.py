"""CDAN+E conditioning: the multilinear map and the entropy weight."""

from __future__ import annotations

import torch


def multilinear_map(feat: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Outer product of features and predictions, flattened.

    Args:
        feat: ``(N, d)`` features.
        pred: ``(N, C)`` softmax predictions.

    Returns:
        ``(N, d * C)``.

    Do not detach `feat` here. The adversarial signal reaches the feature
    extractor through this tensor; detaching it makes training fail silently —
    the loss curves look plausible and target accuracy collapses to chance.
    The caller detaches `pred` instead, following the official CDAN.
    """
    n = feat.size(0)
    return torch.bmm(feat.unsqueeze(2), pred.unsqueeze(1)).view(n, -1)


def cdan_entropy_weight(probs: torch.Tensor) -> torch.Tensor:
    """The ``+E`` of CDAN+E: ``w = 1 + exp(-H(p))``.

    Confident samples approach 2, uncertain ones approach 1, so every sample
    still contributes but confident ones dominate the domain loss. Detached: it
    is a coefficient, not a path for gradients.
    """
    entropy = -(probs * torch.log(probs + 1e-5)).sum(dim=1)
    return (1.0 + torch.exp(-entropy)).detach()
