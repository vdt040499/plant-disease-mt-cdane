"""Gradient reversal and its schedule."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_lambda(step: int, max_steps: int, gamma: float = 10.0) -> float:
    """DANN's progressive lambda schedule, rising from ~0 to ~1.

    Classification is learned first; the adversarial pressure ramps in as
    training proceeds instead of destabilising the features from step one.
    """
    p = step / max_steps
    return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0


def domain_labels(n: int, is_source: bool, device) -> torch.Tensor:
    """Domain targets for the discriminator: source = 0, target = 1."""
    return torch.full((n,), 0 if is_source else 1, dtype=torch.long, device=device)


class GRL(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward.

    This is the whole trick: the discriminator is trained to separate domains
    while the feature extractor receives the negated gradient and is therefore
    trained to make them inseparable.
    """

    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.saved_tensors[0].item() * grad, None


class GradRevLayer(nn.Module):
    """Module wrapper around `GRL` with a lambda that the loop updates."""

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return GRL.apply(x, self.lam)

    def set_lam(self, lam: float) -> None:
        self.lam = lam
