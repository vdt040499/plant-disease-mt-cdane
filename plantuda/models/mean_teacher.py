"""Mean Teacher: ramp schedule, consistency loss and EMA update.

Ported from CuriousAI/mean-teacher and milsever/plant-pathology. Two details
come from those sources rather than from any paper, and both are easy to get
wrong:

1. The reference implementation does not detach the teacher logits; it freezes
   the teacher at construction instead. This port does both, which is
   equivalent and harder to break.
2. `update_ema` walks ``.parameters()`` only — BatchNorm buffers are not
   averaged. The teacher's buffers stay current because the teacher is kept in
   ``.train()`` mode, so its own forward passes update them. ResNet-18 has 20
   BatchNorm layers; leaving the teacher in ``.eval()`` makes it useless.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plantuda.models.networks import Classifier, FeatureExtractor


def sigmoid_rampup(x: float, length: float) -> float:
    """Laine and Aila's sigmoid ramp-up.

    Returns a factor in ``(0, 1]``. At ``x = 0`` it is ``exp(-5) ~ 0.0067``,
    not 0 — the caller gates the true zero.
    """
    if length == 0:
        return 1.0
    x = float(np.clip(x, 0.0, length)) / length
    return float(np.exp(-5.0 * (1.0 - x) ** 2))


def consistency_weight(
    epoch: int, cons_max: float = 9.0, start: int = 100, ramp_len: int = 50
) -> float:
    """``w(t)``: exactly zero before `start`, then ramping to `cons_max`."""
    if epoch < start:
        return 0.0
    return cons_max * sigmoid_rampup(epoch - start, ramp_len)


def softmax_mse_loss(input_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """MSE between two softmax distributions, summed over the batch.

    Kept in the reference implementation's form: the sum is divided by the
    number of classes here and by the batch size at the call site.
    """
    if input_logits.size() != target_logits.size():
        raise ValueError("student and teacher logits must have the same shape")
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction="sum") / num_classes


def create_teacher(
    student_phi: nn.Module, student_g: nn.Module, num_classes: int, device
) -> Tuple[FeatureExtractor, Classifier]:
    """Build a frozen copy of the student.

    Call this inside ``torch.random.fork_rng(devices=[])``. It constructs two
    modules and therefore consumes RNG; without the fork, every source
    augmentation afterwards would be out of phase with the baseline arm and the
    paired comparison would silently break.

    `pretrained=False` because the weights are overwritten immediately.
    """
    phi_t = FeatureExtractor(pretrained=False).to(device)
    g_t = Classifier(phi_t.out_dim, num_classes).to(device)
    phi_t.load_state_dict(student_phi.state_dict())
    g_t.load_state_dict(student_g.state_dict())
    for p in list(phi_t.parameters()) + list(g_t.parameters()):
        p.requires_grad_(False)
    return phi_t, g_t


def update_ema(
    student_mods: Sequence[nn.Module],
    teacher_mods: Sequence[nn.Module],
    global_step: int,
    ema_decay: float = 0.99,
) -> float:
    """Exponential moving average of the student's parameters.

    Alpha rises from 0 to `ema_decay`, which makes the early teacher a true
    running average instead of a badly-biased exponential one.

    Returns the alpha used, for logging.
    """
    alpha = min(1 - 1 / (global_step + 1), ema_decay)
    with torch.no_grad():
        for student, teacher in zip(student_mods, teacher_mods):
            for ema_p, p in zip(teacher.parameters(), student.parameters()):
                ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)
    return alpha
