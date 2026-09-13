"""Checkpointing.

A 300-epoch run on a hosted runtime will be interrupted, so resuming is a
requirement rather than a nicety. `step` is part of the state: it drives both
the lambda schedule and the EMA alpha, and restarting it at zero would rewind
both.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch


def save_checkpoint(
    path: str | os.PathLike,
    *,
    epoch: int,
    step: int,
    best_loss: float,
    best_state: Optional[dict],
    phi,
    g,
    disc,
    optimizer,
    scheduler,
    phi_teacher=None,
    g_teacher=None,
) -> None:
    """Write a resumable checkpoint.

    Args:
        epoch: The epoch that just finished.
        step: Global iteration counter.
        best_state: ``{'phi', 'G'}`` of the best model so far, or None.
        phi_teacher, g_teacher: Teacher modules; omit for the baseline arm.

    The teacher's ``state_dict`` includes its BatchNorm buffers. Those are not
    EMA-averaged but accumulated through forward passes, so dropping them would
    reset the teacher to ImageNet statistics on resume.
    """
    ckpt = {
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "best_state": best_state,
        "phi": phi.state_dict(),
        "G": g.state_dict(),
        "D": disc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if phi_teacher is not None:
        ckpt["phi_t"] = phi_teacher.state_dict()
        ckpt["G_t"] = g_teacher.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | os.PathLike,
    phi,
    g,
    disc,
    optimizer,
    scheduler,
    phi_teacher=None,
    g_teacher=None,
    device="cpu",
) -> Tuple[int, int, float, Optional[dict]]:
    """Restore state if the file exists.

    Returns:
        ``(start_epoch, step, best_loss, best_state)``, or
        ``(1, 0, inf, None)`` when there is nothing to resume from.
    """
    if not os.path.exists(path):
        return 1, 0, float("inf"), None

    ckpt = torch.load(path, map_location=device)
    phi.load_state_dict(ckpt["phi"])
    g.load_state_dict(ckpt["G"])
    disc.load_state_dict(ckpt["D"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if phi_teacher is not None and "phi_t" in ckpt:
        phi_teacher.load_state_dict(ckpt["phi_t"])
        g_teacher.load_state_dict(ckpt["G_t"])
    return ckpt["epoch"] + 1, ckpt["step"], ckpt["best_loss"], ckpt["best_state"]
