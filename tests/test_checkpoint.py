"""Checkpoint round-trips, including the teacher state a resume depends on."""

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from plantuda.engine.checkpoint import load_checkpoint, save_checkpoint


def _bundle():
    phi = nn.Linear(4, 4)
    g = nn.Linear(4, 3)
    disc = nn.Linear(12, 2)
    phi_t = nn.Linear(4, 4)
    g_t = nn.Linear(4, 3)
    opt = optim.AdamW(list(phi.parameters()) + list(g.parameters()) + list(disc.parameters()))
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    return phi, g, disc, phi_t, g_t, opt, sch


def test_returns_fresh_start_when_file_missing(tmp_path):
    phi, g, disc, phi_t, g_t, opt, sch = _bundle()
    start_epoch, step, best_loss, best_state = load_checkpoint(
        tmp_path / "nope.pth", phi, g, disc, opt, sch, phi_t, g_t
    )
    assert start_epoch == 1
    assert step == 0
    assert math.isinf(best_loss)
    assert best_state is None


def test_roundtrip_restores_teacher_weights(tmp_path):
    phi, g, disc, phi_t, g_t, opt, sch = _bundle()
    with torch.no_grad():
        g_t.weight.fill_(7.0)
    path = tmp_path / "ck.pth"
    save_checkpoint(
        path, epoch=42, step=1234, best_loss=0.5, best_state={"phi": phi.state_dict()},
        phi=phi, g=g, disc=disc, optimizer=opt, scheduler=sch,
        phi_teacher=phi_t, g_teacher=g_t,
    )

    phi2, g2, disc2, phi_t2, g_t2, opt2, sch2 = _bundle()
    start_epoch, step, best_loss, best_state = load_checkpoint(
        path, phi2, g2, disc2, opt2, sch2, phi_t2, g_t2
    )
    assert start_epoch == 43          # resume at the NEXT epoch
    assert step == 1234
    assert best_loss == pytest.approx(0.5)
    assert best_state is not None
    assert torch.allclose(g_t2.weight, torch.full_like(g_t2.weight, 7.0))


def test_roundtrip_works_without_teacher(tmp_path):
    """The baseline arm has no teacher to store."""
    phi, g, disc, _, _, opt, sch = _bundle()
    path = tmp_path / "ck.pth"
    save_checkpoint(
        path, epoch=5, step=10, best_loss=1.0, best_state=None,
        phi=phi, g=g, disc=disc, optimizer=opt, scheduler=sch,
    )

    phi2, g2, disc2, _, _, opt2, sch2 = _bundle()
    start_epoch, step, best_loss, best_state = load_checkpoint(
        path, phi2, g2, disc2, opt2, sch2, None, None
    )
    assert start_epoch == 6
    assert best_state is None


def test_teacher_batchnorm_buffers_survive(tmp_path):
    """Teacher buffers are not EMA-averaged, so losing them resets the teacher
    to ImageNet statistics on resume."""
    phi, g, disc, _, _, opt, sch = _bundle()
    phi_t, g_t = nn.BatchNorm1d(4), nn.Linear(4, 3)
    with torch.no_grad():
        phi_t.running_mean.fill_(3.0)
    path = tmp_path / "ck.pth"
    save_checkpoint(
        path, epoch=1, step=1, best_loss=1.0, best_state=None,
        phi=phi, g=g, disc=disc, optimizer=opt, scheduler=sch,
        phi_teacher=phi_t, g_teacher=g_t,
    )

    phi2, g2, disc2, _, _, opt2, sch2 = _bundle()
    phi_t2, g_t2 = nn.BatchNorm1d(4), nn.Linear(4, 3)
    load_checkpoint(path, phi2, g2, disc2, opt2, sch2, phi_t2, g_t2)
    assert torch.allclose(phi_t2.running_mean, torch.full((4,), 3.0))
