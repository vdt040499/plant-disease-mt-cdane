"""Mean Teacher primitives: ramp, consistency loss, teacher construction, EMA."""

import copy
import math

import pytest
import torch
import torch.nn as nn

from plantuda.models.cdan import cdan_entropy_weight
from plantuda.models.mean_teacher import (
    consistency_weight,
    create_teacher,
    sigmoid_rampup,
    softmax_mse_loss,
    update_ema,
)
from plantuda.models.networks import Classifier, FeatureExtractor


# ── sigmoid_rampup ────────────────────────────────────────────────────────────

def test_rampup_starts_near_zero_and_saturates_at_one():
    assert sigmoid_rampup(0, 50) == pytest.approx(math.exp(-5.0))
    assert sigmoid_rampup(50, 50) == pytest.approx(1.0)


def test_rampup_clips_beyond_length():
    assert sigmoid_rampup(500, 50) == pytest.approx(1.0)


def test_rampup_is_monotonic():
    vals = [sigmoid_rampup(x, 50) for x in range(0, 51, 5)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_rampup_zero_length_returns_one():
    assert sigmoid_rampup(0, 0) == 1.0


# ── consistency_weight ────────────────────────────────────────────────────────

def test_consistency_is_exactly_zero_before_start():
    assert consistency_weight(1) == 0.0
    assert consistency_weight(99) == 0.0


def test_consistency_saturates_at_cons_max():
    assert consistency_weight(150) == pytest.approx(9.0)
    assert consistency_weight(300) == pytest.approx(9.0)


def test_consistency_midpoint_is_between():
    assert 0.0 < consistency_weight(125) < 9.0


# ── softmax_mse_loss ──────────────────────────────────────────────────────────

def test_mse_zero_for_identical_logits():
    a = torch.randn(4, 3)
    assert softmax_mse_loss(a, a).item() == pytest.approx(0.0, abs=1e-10)


def test_mse_positive_for_different_logits():
    a = torch.zeros(4, 3)
    b = torch.tensor([[10.0, 0.0, 0.0]] * 4)
    assert softmax_mse_loss(a, b).item() > 0.0


def test_mse_divides_by_num_classes():
    """Reference formula: squared differences summed, divided by class count."""
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[100.0, -100.0]])   # softmax ~ [1, 0]
    expected = ((0.5 - 1.0) ** 2 + (0.5 - 0.0) ** 2) / 2
    assert softmax_mse_loss(a, b).item() == pytest.approx(expected, abs=1e-6)


def test_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        softmax_mse_loss(torch.zeros(2, 3), torch.zeros(2, 4))


# ── update_ema ────────────────────────────────────────────────────────────────

def _two_linears():
    student = nn.Linear(4, 3)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        student.weight.fill_(1.0)
        teacher.weight.fill_(0.0)
    return student, teacher


def test_ema_alpha_is_zero_at_step_zero():
    """alpha = min(1 - 1/(0+1), 0.99) = 0, so the teacher takes the student."""
    student, teacher = _two_linears()
    alpha = update_ema([student], [teacher], global_step=0)
    assert alpha == 0.0
    assert torch.allclose(teacher.weight, torch.ones_like(teacher.weight))


def test_ema_alpha_is_half_at_step_one():
    student, teacher = _two_linears()
    alpha = update_ema([student], [teacher], global_step=1)
    assert alpha == pytest.approx(0.5)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5))


def test_ema_alpha_caps_at_decay():
    student, teacher = _two_linears()
    alpha = update_ema([student], [teacher], global_step=10_000, ema_decay=0.99)
    assert alpha == pytest.approx(0.99)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.01))


def test_ema_does_not_touch_buffers():
    """BatchNorm buffers are not averaged; that is why the teacher stays in
    train mode, where its own forward passes update them."""
    student = nn.BatchNorm1d(4)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        student.running_mean.fill_(5.0)
    update_ema([student], [teacher], global_step=1)
    assert torch.allclose(teacher.running_mean, torch.zeros(4))


# ── create_teacher ────────────────────────────────────────────────────────────

def _student():
    phi = FeatureExtractor(pretrained=False)
    g = Classifier(phi.out_dim, 3)
    return phi, g


def test_teacher_starts_identical_to_student():
    phi, g = _student()
    _, g_t = create_teacher(phi, g, 3, "cpu")
    assert torch.allclose(g_t.fc.weight, g.fc.weight)


def test_teacher_params_require_no_grad():
    phi, g = _student()
    phi_t, g_t = create_teacher(phi, g, 3, "cpu")
    assert all(not p.requires_grad for p in phi_t.parameters())
    assert all(not p.requires_grad for p in g_t.parameters())


def test_teacher_output_carries_no_grad():
    phi, g = _student()
    phi_t, g_t = create_teacher(phi, g, 3, "cpu")
    phi_t.eval(); g_t.eval()
    out = g_t(phi_t(torch.randn(2, 3, 64, 64)))
    assert out.requires_grad is False


def test_bn_buffers_move_in_train_mode():
    phi, g = _student()
    phi_t, _ = create_teacher(phi, g, 3, "cpu")
    phi_t.train()
    before = phi_t.net[1].running_mean.clone()
    with torch.no_grad():
        phi_t(torch.randn(4, 3, 64, 64))
    assert not torch.allclose(before, phi_t.net[1].running_mean)


def test_bn_buffers_frozen_in_eval_mode():
    phi, g = _student()
    phi_t, _ = create_teacher(phi, g, 3, "cpu")
    phi_t.eval()
    before = phi_t.net[1].running_mean.clone()
    with torch.no_grad():
        phi_t(torch.randn(4, 3, 64, 64))
    assert torch.allclose(before, phi_t.net[1].running_mean)


# ── cdan_entropy_weight ───────────────────────────────────────────────────────

def test_entropy_weight_confident_near_two():
    """H(p) ~ 0 gives w = 1 + exp(0) = 2."""
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    assert cdan_entropy_weight(probs).item() == pytest.approx(2.0, abs=1e-3)


def test_entropy_weight_is_detached():
    probs = torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True)
    assert cdan_entropy_weight(probs).requires_grad is False
