"""Integration smoke test for the shared training loop, on CPU with fake data.

The loop cannot be unit-tested apart, and it is where mistakes concentrate:
unpacking the three-view batch, the multilinear map's shape, gradient flow
through the GRL, the EMA/optimiser ordering, the model-selection path. Two
epochs on 64x64 images catch those in seconds instead of after 300 epochs on a
hosted runtime.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from plantuda.config import MeanTeacherConfig, TrainConfig
from plantuda.data.loaders import Loaders
from plantuda.engine.train import train_arm
from plantuda.models.mean_teacher import create_teacher, update_ema
from plantuda.models.networks import Classifier, FeatureExtractor

N_CLASSES = 3
IMG = 64        # smaller than 224 for speed; ResNet-18 still accepts it
BATCH = 4
N_BATCH = 2


class _Labeled(Dataset):
    def __init__(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, IMG, IMG, generator=g)
        self.y = torch.randint(0, N_CLASSES, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _ThreeView(Dataset):
    """Mirrors the batch shape of the real target loader: ((plain, v1, v2), -1)."""

    def __init__(self, n, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, IMG, IMG, generator=g)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        base = self.x[i]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.epoch * 1000 + i)
            v1 = base + 0.1 * torch.randn_like(base)
            v2 = base + 0.1 * torch.randn_like(base)
        return (base, v1, v2), -1


@pytest.fixture(scope="module")
def loaders():
    three_view = _ThreeView(BATCH * N_BATCH)
    return Loaders(
        source_train=DataLoader(_Labeled(BATCH * N_BATCH), batch_size=BATCH, drop_last=True),
        source_val=DataLoader(_Labeled(BATCH), batch_size=BATCH),
        target_three_view=DataLoader(three_view, batch_size=BATCH, drop_last=True),
        target_plain=DataLoader(_Labeled(BATCH * N_BATCH), batch_size=BATCH, drop_last=True),
        test=DataLoader(_Labeled(BATCH), batch_size=BATCH),
        target_three_view_dataset=three_view,
    )


def _run(loaders, use_mt, **overrides):
    train_kw = dict(seed=1, max_epochs=2, min_select_epoch=1, log_every=99)
    train_kw.update({k: v for k, v in overrides.items() if k in TrainConfig.__dataclass_fields__})
    mt_kw = dict(cons_start=1, cons_ramp_len=1)
    mt_kw.update({k: v for k, v in overrides.items() if k in MeanTeacherConfig.__dataclass_fields__})
    return train_arm(
        loaders,
        use_mt=use_mt,
        device="cpu",
        train_cfg=TrainConfig(**train_kw),
        mt_cfg=MeanTeacherConfig(**mt_kw),
        ckpt_dir=None,
        pretrained=False,          # no ImageNet download in a unit test
        verbose=overrides.get("verbose", False),
    )


def test_baseline_arm_completes(loaders):
    result = _run(loaders, use_mt=False)
    assert result.use_mt is False
    assert result.arm == "CDAN+E"
    assert 0.0 <= result.acc_selected <= 100.0
    assert torch.isfinite(torch.tensor(result.best_val_loss))


def test_mean_teacher_arm_completes(loaders):
    result = _run(loaders, use_mt=True)
    assert result.use_mt is True
    assert 0.0 <= result.acc_selected <= 100.0


def test_arms_diverge(loaders):
    """Identical results in both arms would mean use_mt does nothing."""
    base = _run(loaders, use_mt=False)
    mt = _run(loaders, use_mt=True)
    assert (base.acc_selected != mt.acc_selected
            or base.best_val_loss != mt.best_val_loss)


def test_selection_falls_back_when_window_never_opens(loaders, capsys):
    result = _run(loaders, use_mt=True, max_epochs=1, min_select_epoch=99)
    assert 0.0 <= result.acc_selected <= 100.0
    assert "selection window never opened" in capsys.readouterr().out


def test_baseline_reports_zero_consistency(loaders, capsys):
    _run(loaders, use_mt=False, log_every=1, verbose=True)
    out = capsys.readouterr().out
    assert "cons=0.0000" in out, out


def test_ema_moves_teacher_towards_student():
    torch.manual_seed(1)
    phi = FeatureExtractor(pretrained=False)
    g = Classifier(phi.out_dim, N_CLASSES)
    phi_t, g_t = create_teacher(phi, g, N_CLASSES, "cpu")
    before = g_t.fc.weight.clone()
    with torch.no_grad():
        g.fc.weight.add_(1.0)
    update_ema([phi, g], [phi_t, g_t], global_step=100)
    assert not torch.allclose(g_t.fc.weight, before)
    assert not torch.allclose(g_t.fc.weight, g.fc.weight)


def test_baseline_never_updates_the_teacher():
    """The baseline calls no EMA update, so its teacher keeps a random head.
    That is why the baseline measures its class histogram on the student."""
    torch.manual_seed(1)
    phi = FeatureExtractor(pretrained=False)
    g = Classifier(phi.out_dim, N_CLASSES)
    _, g_t = create_teacher(phi, g, N_CLASSES, "cpu")
    before = g_t.fc.weight.clone()
    with torch.no_grad():
        g.fc.weight.add_(1.0)
    assert torch.allclose(before, g_t.fc.weight)
