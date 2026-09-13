"""The three-view target dataset, which is what makes the comparison fair.

Two properties are load-bearing and both fail silently when broken: the plain
view must stay deterministic, and generating the augmented views must not
perturb the global RNG.
"""

import torch
from PIL import Image

from plantuda.data.datasets import TargetThreeViewDataset
from plantuda.data.transforms import TRAIN_TRANSFORM, VAL_TRANSFORM


def _structured_image(tmp_path, name="leaf.jpg"):
    """A gradient image. A flat colour would survive any crop unchanged and so
    could not detect a broken augmentation."""
    arr = (torch.arange(300 * 300 * 3, dtype=torch.uint8) % 251).reshape(300, 300, 3)
    path = tmp_path / name
    Image.fromarray(arr.numpy()).save(path)
    return str(path)


def _dataset(tmp_path, n=3):
    samples = [(_structured_image(tmp_path, f"leaf{i}.jpg"), -1) for i in range(n)]
    return TargetThreeViewDataset(samples, VAL_TRANSFORM, TRAIN_TRANSFORM, base_seed=42)


def test_returns_three_views_and_a_masked_label(tmp_path):
    (plain, v1, v2), label = _dataset(tmp_path)[0]
    assert plain.shape == v1.shape == v2.shape == (3, 224, 224)
    assert label == -1, "target labels must never reach the training loop"


def test_views_differ_from_each_other(tmp_path):
    """Identical views would make L_cons zero for 300 epochs, with no error."""
    (_, v1, v2), _ = _dataset(tmp_path)[0]
    assert not torch.allclose(v1, v2)


def test_plain_view_is_not_augmented(tmp_path):
    """The plain view is the adversarial branch's input in both arms."""
    (plain, v1, _), _ = _dataset(tmp_path)[0]
    assert not torch.allclose(plain, v1)


def test_plain_view_is_deterministic(tmp_path):
    ds = _dataset(tmp_path)
    ds.set_epoch(1)
    first = ds[0][0][0]
    ds.set_epoch(9)
    assert torch.allclose(first, ds[0][0][0])


def test_does_not_disturb_the_global_rng(tmp_path):
    """The invariant the whole paired comparison rests on: if augmentation
    leaked into the global stream, the source images of the Mean Teacher arm
    would diverge from the baseline's after the first batch."""
    ds = _dataset(tmp_path)

    torch.manual_seed(123)
    before = torch.rand(3)

    torch.manual_seed(123)
    ds.set_epoch(7)
    _ = ds[0]
    after = torch.rand(3)

    assert torch.equal(before, after)


def test_views_change_per_epoch_but_repeat_within_one(tmp_path):
    ds = _dataset(tmp_path)
    ds.set_epoch(1)
    epoch1 = ds[0][0][1]
    ds.set_epoch(2)
    epoch2 = ds[0][0][1]
    ds.set_epoch(1)
    epoch1_again = ds[0][0][1]

    assert not torch.allclose(epoch1, epoch2), "views must vary across epochs"
    assert torch.allclose(epoch1, epoch1_again), "an epoch must be reproducible"
