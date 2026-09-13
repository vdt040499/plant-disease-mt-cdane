"""Dataset classes for both domains."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

Sample = Tuple[str, int]


class PlantDiseaseDataset(Dataset):
    """One dataset class for both the source and the target domain.

    Args:
        data_dir: Root whose subfolders are class names (auto-discovery mode).
            Mutually exclusive with `samples`.
        samples: Pre-built ``(path, label)`` list, for when the caller has
            already filtered or subsampled.
        classes: Class names, required together with `samples`.
        transform: Applied to every image.
        labeled: When False, ``__getitem__`` returns ``-1`` instead of the true
            label. The target domain uses this so an accidental supervised loss
            on target data fails loudly instead of silently leaking labels.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        samples: Sequence[Sample] | None = None,
        classes: Sequence[str] | None = None,
        transform=None,
        labeled: bool = True,
    ):
        if (data_dir is None) == (samples is None):
            raise ValueError("pass exactly one of data_dir or samples")

        self.transform = transform
        self.labeled = labeled

        if data_dir is not None:
            self.classes = sorted(
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            )
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            discovered: List[Sample] = []
            for cls in self.classes:
                cls_dir = os.path.join(data_dir, cls)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        discovered.append(
                            (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                        )
            self.samples = discovered
        else:
            if classes is None:
                raise ValueError("classes is required when passing samples")
            self.samples = list(samples)
            self.classes = list(classes)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")  # handles grayscale / RGBA
        if self.transform:
            image = self.transform(image)
        return (image, label) if self.labeled else (image, -1)

    def get_labels(self) -> List[int]:
        """All labels, for stratified splitting."""
        return [s[1] for s in self.samples]


class TargetThreeViewDataset(Dataset):
    """Unlabelled target images returned as ``((plain, view1, view2), -1)``.

    This class is where the fairness of the comparison is decided.

    ``plain`` uses the deterministic transform and is the exact tensor the
    CDAN+E branch consumes in *both* arms. ``view1`` and ``view2`` are two
    independent draws of the stochastic transform and are used only by the
    consistency loss.

    Why the views are generated inside ``fork_rng``
    -----------------------------------------------
    The stochastic transform draws from the global PyTorch RNG — the same
    stream that augments the source images. The training loop iterates
    ``zip(source_loader, target_loader)``, so the two sides draw interleaved.
    The baseline arm draws nothing for the target (its transform is
    deterministic). If this class drew two views from the global stream, then
    from the second batch onwards the *source* images of the Mean Teacher arm
    would differ from the baseline's, and the accuracy gap would no longer be
    attributable to the consistency loss.

    ``torch.random.fork_rng`` saves the global state, lets us seed a private
    stream for augmentation, and restores it untouched.

    Why the view seed depends on (seed, epoch, index)
    -------------------------------------------------
    Index alone would replay the same two views every epoch and destroy the
    augmentation diversity the method depends on. Adding the epoch gives fresh
    views each epoch while staying reproducible, including after a resume — an
    internal counter would reset to zero instead.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        plain_transform,
        aug_transform,
        base_seed: int,
    ):
        self.samples = list(samples)
        self.plain_transform = plain_transform
        self.aug_transform = aug_transform
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Must be called once per epoch by the training loop.

        Skipping it means 300 epochs share a single pair of views and the
        consistency loss has almost nothing to learn from.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # Outside the fork: uses the global RNG, but draws nothing from it.
        plain = self.plain_transform(img)

        # Inside the fork: global state is saved and restored verbatim.
        # devices=[] skips forking the CUDA RNG, which is slow and unused here.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                (self.base_seed * 1_000_003 + self.epoch * 10_007 + idx) % (2**31 - 1)
            )
            view1 = self.aug_transform(img)
            view2 = self.aug_transform(img)

        return (plain, view1, view2), -1
