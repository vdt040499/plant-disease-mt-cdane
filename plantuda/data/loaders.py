"""DataLoader construction.

Every loader shares one generator (`LOADER_GENERATOR`) so that a single
per-epoch re-seed pins the batch order of all of them at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from torch.utils.data import DataLoader, Dataset, Subset

from plantuda.config import CLASS_NAMES, DataConfig
from plantuda.data.datasets import (
    PlantDiseaseDataset,
    Sample,
    TargetThreeViewDataset,
)
from plantuda.data.splits import (
    load_ppd_samples,
    load_pvd_samples,
    split_pvd,
    split_ppd,
    subset_samples,
)
from plantuda.data.transforms import TRAIN_TRANSFORM, VAL_TRANSFORM
from plantuda.seeding import LOADER_GENERATOR, seed_worker


def make_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a reproducible DataLoader.

    `drop_last=True` on training and target loaders keeps source and target
    batch sizes equal, which the domain losses assume.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=LOADER_GENERATOR,
    )


@dataclass
class Loaders:
    """Everything a training run consumes.

    Attributes:
        source_train: FBR-composited source images, labelled, augmented.
        source_val: FBR-composited source images, labelled, deterministic.
            Model selection reads this and nothing else — using target labels
            would break the unsupervised premise.
        target_three_view: Unlabelled target, yields (plain, view1, view2).
            Both arms iterate this loader so their batch order is identical;
            the baseline simply ignores the two augmented views.
        target_plain: Unlabelled target, single deterministic view. Only used
            by the RNG probe to emulate a Mean-Teacher-free data stream.
        test: Held-out labelled field images. Final evaluation only.
        target_three_view_dataset: The dataset behind `target_three_view`,
            exposed so the training loop can call `set_epoch`.
    """

    source_train: DataLoader
    source_val: DataLoader
    target_three_view: DataLoader
    target_plain: DataLoader
    test: DataLoader
    target_three_view_dataset: TargetThreeViewDataset

    @property
    def steps_per_epoch(self) -> int:
        """Batches per epoch — the shorter of source and target."""
        return min(len(self.source_train), len(self.target_three_view))


def build_loaders(
    fbr_samples: Sequence[Sample],
    ppd_samples: Sequence[Sample],
    cfg: DataConfig = DataConfig(),
) -> Loaders:
    """Assemble every loader from cached FBR source images and target samples.

    Args:
        fbr_samples: ``(path, label)`` of the FBR-composited source images.
        ppd_samples: ``(path, label)`` of all single-label field images.
        cfg: Batch size, split ratios and seed.

    The source train/val split is stratified and both halves are FBR images:
    selecting on non-FBR validation would measure a distribution the model is
    not trained on.
    """
    from sklearn.model_selection import train_test_split  # local: heavy import

    fbr_ds_train = PlantDiseaseDataset(
        samples=fbr_samples, classes=CLASS_NAMES, transform=TRAIN_TRANSFORM
    )
    fbr_ds_val = PlantDiseaseDataset(
        samples=fbr_samples, classes=CLASS_NAMES, transform=VAL_TRANSFORM
    )
    train_idx, val_idx = train_test_split(
        list(range(len(fbr_samples))),
        test_size=cfg.pvd_val_ratio,
        stratify=[s[1] for s in fbr_samples],
        random_state=cfg.seed,
    )

    target_subset, test_subset = split_ppd(
        ppd_samples, target_ratio=cfg.ppd_target_ratio, seed=cfg.seed
    )
    target_samples = subset_samples(target_subset)

    three_view = TargetThreeViewDataset(
        samples=target_samples,
        plain_transform=VAL_TRANSFORM,
        aug_transform=TRAIN_TRANSFORM,
        base_seed=cfg.seed,
    )
    plain_target = PlantDiseaseDataset(
        samples=target_samples,
        classes=CLASS_NAMES,
        transform=VAL_TRANSFORM,
        labeled=False,
    )

    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    loaders = Loaders(
        source_train=make_loader(Subset(fbr_ds_train, train_idx), shuffle=True, drop_last=True, **kw),
        source_val=make_loader(Subset(fbr_ds_val, val_idx), shuffle=False, **kw),
        target_three_view=make_loader(three_view, shuffle=True, drop_last=True, **kw),
        target_plain=make_loader(plain_target, shuffle=True, drop_last=True, **kw),
        test=make_loader(test_subset, shuffle=False, **kw),
        target_three_view_dataset=three_view,
    )

    if len(loaders.target_three_view) != len(loaders.target_plain):
        raise RuntimeError(
            "three-view and plain target loaders disagree on batch count; "
            "the lambda schedules of the two arms would diverge"
        )
    return loaders
