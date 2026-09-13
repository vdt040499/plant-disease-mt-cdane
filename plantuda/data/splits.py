"""Loading and splitting both domains.

Every split is seeded. The target/test split especially: a drifting test set
would make the two arms incomparable without any visible error.
"""

from __future__ import annotations

import os
import random
from typing import List, Sequence, Tuple

from torch.utils.data import Subset

from plantuda.config import (
    CLASS_NAMES,
    PVD_CLASS_MAP,
    VALID_IMAGE_EXTS,
)
from plantuda.data.datasets import PlantDiseaseDataset, Sample
from plantuda.data.transforms import TRAIN_TRANSFORM, VAL_TRANSFORM


def load_pvd_samples(
    data_dir: str | os.PathLike,
    class_map=PVD_CLASS_MAP,
    max_per_class: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> List[Sample]:
    """Sample the labelled source domain (PlantVillage, laboratory images).

    Folder names drift between dataset releases, so a missing folder falls back
    to a case-insensitive substring match before giving up.
    """
    random.seed(seed)
    data_dir = str(data_dir)
    samples: List[Sample] = []

    for folder_name, label_idx in class_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            matches = [
                f for f in os.listdir(data_dir) if folder_name.lower() in f.lower()
            ]
            if not matches:
                if verbose:
                    print(f"  WARNING: class folder not found: {folder_name}")
                continue
            folder_path = os.path.join(data_dir, matches[0])

        imgs = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(VALID_IMAGE_EXTS)
        ]
        random.shuffle(imgs)          # shuffle before slicing, not after
        selected = imgs[:max_per_class]
        samples.extend((p, label_idx) for p in selected)
        if verbose:
            print(f"  {folder_name:<28}: {len(imgs):>5} available -> {len(selected):>4} used")

    if verbose:
        print(f"  Total PVD samples: {len(samples)}")
    return samples


def load_ppd_samples(
    data_dir: str | os.PathLike,
    class_names: Sequence[str] = CLASS_NAMES,
    verbose: bool = True,
) -> List[Sample]:
    """Load the target domain (PlantPathology, field images) from train.csv.

    Rows labelled ``multiple_diseases`` are dropped: the source domain has no
    matching class, so they cannot participate in a shared label space.
    """
    import pandas as pd  # local: training never reads a CSV

    data_dir = str(data_dir)
    train_csv = os.path.join(data_dir, "train.csv")
    img_dir = os.path.join(data_dir, "images")
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples: List[Sample] = []

    df = pd.read_csv(train_csv)
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, f"{row['image_id']}.jpg")
        if not os.path.exists(img_path):
            continue
        for cls in class_names:              # break keeps single-label rows only
            if row.get(cls, 0) == 1:
                samples.append((img_path, class_to_idx[cls]))
                break

    if verbose:
        print(f"  Total PPD samples (single-label): {len(samples)}")
    return samples


def split_pvd(
    samples: Sequence[Sample], val_ratio: float = 0.25, seed: int = 42
) -> Tuple[Subset, Subset]:
    """Stratified train/val split of the labelled source domain.

    Train uses the stochastic transform, val the deterministic one, so model
    selection is not measured through augmentation noise.
    """
    from sklearn.model_selection import train_test_split  # local: heavy import

    labels = [s[1] for s in samples]
    train_idx, val_idx = train_test_split(
        list(range(len(samples))),
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
    )
    ds_train = PlantDiseaseDataset(
        samples=samples, classes=CLASS_NAMES, transform=TRAIN_TRANSFORM, labeled=True
    )
    ds_val = PlantDiseaseDataset(
        samples=samples, classes=CLASS_NAMES, transform=VAL_TRANSFORM, labeled=True
    )
    return Subset(ds_train, train_idx), Subset(ds_val, val_idx)


def split_ppd(
    samples: Sequence[Sample], target_ratio: float = 0.6, seed: int = 42
) -> Tuple[Subset, Subset]:
    """Split the field domain into an unlabelled adaptation set and a test set.

    The test half keeps its labels but is only ever touched by the final
    evaluation. Never for training, never for model selection.
    """
    ds = PlantDiseaseDataset(
        samples=samples, classes=CLASS_NAMES, transform=VAL_TRANSFORM, labeled=True
    )
    indices = list(range(len(samples)))
    random.seed(seed)                  # fixed seed BEFORE the shuffle
    random.shuffle(indices)
    n_target = int(len(indices) * target_ratio)
    return Subset(ds, indices[:n_target]), Subset(ds, indices[n_target:])


def subset_samples(subset: Subset) -> List[Sample]:
    """Recover the ``(path, label)`` list behind a Subset."""
    return [subset.dataset.samples[i] for i in subset.indices]


def load_cached_fbr_samples(cache_dir: str | os.PathLike) -> List[Sample]:
    """Rebuild the source sample list from a pre-computed FBR cache.

    The cache filenames encode both the original order and the label
    (``00042_c1.jpg``), so training never needs SAM or OpenCV installed.

    Raises:
        FileNotFoundError: If the cache directory is missing or empty. Build it
            first with ``scripts/build_fbr_cache.py``.
    """
    cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"FBR cache not found: {cache_dir}")

    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".jpg"))
    if not files:
        raise FileNotFoundError(f"FBR cache is empty: {cache_dir}")

    samples: List[Sample] = []
    for fname in files:
        stem = os.path.splitext(fname)[0]
        label = int(stem.rsplit("_c", 1)[1])
        samples.append((os.path.join(cache_dir, fname), label))
    return samples
