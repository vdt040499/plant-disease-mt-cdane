"""Configuration objects.

The notebook carried these values as module-level globals, which made a run
impossible to describe without replaying import order. Here a run is fully
described by three frozen dataclasses plus a `Paths` object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ── Label space ───────────────────────────────────────────────────────────────
# Order matters: it defines the integer label of every sample in both domains.
CLASS_NAMES = ("healthy", "rust", "scab")
NUM_CLASSES = len(CLASS_NAMES)

# PlantVillage folder name -> label index. Folder names differ slightly between
# releases of the dataset, so the loader also falls back to fuzzy matching.
PVD_CLASS_MAP = {
    "Apple___healthy": 0,
    "Apple___Cedar_apple_rust": 1,
    "Apple___Apple_scab": 2,
}

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# ImageNet statistics — the ResNet-18 backbone was pretrained under them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Paths:
    """Filesystem layout for datasets, caches and checkpoints.

    A single root keeps a Colab run (root on Google Drive) and a local run
    (root anywhere) identical apart from that one path.
    """

    root: Path

    @classmethod
    def from_env(cls, default: str = "./runs") -> "Paths":
        """Read the root from ``UDA_SAVE_DIR``, falling back to ``default``."""
        return cls(Path(os.environ.get("UDA_SAVE_DIR", default)).expanduser())

    @property
    def plant_village(self) -> Path:
        return self.root / "PlantVillage"

    @property
    def plant_village_images(self) -> Path:
        """PlantVillage stores the colour images under raw/color/<class>/."""
        return self.plant_village / "raw" / "color"

    @property
    def plant_pathology(self) -> Path:
        """Contains train.csv and images/ directly at its root."""
        return self.root / "PlantPathology"

    @property
    def fbr_cache(self) -> Path:
        return self.root / "fbr_cache"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def ensure(self) -> "Paths":
        """Create every directory this layout owns. Idempotent."""
        for p in (self.root, self.fbr_cache, self.checkpoints, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class DataConfig:
    """Dataset sampling, splitting and loading.

    `seed` fixes both splits. The test split in particular must stay identical
    across arms, otherwise the accuracies are not comparable.
    """

    seed: int = 42
    max_per_class: int = 300      # source images sampled per class (paper: ~275-300)
    pvd_val_ratio: float = 0.25   # source train/val split
    ppd_target_ratio: float = 0.6  # target/test split of the field domain
    batch_size: int = 64
    num_workers: int = 0          # 0 is the stable choice on Colab


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and model selection."""

    seed: int = 42
    max_epochs: int = 300
    min_select_epoch: int = 250   # selection window opens here (paper: 250)
    lr: float = 1e-3
    weight_decay: float = 0.01
    grl_gamma: float = 10.0       # lambda schedule growth rate
    log_every: int = 10           # also the checkpoint interval


@dataclass(frozen=True)
class MeanTeacherConfig:
    """Mean Teacher hyperparameters.

    The ramp starts late on purpose: lambda_GRL is still rising before epoch
    100, and forcing consistency while the feature space is being reshaped
    invites confirmation bias.
    """

    ema_decay: float = 0.99
    cons_max: float = 9.0      # = C^2 with C = 3 classes
    cons_start: int = 100      # epoch at which w(t) leaves zero
    cons_ramp_len: int = 50    # epochs to reach cons_max
