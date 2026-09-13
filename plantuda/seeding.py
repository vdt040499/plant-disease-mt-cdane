"""Reproducibility primitives.

Deep learning draws randomness from several independent sources. Two of them
matter for the paired comparison between arms:

1. The global PyTorch RNG — weight initialisation and image augmentation.
2. `LOADER_GENERATOR` — the shuffle order of every DataLoader.

`set_seed` deliberately does **not** touch `LOADER_GENERATOR`. That generator is
shared by all loaders and its state advances continuously, so calling
`set_seed` twice in one process does not rewind data order. The training loop
therefore re-seeds it per epoch through `seed_loader_epoch`; without that call,
running the same arm twice in one session produces different results even with
an identical seed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Shared by every DataLoader built in this project. Seeded once here, then
# re-seeded per epoch by the training loop.
LOADER_GENERATOR = torch.Generator()
LOADER_GENERATOR.manual_seed(42)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA), and force determinism.

    Note the omission: `LOADER_GENERATOR` is untouched, see module docstring.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # the autotuner is load-dependent

    # warn_only: a few ops have no deterministic implementation; warn instead of
    # aborting a 300-epoch run.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Required by cuBLAS for deterministic matmuls. Read at CUDA init, so this
    # must be set before the first CUDA operation of the process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process (only relevant when num_workers > 0)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_loader_epoch(seed: int, epoch: int) -> None:
    """Pin the shuffle order of one epoch.

    Both arms call this with the same formula, so batch k of epoch e holds the
    same images in both runs. Resuming from a checkpoint also lands on the
    correct order rather than an arbitrary one.
    """
    LOADER_GENERATOR.manual_seed(seed * 1000 + epoch)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
