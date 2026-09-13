"""Pre-computing the FBR source images.

Running SAM on every image every epoch would dominate training time and, worse,
would make the source distribution change between epochs. The cache fixes both:
one composite per source image, written once and reused.
"""

from __future__ import annotations

import os
import shutil
from typing import List, Sequence, Tuple

from PIL import Image

from plantuda.data.datasets import Sample
from plantuda.fbr.compose import fbr_single
from plantuda.seeding import set_seed


def build_fbr_cache(
    pvd_samples: Sequence[Sample],
    bg_paths: Sequence[str],
    predictor,
    cache_dir: str | os.PathLike,
    seed: int = 42,
    verbose: bool = True,
) -> List[Sample]:
    """Composite every source image once and return the cached sample list.

    Args:
        pvd_samples: Source ``(path, label)`` pairs.
        bg_paths: Field images used as backgrounds. Pass the *adaptation*
            split only — drawing from the test split would leak test-set
            backgrounds into training.
        predictor: A `SamPredictor`.
        cache_dir: Destination directory; existing files are reused.

    A failed segmentation falls back to copying the original image rather than
    aborting a long run; the fallback is counted in the summary.
    """
    cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    set_seed(seed)  # reproducible background choice and rotation

    cached: List[Sample] = []
    n_new = n_reused = n_fallback = 0

    for idx, (orig_path, label) in enumerate(pvd_samples):
        cache_path = os.path.join(cache_dir, f"{idx:05d}_c{label}.jpg")
        if os.path.exists(cache_path):
            n_reused += 1
        else:
            try:
                result = fbr_single(orig_path, bg_paths, predictor)
                Image.fromarray(result).save(cache_path, quality=92)
            except Exception:
                shutil.copy(orig_path, cache_path)
                n_fallback += 1
            n_new += 1
        cached.append((cache_path, label))

        if verbose and (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(pvd_samples)} processed")

    if verbose:
        print(f"FBR cache ready: computed {n_new}, reused {n_reused}, fallback {n_fallback}")
        print(f"Cache directory: {cache_dir}")
    return cached
