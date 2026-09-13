"""Field-adaptive Background Recomposition.

Importing this package pulls in OpenCV and segment-anything, which the training
code does not need. Import it only when building the source-image cache.
"""

from plantuda.fbr.segment import load_sam, refine_mask, segment_leaf
from plantuda.fbr.compose import composite, fbr_single, random_field_background
from plantuda.fbr.cache import build_fbr_cache

__all__ = [
    "load_sam",
    "segment_leaf",
    "refine_mask",
    "composite",
    "random_field_background",
    "fbr_single",
    "build_fbr_cache",
]
