"""Compositing a segmented leaf onto a field background."""

from __future__ import annotations

import random
from typing import Sequence

import cv2
import numpy as np

from plantuda.fbr.segment import refine_mask, segment_leaf


def composite(lab_img: np.ndarray, mask: np.ndarray, bg_img: np.ndarray) -> np.ndarray:
    """Alpha-composite: ``mask * leaf + (1 - mask) * background``."""
    h, w = lab_img.shape[:2]
    bg = cv2.resize(bg_img, (w, h))
    alpha = mask[:, :, np.newaxis]
    out = alpha * lab_img.astype(np.float32) + (1 - alpha) * bg.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def random_field_background(bg_path: str, out_w: int, out_h: int) -> np.ndarray:
    """Build one background patch: random crop, random rotation, centre crop.

    Taking a *patch* rather than the whole scene matters. A downscaled full
    field photo drags a second leaf into the background, and the model can then
    learn "two leaves means target domain" instead of learning the disease.

    The centre crop by ``1/sqrt(2)`` is the largest square guaranteed to lie
    inside the rotated area, so no black corners survive.
    """
    img = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    side = int(random.uniform(0.30, 0.70) * min(height, width))
    x = random.randint(0, width - side)
    y = random.randint(0, height - side)
    patch = img[y : y + side, x : x + side]

    rot = cv2.getRotationMatrix2D((side / 2, side / 2), random.uniform(0, 360), 1.0)
    patch = cv2.warpAffine(patch, rot, (side, side))

    crop = int(side / np.sqrt(2))
    off = (side - crop) // 2
    patch = patch[off : off + crop, off : off + crop]

    return cv2.resize(patch, (out_w, out_h))


def fbr_single(lab_path: str, bg_paths: Sequence[str], predictor) -> np.ndarray:
    """Run the full FBR pipeline on one laboratory image.

    Returns:
        ``(H, W, 3)`` uint8 image: the segmented leaf on a field background.
    """
    lab_rgb = cv2.cvtColor(cv2.imread(lab_path), cv2.COLOR_BGR2RGB)
    h, w = lab_rgb.shape[:2]

    mask = refine_mask(segment_leaf(predictor, lab_rgb))
    bg_rgb = random_field_background(random.choice(list(bg_paths)), w, h)

    return composite(lab_rgb, mask, bg_rgb)
