"""Leaf segmentation with SAM, plus mask clean-up."""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def load_sam(checkpoint: str | os.PathLike, model_type: str = "vit_b", device="cpu"):
    """Download (once) and load SAM, returning a predictor.

    ``vit_b`` (~375 MB) is enough here: the lab images show one centred leaf on
    a plain background, which is the easiest case a segmenter can be given.
    """
    from segment_anything import SamPredictor, sam_model_registry

    checkpoint = str(checkpoint)
    if not os.path.exists(checkpoint):
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        print(f"Downloading SAM checkpoint ({model_type})...")
        urllib.request.urlretrieve(SAM_URLS[model_type], checkpoint)

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)


def segment_leaf(predictor, image_rgb: np.ndarray) -> np.ndarray:
    """Segment the leaf from a single centre-point prompt.

    Args:
        predictor: A `SamPredictor`.
        image_rgb: ``(H, W, 3)`` uint8 RGB image.

    Returns:
        ``(H, W)`` boolean mask, the highest-confidence of SAM's three
        candidates.
    """
    h, w = image_rgb.shape[:2]
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[w // 2, h // 2]]),
        point_labels=np.array([1]),      # 1 = foreground
        multimask_output=True,
    )
    return masks[np.argmax(scores)]


def refine_mask(
    mask: np.ndarray, k_d: int = 15, k_e: int = 10, k_g: int = 5, iterations: int = 2
) -> np.ndarray:
    """Morphological clean-up of a raw SAM mask.

    Dilation fills holes and merges fragments; the smaller erosion kernel
    trims boundary noise while leaving a net expansion. The Gaussian blur turns
    the binary edge into an alpha ramp so the composite does not show a cut-out
    outline.

    Args:
        mask: ``(H, W)`` boolean mask.
        k_d: Dilation kernel size.
        k_e: Erosion kernel size, must be smaller than `k_d`.
        k_g: Gaussian kernel size, odd.
        iterations: Dilate/erode repetitions.

    Returns:
        ``(H, W)`` float32 soft mask in ``[0, 1]``, used as the alpha channel.
    """
    m = mask.astype(np.uint8) * 255
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_d, k_d))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_e, k_e))
    for _ in range(iterations):
        m = cv2.dilate(m, kd)
        m = cv2.erode(m, ke)
    m = cv2.GaussianBlur(m, (k_g, k_g), 0)
    return m.astype(np.float32) / 255.0
