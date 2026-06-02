"""Image preprocessing for OCR pipeline.

Current pipeline: load → resize to MAX_DIM. Nothing else.
VLM (Qwen2-VL) handles its own normalization; heavier preprocessing
(CLAHE, deskew, LCD enhancement) was removed after benchmarking showed
no accuracy gain on test set vs. plain resize.

If you need device-specific enhancement later, add it as a new path
and route via `main.process_image_async` rather than re-inflating this module.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_DIM = 1024


def preprocess_for_vlm(image_path: str, max_dim: int = MAX_DIM) -> np.ndarray:
    """Load image from disk, resize to fit max_dim on longest side."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return preprocess_for_vlm_array(img, max_dim)


def preprocess_for_vlm_array(image: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Downscale array if longest side > max_dim, pass through otherwise."""
    return _resize(image, max_dim)


def _resize(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)
