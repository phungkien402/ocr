"""Image preprocessing for OCR pipeline.

preprocess_for_vlm() is the only path used now (VietOCR removed).
It auto-detects the image type and applies targeted preprocessing
that keeps latency < 30ms while improving model accuracy.

Image type detection + preprocessing logic:
  - LCD / digital monitor  → CLAHE + unsharp mask (improve digit contrast)
  - Handwritten form       → deskew + adaptive threshold overlay on color
  - Printed / typed form   → minimal processing, resize only
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_DIM = 1024


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

_SKIP_RESIZE_BYTES = 100 * 1024  # 100 KB


def preprocess_for_vlm(image_path: str, max_dim: int = MAX_DIM) -> np.ndarray:
    """Load image. Skip resize if file is already under 100 KB."""
    import os
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return preprocess_for_vlm_array(img, max_dim)


def preprocess_for_vlm_array(image: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Downscale if needed, pass through otherwise."""
    return _resize(image, max_dim)


# ─────────────────────────────────────────────
# Image type detection  (~2 ms)
# ─────────────────────────────────────────────

def _detect_type(img: np.ndarray) -> str:
    """Classify image as 'lcd', 'handwritten', or 'printed'.

    Heuristics (fast, no ML):
    - LCD:        dark background OR high local contrast with bimodal histogram
    - Handwritten: large bright background, low edge density, irregular strokes
    - Printed:    bright background, high regular edge density
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]

    # 1. Dark-background check → LCD monitor
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 80:
        return "lcd"

    # 2. Bimodal histogram → LCD with light digits on dark or dark digits on light
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    # Check if significant mass in both dark (<80) and light (>200) regions
    dark_mass = float(hist_norm[:80].sum())
    light_mass = float(hist_norm[200:].sum())
    if dark_mass > 0.25 and light_mass > 0.25:
        return "lcd"

    # 3. Edge density → printed vs handwritten
    # Canny on small thumbnail for speed
    small = cv2.resize(gray, (min(w, 256), min(h, 256)))
    edges = cv2.Canny(small, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    if edge_density > 0.08:
        return "printed"

    return "handwritten"


# ─────────────────────────────────────────────
# LCD enhancement  (~5 ms)
# ─────────────────────────────────────────────

def _enhance_lcd(img: np.ndarray) -> np.ndarray:
    """Improve digit legibility on LCD/LED vital signs monitors.

    Pipeline (all steps < 10 ms):
    1. Invert dark-background images → digits become dark on white
       (VLM trained on light-bg text — this closes the distribution gap)
    2. Grayscale + Otsu threshold → clean binary digit mask
    3. Dilate 1px to close inter-segment gaps in 7-segment digits
       (prevents "0"→"C", "8"→"3", split segments being misread)
    4. Blend binary mask back onto color image → VLM keeps color context
       while seeing clean digit shapes
    5. CLAHE + unsharp mask for final sharpness
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    img_out = img.copy()

    # Step 1: invert if dark background
    if mean_brightness < 100:
        img_out = cv2.bitwise_not(img_out)
        gray = cv2.bitwise_not(gray)

    # Step 2: Otsu threshold → binary digit mask
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: dilate to close 7-segment gaps (1px kernel, 1 iteration)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Step 4: blend — use binary as alpha to clean digit edges on color image
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    img_out = cv2.bitwise_and(img_out, binary_bgr)
    # Set cleaned-out background to white so VLM sees dark text on white
    bg_mask = cv2.bitwise_not(binary)
    bg_bgr = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)
    img_out = cv2.add(img_out, bg_bgr)

    # Step 5: CLAHE + unsharp mask
    lab = cv2.cvtColor(img_out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img_out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(img_out, (0, 0), 1.5)
    img_out = cv2.addWeighted(img_out, 1.8, blur, -0.8, 0)

    return img_out


# ─────────────────────────────────────────────
# Handwritten enhancement  (~10 ms)
# ─────────────────────────────────────────────

def _enhance_handwritten(img: np.ndarray) -> np.ndarray:
    """Improve handwritten vital signs form legibility.

    Steps:
    1. Deskew (correct page tilt up to ±15°)
    2. Mild CLAHE to normalize lighting (pen ink on yellowish paper)
    3. Keep color — VLM reads color images better than grayscale
    """
    img_out = _deskew(img)

    lab = cv2.cvtColor(img_out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return img_out


def _deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate image to correct skew angle detected from text lines.

    Uses Hough line transform on a small thumbnail — ~5 ms.
    Skips rotation if detected angle is outside ±max_angle (likely wrong).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]

    # Work on thumbnail for speed
    scale = min(1.0, 512 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # Edge detect + Hough
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=small.shape[1] // 6, maxLineGap=10)

    if lines is None or len(lines) < 3:
        return img  # not enough lines to estimate angle

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < max_angle:
                angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img  # negligible skew

    logger.debug("Deskew: rotating %.1f°", -median_angle)
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _resize(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────
# Legacy — kept for import compatibility
# ─────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """Deprecated: was used for VietOCR. Now just calls preprocess_for_vlm."""
    return preprocess_for_vlm(image_path)


def preprocess_for_handwritten(image: np.ndarray) -> np.ndarray:
    """Legacy VietOCR helper — adaptive threshold."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

