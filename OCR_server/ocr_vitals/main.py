"""Main entry point for OCR vital signs extraction pipeline.

Pipeline:
  1. YOLO digit detector (bp_detector/predict.py) — fast, accurate for LCD BP monitors
     → extracts SYS / DIA / PUL directly from 7-segment digits
  2. Qwen VLM fallback — handles remaining fields (temp, SpO2, etc.)
     or when YOLO confidence is too low
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import DEVICE
from .ocr_engine import extract_text, extract_text_async, OLLAMA_MODEL
from .parser import parse_vitals
from .preprocessor import preprocess_for_vlm
from .validator import validate_vitals

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Path to bp_detector folder (relative to this file's package root)
_BP_DETECTOR_DIR = Path(__file__).parent.parent.parent / "bp_detector"
_BP_MODEL_PATH   = _BP_DETECTOR_DIR / "best.pt"


# ─────────────────────────────────────────────
# YOLO BP detector
# ─────────────────────────────────────────────

def _try_yolo(image_path: str) -> dict | None:
    """Run YOLO digit detector. Returns partial vitals dict or None on failure.

    Returns dict with keys: huyet_ap, mach, debug_steps (others stay None).
    Returns None if best.pt not found or detection fails.
    """
    if not _BP_MODEL_PATH.exists():
        logger.debug("best.pt not found at %s — skipping YOLO", _BP_MODEL_PATH)
        return None

    try:
        import sys as _sys
        if str(_BP_DETECTOR_DIR) not in _sys.path:
            _sys.path.insert(0, str(_BP_DETECTOR_DIR))
        from predict import run_prediction

        result = run_prediction(image_path, model_path=str(_BP_MODEL_PATH))
        if result.get("status") != "success":
            logger.debug("YOLO: no digits detected")
            return None

        v = result.get("vitals", {})
        sys_val = _safe_int(v.get("systolic"))
        dia_val = _safe_int(v.get("diastolic"))
        pul_val = _safe_int(v.get("pulse"))

        if sys_val and not (60 <= sys_val <= 250):
            sys_val = None
        if dia_val and not (30 <= dia_val <= 150):
            dia_val = None
        if pul_val and not (30 <= pul_val <= 220):
            pul_val = None

        if sys_val is None and dia_val is None and pul_val is None:
            logger.debug("YOLO: all values out of range — falling back")
            return None

        logger.info("YOLO: SYS=%s DIA=%s PUL=%s", sys_val, dia_val, pul_val)
        return {
            "huyet_ap": {"tam_thu": sys_val, "tam_truong": dia_val}
                        if (sys_val or dia_val) else None,
            "mach": pul_val,
            "debug_steps": result.get("debug_steps", {}),
        }
    except Exception as e:
        logger.warning("YOLO error: %s", e)
        return None


def _safe_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────
# Core pipeline (sync — for CLI)
# ─────────────────────────────────────────────

def process_image(image_path: str, device: str = "cuda:0", mode: str = "auto") -> dict:
    """Sync pipeline — for CLI usage."""
    filename = os.path.basename(image_path)
    try:
        yolo_vitals = _try_yolo(image_path)
        raw_img = preprocess_for_vlm(image_path)
        raw_text = extract_text(raw_img, device=device)
        return _build_result(filename, raw_text, yolo_vitals)
    except Exception as e:
        logger.error("Error processing %s: %s", filename, e)
        return _error_result(filename, str(e))


# ─────────────────────────────────────────────
# Async pipeline (for FastAPI)
# ─────────────────────────────────────────────

async def process_image_async(image_path: str, device: str = "cuda:0") -> dict:
    """Async pipeline — use in FastAPI endpoints to avoid blocking."""
    filename = os.path.basename(image_path)
    try:
        yolo_vitals = _try_yolo(image_path)
        raw_img = preprocess_for_vlm(image_path)
        raw_text = await extract_text_async(raw_img, device=device)
        return _build_result(filename, raw_text, yolo_vitals)
    except Exception as e:
        logger.error("Error processing %s: %s", filename, e)
        return _error_result(filename, str(e))


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _build_result(filename: str, raw_text: str, yolo_vitals: dict | None = None) -> dict:
    """Parse raw OCR text into a full result dict.
    YOLO results override Qwen for huyet_ap and mach when available.
    """
    from .config import VITALS_INFO

    vitals = parse_vitals(raw_text)
    units = vitals.pop("_units", None)

    # YOLO overrides Qwen for BP + pulse (more accurate for LCD digits)
    if yolo_vitals:
        if yolo_vitals.get("huyet_ap") is not None:
            vitals["huyet_ap"] = yolo_vitals["huyet_ap"]
        if yolo_vitals.get("mach") is not None:
            vitals["mach"] = yolo_vitals["mach"]

    validation, missing_fields = validate_vitals(vitals)

    fields_meta = {
        field: {
            "label_vn": info["label_vn"],
            "label_en": info["label_en"],
            "unit": info["unit"],
            "normal_range": info["normal_range"],
            "value": vitals.get(field),
        }
        for field, info in VITALS_INFO.items()
    }

    engine = _detect_engine()
    if yolo_vitals:
        engine = "yolo+vlm"

    result = {
        "source_image": filename,
        "ocr_raw_text": raw_text,
        "vitals": vitals,
        "fields_meta": fields_meta,
        "validation": validation,
        "missing_fields": missing_fields,
        "ocr_engine": engine,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "debug_steps": yolo_vitals.get("debug_steps", {}) if yolo_vitals else {},
    }
    if units:
        result["units_detected"] = units
    return result


def _detect_engine() -> str:
    return f"ollama/{OLLAMA_MODEL}"


def _error_result(filename: str, error: str) -> dict:
    return {
        "source_image": filename,
        "error": error,
        "ocr_engine": "unknown",
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def save_result(result: dict, output_dir: str):
    source = result.get("source_image", "unknown")
    out_path = os.path.join(output_dir, f"{Path(source).stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", out_path)


def get_image_files(input_path: str) -> list:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path] if Path(input_path).suffix.lower() in SUPPORTED_EXTENSIONS else []
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return []


def main():
    parser = argparse.ArgumentParser(description="Extract vital signs from medical images")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--device", "-d", default=DEVICE)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.makedirs(args.output, exist_ok=True)
    files = get_image_files(args.input)
    if not files:
        logger.warning("No images found")
        sys.exit(0)

    ok = err = 0
    for path in files:
        result = process_image(path, device=args.device)
        save_result(result, args.output)
        if "error" in result:
            err += 1
        else:
            ok += 1
    logger.info("Done. OK=%d  Error=%d", ok, err)


if __name__ == "__main__":
    main()
