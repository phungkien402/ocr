"""Main entry point for OCR vital signs extraction pipeline."""

import argparse, json, logging, os, sys
from datetime import datetime
from pathlib import Path

from .config import DEVICE
from .ocr_engine import extract_vitals, OLLAMA_MODEL
from .parser import parse_vitals
from .preprocessor import preprocess_for_vlm
from .validator import validate_vitals

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}
_BP_DETECTOR_DIR = Path(__file__).parent.parent.parent / "bp_detector"
_BP_MODEL_PATH   = _BP_DETECTOR_DIR / "best.pt"


def _try_yolo(image_path):
    """Run YOLO detector. Returns dict with vitals + confident=True if all 3 BP fields found."""
    if not _BP_MODEL_PATH.exists():
        return None
    try:
        if str(_BP_DETECTOR_DIR) not in sys.path:
            sys.path.insert(0, str(_BP_DETECTOR_DIR))
        from predict import run_prediction
        result = run_prediction(image_path, model_path=str(_BP_MODEL_PATH))
        if result.get("status") != "success":
            return None
        v = result.get("vitals", {})
        def si(x):
            try: return int(str(x).strip())
            except: return None
        sys_val, dia_val, pul_val = si(v.get("systolic")), si(v.get("diastolic")), si(v.get("pulse"))
        if sys_val and not (60 <= sys_val <= 250): sys_val = None
        if dia_val and not (30 <= dia_val <= 150): dia_val = None
        if pul_val and not (30 <= pul_val <= 220): pul_val = None
        if not any([sys_val, dia_val, pul_val]): return None
        # All 3 present = high confidence, skip VLM entirely
        confident = all([sys_val, dia_val, pul_val])
        logger.info("YOLO: SYS=%s DIA=%s PUL=%s (confident=%s)", sys_val, dia_val, pul_val, confident)
        return {
            "huyet_ap": {"tam_thu": sys_val, "tam_truong": dia_val} if (sys_val or dia_val) else None,
            "mach": pul_val,
            "confident": confident,
            "debug_steps": result.get("debug_steps", {}),
        }
    except Exception as e:
        logger.warning("YOLO error: %s", e)
        return None


def process_image(image_path, device="cuda:0", mode="auto"):
    filename = os.path.basename(image_path)
    try:
        yolo_vitals = _try_yolo(image_path)

        # Fast path: YOLO confident → skip VLM
        if yolo_vitals and yolo_vitals.get("confident"):
            logger.info("YOLO confident — skipping VLM")
            return _build_result(filename, "", yolo_vitals)

        # Slow path: single VLM call
        raw_img = preprocess_for_vlm(image_path)
        raw_text = extract_vitals(raw_img)
        return _build_result(filename, raw_text, yolo_vitals)
    except Exception as e:
        logger.error("Error processing %s: %s", filename, e)
        return _error_result(filename, str(e))


async def process_image_async(image_path, device="cuda:0"):
    filename = os.path.basename(image_path)
    try:
        yolo_vitals = _try_yolo(image_path)

        if yolo_vitals and yolo_vitals.get("confident"):
            logger.info("YOLO confident — skipping VLM")
            return _build_result(filename, "", yolo_vitals)

        raw_img = preprocess_for_vlm(image_path)
        raw_text = extract_vitals(raw_img)
        return _build_result(filename, raw_text, yolo_vitals)
    except Exception as e:
        logger.error("Error processing %s: %s", filename, e)
        return _error_result(filename, str(e))


def _build_result(filename, raw_text, yolo_vitals=None):
    from .config import VITALS_INFO
    vitals = parse_vitals(raw_text)
    units = vitals.pop("_units", None)
    detected_device = vitals.pop("_device", "unknown")

    # Merge YOLO results (override VLM for BP fields)
    if yolo_vitals:
        if yolo_vitals.get("huyet_ap") is not None:
            vitals["huyet_ap"] = yolo_vitals["huyet_ap"]
        if yolo_vitals.get("mach") is not None:
            vitals["mach"] = yolo_vitals["mach"]

    validation, missing_fields = validate_vitals(vitals)
    fields_meta = {
        f: {"label_vn": i["label_vn"], "label_en": i["label_en"],
            "unit": i["unit"], "normal_range": i["normal_range"], "value": vitals.get(f)}
        for f, i in VITALS_INFO.items()
    }
    if yolo_vitals and yolo_vitals.get("confident"):
        engine = "yolo"
    elif yolo_vitals:
        engine = "yolo+vlm"
    else:
        engine = "vlm"

    result = {
        "source_image": filename, "ocr_raw_text": raw_text,
        "vitals": vitals, "fields_meta": fields_meta,
        "validation": validation, "missing_fields": missing_fields,
        "ocr_engine": engine, "device": detected_device,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "debug_steps": yolo_vitals.get("debug_steps", {}) if yolo_vitals else {},
    }
    if units:
        result["units_detected"] = units
    return result


def _error_result(filename, error):
    return {"source_image": filename, "error": error, "ocr_engine": "unknown",
            "processed_at": datetime.now().isoformat(timespec="seconds")}


def save_result(result, output_dir):
    out_path = os.path.join(output_dir, f"{Path(result.get('source_image','unknown')).stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def get_image_files(input_path):
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path] if Path(input_path).suffix.lower() in SUPPORTED_EXTENSIONS else []
    if os.path.isdir(input_path):
        return sorted(os.path.join(input_path, f) for f in os.listdir(input_path)
                      if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS)
    return []


def main():
    parser = argparse.ArgumentParser(description="Extract vital signs from medical images")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--device", "-d", default=DEVICE)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    os.makedirs(args.output, exist_ok=True)
    files = get_image_files(args.input)
    if not files:
        logger.warning("No images found"); sys.exit(0)
    ok = err = 0
    for path in files:
        result = process_image(path, device=args.device)
        save_result(result, args.output)
        if "error" in result: err += 1
        else: ok += 1
    logger.info("Done. OK=%d  Error=%d", ok, err)


if __name__ == "__main__":
    main()
