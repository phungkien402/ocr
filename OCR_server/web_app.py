"""FastAPI server for OCR Vital Signs."""
import logging, os, tempfile, time, pathlib
import cv2
from fastapi import FastAPI, File, UploadFile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="OCR Vital Signs", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

_HERE = pathlib.Path(__file__).parent
_HTML_PATH = _HERE / "static" / "index.html"
_VERSION = "1.0.0"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_UNIT_MAP = {
    "mach":     "bpm",
    "nhiet_do": "°C",
    "nhip_tho": "lần/phút",
    "can_nang": "kg",
    "chieu_cao":"cm",
    "spo2":     "%",
}

_FIELD_TO_KEY = {
    "mach":     "pulse",
    "nhiet_do": "temperature",
    "nhip_tho": "respiratory_rate",
    "can_nang": "weight",
    "chieu_cao":"height",
    "spo2":     "spo2",
}


def _to_v1(result: dict, elapsed: float) -> dict:
    """Convert internal pipeline result → stable v1 API format."""
    vitals = result.get("vitals", {})
    validation = result.get("validation", {})
    missing = result.get("missing_fields", [])
    warnings = []

    readings = {}

    # Blood pressure
    bp = vitals.get("huyet_ap")
    if isinstance(bp, dict) and (bp.get("tam_thu") or bp.get("tam_truong")):
        readings["systolic"]  = {"value": bp.get("tam_thu"),  "unit": "mmHg"}
        readings["diastolic"] = {"value": bp.get("tam_truong"), "unit": "mmHg"}
        v = validation.get("huyet_ap", {})
        if v.get("status") == "abnormal":
            warnings.append("blood_pressure out of normal range")
    else:
        warnings.append("blood_pressure not detected")

    # Scalar fields
    for field, key in _FIELD_TO_KEY.items():
        val = vitals.get(field)
        if val is not None:
            readings[key] = {"value": val, "unit": _UNIT_MAP[field]}
            v = validation.get(field, {})
            if v.get("status") == "abnormal":
                warnings.append(f"{key} out of normal range")
        elif field in missing:
            warnings.append(f"{key} not detected")

    # Confidence: high if YOLO ran, medium if VLM only, low if all null
    engine = result.get("ocr_engine", "")
    found = len(readings)
    if "yolo" in engine and found >= 2:
        confidence = "high"
    elif found >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "success": True,
        "data": {
            "device":     result.get("device", "unknown"),
            "readings":   readings,
            "confidence": confidence,
        },
        "warnings": warnings,
        "meta": {
            "engine":           engine,
            "processing_time_s": round(elapsed, 2),
            "model_version":    _VERSION,
            "processed_at":     result.get("processed_at", ""),
        },
    }


async def _run_pipeline(file: UploadFile):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        return None, None, "Only JPG and PNG images are supported."
    suffix = ".png" if "png" in (file.content_type or "") else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    t0 = time.perf_counter()
    try:
        from ocr_vitals.main import process_image_async
        result = await process_image_async(tmp_path, device="cuda:0")
        elapsed = time.perf_counter() - t0
        return result, elapsed, None
    except Exception as e:
        return None, None, str(e)
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": _VERSION}


@app.post("/v1/extract")
async def extract_v1(file: UploadFile = File(...)):
    """Stable API endpoint for PHR integration.

    Returns standardised {success, data, warnings, meta} format.
    Response contract is fixed regardless of model changes.
    """
    result, elapsed, err = await _run_pipeline(file)
    if err:
        return JSONResponse(status_code=400 if result is None else 500, content={
            "success": False,
            "error": err,
            "meta": {"model_version": _VERSION},
        })
    return JSONResponse(content=_to_v1(result, elapsed))


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """Legacy endpoint — full internal result (debug/dev use)."""
    result, elapsed, err = await _run_pipeline(file)
    if err:
        return JSONResponse(status_code=500, content={"detail": f"Processing error: {err}"})
    result["processing_time_s"] = round(elapsed, 2)
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
