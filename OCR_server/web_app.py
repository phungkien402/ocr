"""FastAPI server for OCR Vital Signs."""
import logging, os, tempfile, time, pathlib
from datetime import datetime
import cv2
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from ocr_vitals import storage

app = FastAPI(title="OCR Vital Signs", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

_HERE = pathlib.Path(__file__).parent
_HTML_PATH = _HERE / "static" / "index.html"
_TEST_HTML_PATH = _HERE / "static" / "test.html"
_VERSION = "1.1.0"
_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
_API_KEY = os.environ.get("OCR_API_KEY", "").strip()  # empty → auth disabled

# Fixed schema: 7 vital signs. Null if not detected.
# huyet_ap is nested {tam_thu, tam_truong} since it's 2 values.
_FIELDS = ("mach", "nhiet_do", "huyet_ap", "nhip_tho",
           "can_nang", "chieu_cao", "spo2")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────

def _require_api_key(x_api_key: str | None) -> None:
    """Reject if OCR_API_KEY is set and X-API-Key header doesn't match."""
    if not _API_KEY:
        return  # auth disabled (dev mode)
    if not x_api_key or x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _to_v1(result: dict) -> dict:
    """Minimal 7-field JSON. None for missing fields."""
    vitals = result.get("vitals", {}) or {}
    out = {f: vitals.get(f) for f in _FIELDS}
    bp = out.get("huyet_ap")
    if isinstance(bp, dict):
        sys_v = bp.get("tam_thu")
        dia_v = bp.get("tam_truong")
        if sys_v is None and dia_v is None:
            out["huyet_ap"] = None
        else:
            out["huyet_ap"] = {"tam_thu": sys_v, "tam_truong": dia_v}
    else:
        out["huyet_ap"] = None
    return out


async def _read_upload(file: UploadFile) -> tuple[bytes | None, str | None]:
    """Read upload bytes with size limit. Returns (bytes, error)."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        return None, "Only JPG and PNG images are supported."
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        return None, f"Image too large ({len(data)} > {_MAX_UPLOAD_BYTES} bytes)"
    return data, None


async def _run_pipeline_on_bytes(image_bytes: bytes, suffix: str):
    """Write to tempfile, run pipeline, cleanup. Returns (result, elapsed, error)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
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
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _persist(request_id: str, image_bytes: bytes, suffix: str,
             result: dict | None, elapsed: float | None) -> None:
    """Save image + metadata to disk. Failures logged but never raised."""
    image_path = storage.save_image(request_id, image_bytes, suffix=suffix)
    if image_path is None:
        return  # storage disabled or failed
    meta = {
        "request_id":      request_id,
        "timestamp":       datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_path":      image_path,
        "image_sha256":    storage.image_sha256(image_bytes),
        "image_bytes":     len(image_bytes),
        "model_version":   _VERSION,
        "processing_time_s": round(elapsed, 3) if elapsed is not None else None,
        "extracted":       _to_v1(result) if result else None,
        "raw_vlm_output":  (result or {}).get("ocr_raw_text") or None,
        "ocr_engine":      (result or {}).get("ocr_engine") or None,
        "error":           None if result else "pipeline_error",
    }
    storage.save_metadata(request_id, meta)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": _VERSION,
        "storage_enabled": bool(os.environ.get("STORAGE_PATH")),
        "auth_enabled": bool(_API_KEY),
    }


@app.post("/v1/extract")
async def extract_v1(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    """Extract 7 vital signs from image.

    Headers:
        X-API-Key: required if OCR_API_KEY env is set.

    Response: flat JSON with 7 fields + request_id. null per missing field.
    """
    _require_api_key(x_api_key)

    image_bytes, read_err = await _read_upload(file)
    if read_err:
        empty = {"request_id": None, **{f: None for f in _FIELDS}}
        return JSONResponse(status_code=400, content=empty)

    suffix = ".png" if "png" in (file.content_type or "") else ".jpg"
    request_id = storage.new_request_id()
    result, elapsed, err = await _run_pipeline_on_bytes(image_bytes, suffix)

    # Persist whatever we got (image + extracted output or error)
    _persist(request_id, image_bytes, suffix, result, elapsed)

    if err:
        empty = {"request_id": request_id, **{f: None for f in _FIELDS}}
        return JSONResponse(status_code=500, content=empty)

    return JSONResponse(content={"request_id": request_id, **_to_v1(result)})


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """Legacy endpoint — full internal result (debug/dev use). No auth, no storage."""
    image_bytes, read_err = await _read_upload(file)
    if read_err:
        return JSONResponse(status_code=400, content={"detail": read_err})
    suffix = ".png" if "png" in (file.content_type or "") else ".jpg"
    result, elapsed, err = await _run_pipeline_on_bytes(image_bytes, suffix)
    if err:
        return JSONResponse(status_code=500, content={"detail": f"Processing error: {err}"})
    result["processing_time_s"] = round(elapsed, 2)
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML_PATH.read_text(encoding="utf-8")


@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Test UI for /v1/extract — production endpoint with API key + storage."""
    return _TEST_HTML_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
