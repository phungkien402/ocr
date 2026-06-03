"""FastAPI server for OCR Vital Signs.

Production hardening:
- API key auth via X-API-Key (env OCR_API_KEY)
- Per-IP rate limit via slowapi (env RATE_LIMIT, default "60/minute")
- Max upload size (env MAX_UPLOAD_BYTES, default 10MB)
- Image + metadata storage for later fine-tune (env STORAGE_PATH)
- PHI-safe logs: no raw vitals values logged, only image hash for correlation
"""
import hashlib
import logging
import os
import pathlib
import tempfile
import time
from datetime import datetime

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from ocr_vitals import storage
from ocr_vitals.vlm_client import get_breaker as _get_vlm_breaker

# ─── Config from env ──────────────────────────────────────────────────────────
_VERSION           = "1.2.0"
_MAX_UPLOAD_BYTES  = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
_API_KEY           = os.environ.get("OCR_API_KEY", "").strip()
_RATE_LIMIT        = os.environ.get("RATE_LIMIT", "60/minute")

_FIELDS = ("mach", "nhiet_do", "huyet_ap", "nhip_tho",
           "can_nang", "chieu_cao", "spo2")

_HERE = pathlib.Path(__file__).parent
_HTML_PATH      = _HERE / "static" / "index.html"
_TEST_HTML_PATH = _HERE / "static" / "test.html"
_DEMO_HTML_PATH = _HERE / "static" / "demo.html"


# ─── App + middleware ─────────────────────────────────────────────────────────

def _rate_key(request: Request) -> str:
    """Rate-limit key: API key if present, else remote IP. Prevents 1 IP from
    blocking other users behind a NAT, and prevents 1 API key from abuse."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_key, default_limits=[_RATE_LIMIT])

app = FastAPI(title="OCR Vital Signs", version=_VERSION)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


# ─── Auth ─────────────────────────────────────────────────────────────────────

def _require_api_key(x_api_key: str | None) -> None:
    if not _API_KEY:
        return  # dev mode
    if not x_api_key or x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_v1(result: dict) -> dict:
    vitals = result.get("vitals", {}) or {}
    out = {f: vitals.get(f) for f in _FIELDS}
    bp = out.get("huyet_ap")
    if isinstance(bp, dict):
        sys_v, dia_v = bp.get("tam_thu"), bp.get("tam_truong")
        if sys_v is None and dia_v is None:
            out["huyet_ap"] = None
        else:
            out["huyet_ap"] = {"tam_thu": sys_v, "tam_truong": dia_v}
    else:
        out["huyet_ap"] = None
    return out


async def _read_upload(file: UploadFile) -> tuple[bytes | None, str | None]:
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        return None, "Only JPG and PNG images are supported."
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        return None, f"Image too large ({len(data)} > {_MAX_UPLOAD_BYTES} bytes)"
    return data, None


async def _run_pipeline_on_bytes(image_bytes: bytes, suffix: str):
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
        try: os.unlink(tmp_path)
        except OSError: pass


def _persist(request_id: str, image_bytes: bytes, suffix: str,
             result: dict | None, elapsed: float | None) -> None:
    image_path = storage.save_image(request_id, image_bytes, suffix=suffix)
    if image_path is None:
        return
    meta = {
        "request_id":        request_id,
        "timestamp":         datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_path":        image_path,
        "image_sha256":      storage.image_sha256(image_bytes),
        "image_bytes":       len(image_bytes),
        "model_version":     _VERSION,
        "processing_time_s": round(elapsed, 3) if elapsed is not None else None,
        "extracted":         _to_v1(result) if result else None,
        "raw_vlm_output":    (result or {}).get("ocr_raw_text") or None,
        "ocr_engine":        (result or {}).get("ocr_engine") or None,
        "error":             None if result else "pipeline_error",
    }
    storage.save_metadata(request_id, meta)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
@limiter.exempt
async def health(request: Request):
    return {
        "status":           "ok",
        "version":          _VERSION,
        "storage_enabled":  storage.is_enabled(),
        "auth_enabled":     bool(_API_KEY),
        "rate_limit":       _RATE_LIMIT,
        "vlm_circuit":      _get_vlm_breaker().snapshot(),
    }


@app.post("/v1/extract")
@limiter.limit(_RATE_LIMIT)
async def extract_v1(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    """Extract 7 vital signs from image. Returns flat JSON + request_id."""
    _require_api_key(x_api_key)

    image_bytes, read_err = await _read_upload(file)
    if read_err:
        return JSONResponse(status_code=400,
            content={"request_id": None, **{f: None for f in _FIELDS}})

    suffix = ".png" if "png" in (file.content_type or "") else ".jpg"
    request_id = storage.new_request_id()
    result, elapsed, err = await _run_pipeline_on_bytes(image_bytes, suffix)

    _persist(request_id, image_bytes, suffix, result, elapsed)

    # PHI-safe log line — only ids + status, no values
    logger.info("[extract] request_id=%s status=%s elapsed=%.2fs engine=%s",
                request_id, "err" if err else "ok",
                elapsed or 0, (result or {}).get("ocr_engine", "-"))

    if err:
        return JSONResponse(status_code=500,
            content={"request_id": request_id, **{f: None for f in _FIELDS}})
    return JSONResponse(content={"request_id": request_id, **_to_v1(result)})


@app.post("/process")
@limiter.limit(_RATE_LIMIT)
async def process(request: Request, file: UploadFile = File(...)):
    """Legacy endpoint — full internal result (debug/dev use)."""
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
    return _TEST_HTML_PATH.read_text(encoding="utf-8")


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Full PHR → OCR → HIS flow demo (with FHIR Bundle preview)."""
    return _DEMO_HTML_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
