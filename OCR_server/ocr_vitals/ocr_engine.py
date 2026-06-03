"""OCR engine — async VLM extraction via OpenAI-compatible endpoint.

Works with either llama-server or vLLM. Uses httpx.AsyncClient to avoid
blocking the FastAPI event loop on long VLM calls.

PHI handling: full VLM response is NOT logged. Only length + a short hash
preview is emitted, so logs are safe to ship off-server.
"""

import base64, hashlib, logging, os, re, unicodedata
import cv2
import httpx
import numpy as np
from .preprocessor import preprocess_for_vlm_array

logger = logging.getLogger(__name__)

# ── Endpoint config (env-overridable) ───────────────────────────────────────
VLM_ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")
VLM_MODEL    = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLM_TIMEOUT  = int(os.environ.get("VLM_TIMEOUT_SECONDS", "120"))
MAX_IMAGE_DIM = 1024

# Back-compat aliases (older imports may use these names)
LLAMA_ENDPOINT = VLM_ENDPOINT
OLLAMA_MODEL   = VLM_MODEL
OLLAMA_TIMEOUT = VLM_TIMEOUT
BACKEND = "openai-compat"

# Structured fill-in prompt
PROMPT = (
    "Đọc ảnh và điền giá trị cho từng chỉ số dưới đây. "
    "Nếu không có trong ảnh, ghi: null. Giữ nguyên tên chỉ số.\n"
    "mạch:\n"
    "nhiệt độ:\n"
    "huyết áp:\n"
    "nhịp thở:\n"
    "cân nặng:\n"
    "chiều cao:\n"
    "spo2:"
)


def _resize(image: np.ndarray) -> np.ndarray:
    return preprocess_for_vlm_array(image, max_dim=MAX_IMAGE_DIM)


def _to_b64(image: np.ndarray) -> str:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _normalize_for_match(text: str) -> str:
    """Lowercase, strip Vietnamese diacritics, drop whitespace & punctuation."""
    s = text.lower().replace("đ", "d").replace("Đ", "d")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn" and c.isalnum())


_ECHO_MARKERS = (
    "diengiatrichotungchiso",
    "neukhongcotronganhghi",
    "giunguyentenchiso",
)


def _log_safe_preview(text: str) -> str:
    """Don't log PHI. Return length + sha256 prefix so we can correlate without leaking values."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"len={len(text)} sha8={digest}"


def _build_payload(img_b64: str) -> dict:
    return {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False,
    }


def _process_response(raw_text: str) -> str:
    """Strip <think>, check echo. Return text or '' if rejected."""
    cleaned = _strip_think(raw_text)
    normalized = _normalize_for_match(cleaned)
    hits = sum(1 for m in _ECHO_MARKERS if m in normalized)
    if hits >= 2:
        logger.warning("[VLM] model echoed instructions (%d markers)", hits)
        return ""
    logger.info("[VLM] response %s", _log_safe_preview(cleaned))
    return cleaned


async def extract_vitals_async(image: np.ndarray) -> str:
    """Async VLM call. Use this in async contexts (FastAPI handlers)."""
    resized = _resize(image)
    img_b64 = _to_b64(resized)
    logger.info("[VLM] request %dx%d ~%d KB (model=%s)",
                resized.shape[1], resized.shape[0],
                len(img_b64) * 3 // 4 // 1024, VLM_MODEL)
    try:
        async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
            r = await client.post(VLM_ENDPOINT, json=_build_payload(img_b64))
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        return _process_response(raw)
    except Exception as e:
        logger.warning("[VLM] error: %s", type(e).__name__)
        return ""


def extract_vitals(image: np.ndarray) -> str:
    """Sync wrapper. Prefer extract_vitals_async in FastAPI handlers."""
    import requests
    resized = _resize(image)
    img_b64 = _to_b64(resized)
    logger.info("[VLM] request %dx%d ~%d KB (model=%s) [sync]",
                resized.shape[1], resized.shape[0],
                len(img_b64) * 3 // 4 // 1024, VLM_MODEL)
    try:
        r = requests.post(VLM_ENDPOINT, json=_build_payload(img_b64), timeout=VLM_TIMEOUT)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        return _process_response(raw)
    except Exception as e:
        logger.warning("[VLM] error: %s", type(e).__name__)
        return ""


# Legacy compat
def extract_for_device(image: np.ndarray, device_type: str = "handwritten") -> str:
    return extract_vitals(image)

def extract_text(image: np.ndarray, **kwargs) -> str:
    return extract_vitals(image)

async def extract_text_async(image: np.ndarray, **kwargs) -> str:
    return await extract_vitals_async(image)
