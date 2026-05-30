"""OCR engine using qwen2.5vl:7b via Ollama.

Optimizations:
- Auto-detect image type (LCD / handwritten / printed) → targeted preprocessing
- Image resized to max 1024px before base64 (biggest speed win for Ollama)
- num_predict=300 + temperature=0 → deterministic + fast
- <think> tokens stripped from Qwen3 output
- JSON output format → parser is trivial, no regex fragility
- Async-compatible: _qwen3_vl_extract_async for use with httpx in web_app
"""

import base64
import logging
import re

import cv2
import numpy as np

from .preprocessor import preprocess_for_vlm_array

logger = logging.getLogger(__name__)

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
LLAMA_ENDPOINT  = "http://localhost:8080/v1/chat/completions"  # llama-server OpenAI API

# Chọn backend: "ollama" hoặc "llama"
BACKEND = "llama"
OLLAMA_MODEL = "qwen2.5vl:3b"     # dùng khi BACKEND="ollama"
MAX_IMAGE_DIM = 1024
OLLAMA_TIMEOUT = 120


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def extract_text(image: np.ndarray, device: str = "cuda:0",
                 mode: str = "auto", image_path: str = None) -> str:
    """Extract vitals text — route theo BACKEND config."""
    if BACKEND == "llama":
        return _llama_extract(image)
    return _qwen3_vl_extract(image)


async def extract_text_async(image: np.ndarray, device: str = "cuda:0") -> str:
    """Async version."""
    if BACKEND == "llama":
        return _llama_extract(image)   # llama-server sync là đủ fast
    return await _qwen3_vl_extract_async(image)


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def _resize_for_vlm(image: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    """Auto-detect image type, apply targeted preprocessing, then resize."""
    return preprocess_for_vlm_array(image, max_dim=max_dim)


def _image_to_b64(image: np.ndarray) -> str:
    """Encode BGR/gray numpy image to base64 JPEG string."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode()


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

_VITAL_PROMPT = """You are a medical OCR extraction assistant.

Step 1 — Identify the device type from the image:
- blood_pressure_monitor: LCD showing SYS/DIA/PUL (blood pressure cuff)
- spo2_monitor: finger clip device showing SpO2 % and pulse
- thermometer: digital thermometer showing temperature
- glucometer: blood glucose meter showing mg/dL or mmol/L
- vital_signs_monitor: hospital bedside monitor with multiple vitals
- lab_report: printed paper form with lab/vital results
- unknown: anything else

Step 2 — Extract ONLY values with clearly visible labels for that device:
- blood_pressure_monitor: extract mach (PUL), huyet_ap (SYS→tam_thu, DIA→tam_truong) only
- spo2_monitor: extract spo2, mach only
- thermometer: extract nhiet_do only (convert °F to °C if needed)
- glucometer: extract nhiet_do field with glucose value (mg/dL) only
- vital_signs_monitor: extract all visible fields
- lab_report: extract all visible fields
- unknown: extract whatever labeled values are visible

Rules:
- Extract values only when both label AND value are clearly visible
- Never guess, infer, or fabricate — missing = null
- Return numeric values as JSON numbers, never strings
- Return only valid JSON, no markdown, no explanation

Output schema:
{"device": "device_type", "mach": int|null, "nhiet_do": float|null, "huyet_ap": {"tam_thu": int|null, "tam_truong": int|null}|null, "nhip_tho": int|null, "can_nang": float|null, "chieu_cao": float|null, "spo2": int|null}"""


# ─────────────────────────────────────────────
# llama-server (OpenAI-compatible API)
# ─────────────────────────────────────────────

def _llama_extract(image: np.ndarray) -> str:
    """Gọi llama-server /v1/chat/completions — không có thinking mode."""
    try:
        import requests
    except ImportError:
        return ""

    small = _resize_for_vlm(image)
    img_b64 = _image_to_b64(small)

    payload = {
        "model": "local",   # llama-server không cần tên model
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",      "text": _VITAL_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]
        }],
        "temperature": 0,
        "max_tokens": 300,
        "stream": False,
    }

    try:
        resp = requests.post(LLAMA_ENDPOINT, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        cleaned = _strip_think(raw)
        logger.debug("llama-server response: %s", cleaned[:200])
        return cleaned
    except Exception as e:
        logger.warning("Ollama async error: %s", e)
        return ""
