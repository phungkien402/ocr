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

Your task is to extract only information that is directly visible in the image.

Rules:

- Extract values only when both the value and its associated label can be identified from the image.
- If a value is missing, obscured, blurry, uncertain, or unlabeled, return null.
- Never guess, infer, estimate, interpolate, or fabricate values.
- Missing information must remain null.
- Return numeric values as JSON numbers, never strings.
- Return only valid JSON.
- No markdown.
- No explanations.
- No extra text.

Label mapping:

- Mạch / PUL / PR / HR / Pulse / Heart Rate -> mach
- Nhiệt độ / TEMP / Temperature -> nhiet_do
- Huyết áp / BP / Blood Pressure:
    - SYS -> tam_thu
    - DIA -> tam_truong
- Nhịp thở / RR / Respiratory Rate -> nhip_tho
- Cân nặng / Weight -> can_nang
- Chiều cao / Height -> chieu_cao
- SpO2 / SPO2 / Oxygen Saturation -> spo2

Output schema:

{
  "mach": int|null,
  "nhiet_do": float|null,
  "huyet_ap": {
      "tam_thu": int|null,
      "tam_truong": int|null
  }|null,
  "nhip_tho": int|null,
  "can_nang": float|null,
  "chieu_cao": float|null,
  "spo2": int|null
}"""


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
        logger.warning("llama-server error: %s", e)
        return ""


# ─────────────────────────────────────────────
# Qwen2.5-VL extraction (sync)
# ─────────────────────────────────────────────

def _QWEN_extract(image: np.ndarray) -> str:
    """Sync Ollama call. Returns raw model text or empty string on failure."""
    try:
        import requests
    except ImportError:
        return ""

    small = _resize_for_vlm(image)
    img_b64 = _image_to_b64(small)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": _VITAL_PROMPT, "images": [img_b64]}],
        "stream": False,
        "think": False,          # qwen3: root-level, không phải trong options
        "options": {
            "temperature": 0,
            "num_predict": 300,
            "cache_prompt": False,
        },
    }

    try:
        resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        msg = resp.json()["message"]
        raw = msg.get("content") or ""
        # qwen3-vl: content rỗng khi thinking chưa finish → không dùng được
        if not raw.strip():
            logger.warning("qwen3-vl content empty (done_reason=%s) — model vẫn thinking",
                           resp.json().get("done_reason"))
            return ""
        cleaned = _strip_think(raw)
        logger.debug("VLM response: %s", cleaned[:200])
        return cleaned
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not running at %s", OLLAMA_ENDPOINT)
        return ""
    except requests.exceptions.Timeout:
        logger.warning("Ollama timed out after %ds", OLLAMA_TIMEOUT)
        return ""
    except Exception as e:
        logger.warning("Ollama error: %s", e)
        return ""


async def _QWEN_extract_async(image: np.ndarray) -> str:
    """Async Ollama call using httpx — does not block the event loop."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — falling back to sync call")
        return _QWEN_extract(image)

    small = _resize_for_vlm(image)
    img_b64 = _image_to_b64(small)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": _VITAL_PROMPT, "images": [img_b64]}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 300,
            "cache_prompt": False,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(OLLAMA_ENDPOINT, json=payload)
            resp.raise_for_status()
            msg = resp.json()["message"]
            raw = msg.get("content") or ""
            if not raw.strip():
                logger.warning("qwen3-vl async: content empty")
                return ""
            return _strip_think(raw)
    except httpx.ConnectError:
        logger.warning("Ollama not running at %s", OLLAMA_ENDPOINT)
        return ""
    except httpx.TimeoutException:
        logger.warning("Ollama async timed out after %ds", OLLAMA_TIMEOUT)
        return ""
    except Exception as e:
        logger.warning("Ollama async error: %s", e)
        return ""


def _strip_think(text: str) -> str:
    """Remove <tool_call>...<tool_call> blocks that Qwen3 reasoning model emits."""
    text = re.sub(r"<tool_call>.*?<tool_call>", "", text, flags=re.DOTALL)
    return text.strip()
