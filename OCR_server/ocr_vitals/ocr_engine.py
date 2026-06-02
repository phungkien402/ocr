"""OCR engine — single VLM extraction call via OpenAI-compatible endpoint.

Works with either llama-server or vLLM (both expose /v1/chat/completions).
Endpoint + model name are env-configurable so you can swap backends without
touching code.
"""

import base64, logging, os, re, unicodedata
import cv2
import numpy as np
from .preprocessor import preprocess_for_vlm_array

logger = logging.getLogger(__name__)

# ── Endpoint config (env-overridable) ───────────────────────────────────────
# vLLM:        http://localhost:8080/v1/chat/completions
# llama-server: http://localhost:8080/v1/chat/completions  (same path)
VLM_ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")

# Model name sent in payload. vLLM requires the actual HF model id.
# llama-server is permissive — any string works (use "local" historically).
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT_SECONDS", "120"))
MAX_IMAGE_DIM = 1024

# Back-compat aliases (other modules may still import the old names)
LLAMA_ENDPOINT = VLM_ENDPOINT
OLLAMA_MODEL = VLM_MODEL
OLLAMA_TIMEOUT = VLM_TIMEOUT
BACKEND = "openai-compat"

# Structured fill-in prompt: provide canonical labels so VLM uses them verbatim
# instead of hallucinating (e.g. "Chỉ cầu" instead of "chiều cao"). Empty value
# → null. Avoids enumeration drift on small VLMs (Qwen2.5-VL-3B).

PROMPT = (
    "Bạn là trợ lý OCR y tế. Đọc ảnh và liệt kê các chỉ số sinh tồn thấy được.\n"
    "Chỉ đọc giá trị thực sự có trong ảnh, không suy đoán, giữ nguyên giá trị.\n"
    "Huyết áp dạng SYS/DIA thì ghi: Huyết áp: SYS/DIA\n"
    "Trả về từng dòng theo dạng: tên chỉ số: giá trị"
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
    return "".join(
        c for c in nfd
        if unicodedata.category(c) != "Mn" and c.isalnum()
    )


_ECHO_MARKERS = (
    "diengiatrichotungchiso",     # "điền giá trị cho từng chỉ số"
    "neukhongcotronganhghi",      # "Nếu không có trong ảnh, ghi"
    "giunguyentenchiso",          # "Giữ nguyên tên chỉ số"
)


def extract_vitals(image: np.ndarray) -> str:
    """Single VLM call — returns raw text for parser."""
    try:
        import requests
    except ImportError:
        return ""

    resized = _resize(image)
    img_b64 = _to_b64(resized)
    logger.info("[VLM] %dx%d -> %d KB (model=%s)",
                resized.shape[1], resized.shape[0],
                len(img_b64) * 3 // 4 // 1024, VLM_MODEL)

    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False,
    }
    try:
        resp = requests.post(VLM_ENDPOINT, json=payload, timeout=VLM_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        cleaned = _strip_think(raw)
        logger.info("[VLM] response: %s", cleaned[:300])
        normalized = _normalize_for_match(cleaned)
        hits = sum(1 for m in _ECHO_MARKERS if m in normalized)
        if hits >= 2:
            logger.warning("[VLM] model echoed instructions (%d markers hit)", hits)
            return ""
        return cleaned
    except Exception as e:
        logger.warning("[VLM] error: %s", e)
        return ""


# Legacy compat
def extract_for_device(image: np.ndarray, device_type: str = "handwritten") -> str:
    return extract_vitals(image)

def extract_text(image: np.ndarray, **kwargs) -> str:
    return extract_vitals(image)

async def extract_text_async(image: np.ndarray, **kwargs) -> str:
    return extract_vitals(image)
