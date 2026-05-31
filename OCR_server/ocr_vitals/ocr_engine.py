"""OCR engine — single universal extraction prompt."""

import base64, logging, re
import cv2
import numpy as np
from .preprocessor import preprocess_for_vlm_array

logger = logging.getLogger(__name__)

LLAMA_ENDPOINT  = "http://localhost:8080/v1/chat/completions"
BACKEND         = "llama"
OLLAMA_MODEL    = "qwen2.5vl:3b"
MAX_IMAGE_DIM   = 1024
OLLAMA_TIMEOUT  = 120

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


def extract_vitals(image: np.ndarray) -> str:
    """Single VLM call — returns raw text for parser."""
    try:
        import requests
    except ImportError:
        return ""

    resized = _resize(image)
    img_b64 = _to_b64(resized)
    logger.info("[VLM] %dx%d -> %d KB", resized.shape[1], resized.shape[0],
                len(img_b64) * 3 // 4 // 1024)

    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 200,
        "stream": False,
    }
    try:
        resp = requests.post(LLAMA_ENDPOINT, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        cleaned = _strip_think(raw)
        logger.info("[VLM] response: %s", cleaned[:300])
        # Detect model echoing instructions instead of reading image
        echo_markers = ["ten chi so", "moi dong mot", "khong giai thich"]
        if sum(m in cleaned.lower().replace(" ", "") for m in ["tenchisogiátri", "moidongmotchiso", "khonggiaithich"]) >= 2:
            logger.warning("[VLM] model echoed instructions")
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
