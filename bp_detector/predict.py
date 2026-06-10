"""BP monitor digit detector — YOLO11 2-stage pipeline.

Stage 1: detect 3 region SYS / DIA / PUL trên màn hình LCD.
Stage 2: detect digit 0-9 trong từng region đã crop.

Cover cả arm BP monitor (bắp tay) và wrist BP monitor (cổ tay).
Robust hơn approach single-stage cũ:
- Không cần guess row grouping bằng Y-coordinate.
- Mỗi region detect riêng → digit accuracy cao hơn.
- Resilient với perspective + lighting bất thường.
- Nhận label SYS/DIA/PUL trực tiếp → không phụ thuộc thứ tự hiển thị máy.

Model files:
    bp_stage1.pt  ~5MB — region detector (3 class SYS/DIA/PUL)
    bp_stage2.pt  ~5MB — digit detector (10 class 0-9)
"""
import cv2
import json
import numpy as np
import os
import sys
import tempfile
from typing import Optional
from ultralytics import YOLO


def _img_to_b64(img: np.ndarray) -> str:
    """Convert numpy BGR image → base64 JPEG string."""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + __import__('base64').b64encode(buf.tobytes()).decode()


# Cache model load — YOLO(...) đọc weights từ disk + init torch graph (~1-2s).
# Load 1 lần per process, dùng lại cho mọi request. Trước đó load mỗi request
# gây latency YOLO 3s+ — đây là bottleneck chính của fast path.
_MODEL_CACHE: dict[str, YOLO] = {}


def _get_model(model_path: str) -> YOLO:
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def run_prediction(
    image_path: str,
    stage1_path: str = 'bp_stage1.pt',
    stage2_path: str = 'bp_stage2.pt',
    conf: float = 0.35,
) -> dict:
    """Detect BP vitals từ ảnh bằng pipeline YOLO11 2-stage.

    Args:
        image_path: path tới ảnh BP monitor.
        stage1_path: YOLO weights cho region detector (SYS/DIA/PUL).
        stage2_path: YOLO weights cho digit detector (0-9).
        conf: confidence threshold cho cả 2 stage.

    Returns:
        {
            "status":  "success" | "no_detection" | "error",
            "vitals":  {"systolic": "120", "diastolic": "80", "pulse": "70"} (giá trị string hoặc None),
            "regions": {"SYS": [x1,y1,x2,y2], ...} (bbox của từng region),
            "confs":   {"SYS": 0.92, ...} (confidence stage 1),
            "debug_steps": {...} (b64 image các bước, cho dev UI),
            "engine":  "yolo11-2stage"
        }
    """
    # 1. Load 2 model (cached)
    try:
        m1 = _get_model(stage1_path)
        m2 = _get_model(stage2_path)
    except Exception as e:
        return {"error": f"Could not load models: {e}", "status": "error"}

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image", "status": "error"}

    debug_steps = {"1_original": _img_to_b64(img)}

    # 2. Stage 1 — detect SYS/DIA/PUL regions
    r1 = m1(image_path, conf=conf, verbose=False)[0]
    regions: dict[str, Optional[tuple[int, int, int, int]]] = {
        "SYS": None, "DIA": None, "PUL": None,
    }
    confs: dict[str, float] = {}

    for box in r1.boxes:
        cls = r1.names[int(box.cls[0])]
        if cls not in regions:
            continue
        c = float(box.conf[0])
        # Giữ region có conf cao nhất nếu model detect nhiều bbox cho cùng class
        if regions[cls] is None or c > confs.get(cls, 0):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            regions[cls] = (x1, y1, x2, y2)
            confs[cls] = c

    # Vẽ debug: crop overview với 3 region highlight
    overview = img.copy()
    for cls, bbox in regions.items():
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview, cls, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    debug_steps["2_regions"] = _img_to_b64(overview)

    # 3. Stage 2 — detect digits trong từng region crop
    vitals: dict[str, Optional[str]] = {"systolic": None, "diastolic": None, "pulse": None}
    _MAP = {"SYS": "systolic", "DIA": "diastolic", "PUL": "pulse"}

    for cls, bbox in regions.items():
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        r2 = m2(crop, conf=conf, verbose=False)[0]
        digits: list[tuple[float, int]] = []
        for box in r2.boxes:
            label = r2.names[int(box.cls[0])]
            try:
                digit = int(label)
            except ValueError:
                continue
            cx = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            digits.append((cx, digit))

        # Sort theo X (trái → phải) → ghép thành số
        digits.sort()
        value = "".join(str(d) for _, d in digits)
        if value:
            vitals[_MAP[cls]] = value
            debug_steps[f"3_crop_{cls}"] = _img_to_b64(crop)

    found = sum(1 for v in vitals.values() if v is not None)
    return {
        "status":      "success" if found > 0 else "no_detection",
        "vitals":      vitals,
        "regions":     {k: list(v) if v else None for k, v in regions.items()},
        "confs":       confs,
        "debug_steps": debug_steps,
        "engine":      "yolo11-2stage",
    }


if __name__ == "__main__":
    # Usage: python predict.py <image_path> [stage1.pt] [stage2.pt]
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [stage1.pt] [stage2.pt]")
        sys.exit(1)
    img_path = sys.argv[1]
    s1 = sys.argv[2] if len(sys.argv) > 2 else 'bp_stage1.pt'
    s2 = sys.argv[3] if len(sys.argv) > 3 else 'bp_stage2.pt'
    output = run_prediction(img_path, stage1_path=s1, stage2_path=s2)
    # Bỏ debug_steps khỏi stdout cho gọn
    print(json.dumps({k: v for k, v in output.items() if k != "debug_steps"}, indent=2))
