import cv2
import numpy as np
from ultralytics import YOLO
import json
import sys
import tempfile
import os

def _order_corners(pts):
    """Sắp xếp 4 điểm góc: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)], # bottom-left
    ], dtype=np.float32)


def perspective_correct(img: np.ndarray) -> np.ndarray:
    """Tìm màn hình LCD trong ảnh và warp thẳng 90 độ.

    Dùng edge detection + contour để tìm hình chữ nhật lớn nhất.
    Nếu không tìm được → trả lại ảnh gốc (không lỗi).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur nhẹ để giảm noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge
    edges = cv2.Canny(blur, 30, 120)

    # Dilate để nối đường viền bị đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Sắp xếp theo diện tích giảm dần
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    screen_cnt = None
    for cnt in contours[:5]:  # chỉ xét 5 contour lớn nhất
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)

        # Cần đúng 4 góc, diện tích > 10% ảnh
        if len(approx) == 4 and area > h * w * 0.10:
            screen_cnt = approx
            break

    if screen_cnt is None:
        return img  # không tìm được → giữ nguyên

    # Warp perspective
    src = _order_corners(screen_cnt)
    # Tính kích thước output từ khoảng cách các cạnh
    wW = int(max(
        np.linalg.norm(src[1] - src[0]),
        np.linalg.norm(src[2] - src[3])
    ))
    wH = int(max(
        np.linalg.norm(src[3] - src[0]),
        np.linalg.norm(src[2] - src[1])
    ))

    if wW < 50 or wH < 50:
        return img  # quá nhỏ, bỏ qua

    dst = np.array([
        [0, 0], [wW - 1, 0],
        [wW - 1, wH - 1], [0, wH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (wW, wH))
    return warped


def enhance_lcd_image(img: np.ndarray) -> np.ndarray:
    """Image enhancement cho YOLO input:
    - Bilateral filter: giữ edge digit sắc nét, bỏ noise
    - Gamma correction: chuẩn hóa độ sáng
    - Unsharp mask: tăng độ sắc nét digit

    NOTE: KHÔNG dùng binary threshold vì YOLO train trên ảnh tự nhiên.
    Binary chỉ phù hợp cho CNN classifier (như trong paper PMC8177819).
    """
    # Bilateral filter trên từng channel màu — giữ màu, giảm noise
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Gamma correction tự động theo độ sáng trung bình
    mean_bright = np.mean(cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)) / 255.0
    if mean_bright < 0.35:
        gamma = 0.55   # ảnh tối → sáng lên
    elif mean_bright > 0.75:
        gamma = 1.4    # ảnh quá sáng → tối xuống
    else:
        gamma = 1.0
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(filtered, lut)

    # Unsharp mask — làm sắc nét cạnh digit
    blur = cv2.GaussianBlur(corrected, (0, 0), 2.0)
    sharpened = cv2.addWeighted(corrected, 1.7, blur, -0.7, 0)

    return sharpened


def localize_lcd_frame(img: np.ndarray) -> np.ndarray:
    """Crop vùng LCD display từ ảnh theo paper PMC8177819.
    Dùng contour + aspect ratio để tìm LCD frame.
    Fallback về ảnh gốc nếu không tìm được.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bilateral + adaptive threshold để tăng viền LCD
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    binary = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate để nối đường viền
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    h_img, w_img = img.shape[:2]
    best = None
    best_score = 0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(cnt)
        if area < h_img * w_img * 0.05:  # bỏ qua contour quá nhỏ
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0

        # LCD BP monitor thường có aspect ratio 0.8–2.5 (landscape hoặc portrait)
        if 0.6 <= aspect <= 3.0:
            # Score = diện tích * gần trung tâm ảnh
            cx, cy = x + w // 2, y + h // 2
            dist_center = abs(cx - w_img // 2) + abs(cy - h_img // 2)
            score = area / (dist_center + 1)
            if score > best_score:
                best_score = score
                best = (x, y, w, h)

    if best is None:
        return img

    x, y, w, h = best
    # Thêm padding 5%
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cropped


def _img_to_b64(img: np.ndarray) -> str:
    """Convert numpy BGR image → base64 JPEG string."""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + __import__('base64').b64encode(buf.tobytes()).decode()


def preprocess_for_yolo(image_path: str, debug: bool = False):
    """Pipeline preprocessing theo PMC8177819 + upscale:
    1. Perspective correction
    2. LCD frame localization (crop display area)
    3. Image enhancement (bilateral + gamma + adaptive threshold)
    4. Upscale nếu nhỏ

    Args:
        debug: nếu True, trả về (path, dict của các bước dạng base64)
    Returns:
        str (path) nếu debug=False
        (str, dict) nếu debug=True
    """
    img = cv2.imread(image_path)
    if img is None:
        return (image_path, {}) if debug else image_path

    steps = {}
    if debug:
        steps['1_original'] = _img_to_b64(img)

    # 1. Perspective correction
    img = perspective_correct(img)
    if debug:
        steps['2_perspective'] = _img_to_b64(img)

    # 2. Crop LCD frame
    img = localize_lcd_frame(img)
    if debug:
        steps['3_lcd_crop'] = _img_to_b64(img)

    # 3. Enhancement
    img = enhance_lcd_image(img)
    if debug:
        steps['4_enhanced'] = _img_to_b64(img)

    h, w = img.shape[:2]

    # 4. Upscale nếu nhỏ
    min_dim = min(h, w)
    if min_dim < 640:
        scale = 640 / min_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if debug:
        steps['5_final'] = _img_to_b64(img)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(tmp.name, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    tmp.close()
    return (tmp.name, steps) if debug else tmp.name


# Cache model load — YOLO(...) đọc weights từ disk + init torch graph (~1-2s).
# Load 1 lần per process, dùng lại cho mọi request. Trước đó load mỗi request
# gây latency YOLO 3s+ — đây là bottleneck chính của fast path.
_MODEL_CACHE: dict[str, YOLO] = {}


def _get_model(model_path: str) -> YOLO:
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def run_prediction(image_path, model_path='best.pt'):
    # 1. Load Model (cached)
    try:
        model = _get_model(model_path)
    except Exception as e:
        return {"error": f"Could not load model: {e}"}

    # 2. Preprocess + Run Inference
    preprocess_result = preprocess_for_yolo(image_path, debug=True)
    processed_path, debug_steps = preprocess_result
    is_temp = processed_path != image_path
    proc_img = cv2.imread(processed_path)  # read before unlink
    results = model(processed_path, conf=0.35, verbose=False)
    if is_temp:
        os.unlink(processed_path)
    result = results[0]

    dets = []
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        if label == '10':
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        digit = int(label)
        dets.append({'digit': digit, 'x': cx, 'y': cy})
        all_x1.append(x1); all_y1.append(y1)
        all_x2.append(x2); all_y2.append(y2)

    if not dets:
        return {"error": "No digits detected"}

    # Tight crop from digit bounding boxes + padding
    if proc_img is not None:
        h_img, w_img = proc_img.shape[:2]
        pad_x = int((max(all_x2) - min(all_x1)) * 0.15)
        pad_y = int((max(all_y2) - min(all_y1)) * 0.25)
        cx1 = max(0, int(min(all_x1)) - pad_x)
        cy1 = max(0, int(min(all_y1)) - pad_y)
        cx2 = min(w_img, int(max(all_x2)) + pad_x)
        cy2 = min(h_img, int(max(all_y2)) + pad_y)
        digit_crop = proc_img[cy1:cy2, cx1:cx2]
        if digit_crop.size > 0:
            debug_steps['6_digit_crop'] = _img_to_b64(digit_crop)

    # 3. Row Grouping Logic
    # Sort by Y-coordinate to identify rows top-to-bottom
    dets.sort(key=lambda d: d['y'])
    
    row_gap = 40  # Adjust this if rows are being merged or split incorrectly
    rows, cur_row = [], [dets[0]]
    
    for d in dets[1:]:
        if abs(d['y'] - cur_row[-1]['y']) < row_gap:
            cur_row.append(d)
        else:
            rows.append(sorted(cur_row, key=lambda d: d['x']))
            cur_row = [d]
    rows.append(sorted(cur_row, key=lambda d: d['x']))

    # 4. Format Output
    final_values = []
    for r in rows:
        val = ''.join(str(d['digit']) for d in r)
        final_values.append(val)

    return {
        "status": "success",
        "raw_rows": final_values,
        "vitals": {
            "systolic": final_values[0] if len(final_values) > 0 else None,
            "diastolic": final_values[1] if len(final_values) > 1 else None,
            "pulse": final_values[2] if len(final_values) > 2 else None
        },
        "debug_steps": debug_steps,
    }

if __name__ == "__main__":
    # Usage: python predict.py path/to/image.jpg
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
    else:
        img_path = sys.argv[1]
        mdl_path = sys.argv[2] if len(sys.argv) > 2 else 'best.pt'
        output = run_prediction(img_path, mdl_path)
        print(json.dumps(output, indent=2))