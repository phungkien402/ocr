# OCR HealthCare — Notes & Research

## Tối ưu tốc độ

### Vấn đề hiện tại
- Qwen2-VL-2B CPU laptop: **20-50s/ảnh**
- Qwen2-VL-2B GPU server (RTX 3060 Ti): **~1-3s/ảnh**

### Hướng tối ưu (theo thứ tự ưu tiên)

| # | Action | Latency BP | Effort | Note |
|---|--------|-----------|--------|------|
| 1 | **Skip VLM khi YOLO success** | 20s → 0.2s | 10 min | Làm ngay |
| 2 | Collect 1000+ ảnh thật | — | Weeks | Bottleneck chính |
| 3 | Fine-tune SmolVLM 256M | CPU < 2s | Days | Sau khi có data |
| 4 | TensorRT cho YOLO | GPU: 200ms → 4ms | Hours | Cần GPU |
| 5 | vLLM serving | Throughput concurrent | Days | Production scale |

### Skip VLM khi YOLO success
```python
# main.py — process_image_async
yolo_vitals = _try_yolo(image_path)
if yolo_vitals:
    # YOLO got BP + mach → skip VLM, return immediately
    return _build_result(filename, "", yolo_vitals)
# else: fallback to VLM
raw_text = await extract_text_async(...)
```

### SmolVLM 256M fine-tune (Phase 2)
- Model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Nhỏ hơn Qwen2-VL-2B: 8x, nhanh hơn: ~15x trên CPU
- Cần ~1000 ảnh thật có label để fine-tune hiệu quả
- Sau fine-tune: < 2s trên CPU laptop

### vLLM (Production)
- Hybrid DP+TP parallelism: ViT Data Parallel + LLM Tensor Parallel
- Giảm TTFT rõ rệt khi có concurrent requests
- Ref: https://discuss.vllm.ai/t/speeding-up-vllm-inference-for-qwen2-5-vl/615

---

## Tối ưu độ chính xác

### Benchmarks tham khảo
- 2025 MDPI paper: 98.3% accuracy, reject 17.9% ảnh kém chất lượng
- YOLOv8s INT8 quantized: sweet spot accuracy/speed/size cho BP digit detection
- F1-score digit localization: 80%, classification accuracy: 89.7%

### Hướng cải thiện
1. **Dataset thực tế** — quan trọng nhất. Collect ảnh từ bệnh viện/phòng khám, label thủ công, train lại YOLO
2. **Device-specific prompts** — đã implement. BP monitor / SpO2 / thermometer / glucometer / vital signs monitor / lab report
3. **Two-stage pipeline** — detect device → crop ROI → OCR. Đã có `localize_lcd_frame` nhưng cần fine-tune threshold
4. **Quality filter** — reject ảnh blur/tối trước khi đưa vào model (tránh hallucination)

### Wrist monitor (Beurer BC32, v.v.)
- YOLO hiện tại train trên arm BP monitor, không detect wrist monitor
- Fix: thêm wrist monitor vào training data YOLO
- Hoặc: để VLM handle (với device prompt đủ context)

---

## Kiến trúc hiện tại

```
Image (JPG/PNG)
  ↓
FastAPI :8502 (web_app.py)
  ↓
main.py — pipeline
  ├── YOLO path (bp_detector/predict.py)
  │     perspective_correct → localize_lcd_frame → enhance → YOLOv8
  │     → SYS/DIA/PUL + digit_crop debug image
  │
  └── VLM path (ocr_engine.py)
        preprocessor → llama-server :8080
        Qwen2-VL-2B Q4_K_M + mmproj-f16
        Device classify + extract JSON
  ↓
YOLO overrides VLM for huyet_ap + mach
  ↓
parser.py → validator.py
  ↓
JSON response + debug_steps thumbnails
```

---

## Roadmap

### Phase 1 (current)
- [x] YOLO digit detector cho BP monitor
- [x] Qwen2-VL-2B VLM fallback
- [x] Device classification trong prompt
- [x] Preprocessing steps debug UI
- [ ] Skip VLM khi YOLO success

### Phase 2 (~1000 ảnh)
- [ ] Fine-tune SmolVLM 256M
- [ ] Stable API format `{success, data, warnings, meta}`
- [ ] PHR integration + user feedback UI

### Phase 3 (>5000 ảnh)
- [ ] Wrist monitor support
- [ ] SpO2, thermometer, glucometer, lab report
- [ ] FHIR/HL7 mapping → HIS

### Phase 4 (>10k ảnh)
- [ ] Custom lightweight model (~20MB, on-device Android)
- [ ] TensorRT YOLO cho GPU deployment

---

## Server GPU (n1.ckey.vn)

- SSH: `ssh root@n1.ckey.vn -p 2446` (Admin@123)
- GPU: RTX 3060 Ti 8GB
- OS: Ubuntu 22.04, CUDA 12.4
- Project: `/root/OCR_server/`
- Models: `/root/models/`
  - `Qwen3VL-2B-Instruct-Q4_K_M.gguf` (Qwen3-VL, llama-cpp-python không support vision)
- llama-server binary: `/root/llama-b9415/llama-server` (CPU-only)
- llama-cpp-python: v0.3.23 (không support Qwen3-VL vision)
- **Issue**: Cần build llama.cpp CUDA hoặc dùng model khác trên GPU server

## Local (Windows laptop)

- llama-server: `D:\llama\llama-b9434\llama-server.exe`
- Models: `D:\llama\models\`
  - `Qwen2-VL-2B-Instruct-Q4_K_M.gguf`
  - `mmproj-Qwen2-VL-2B-Instruct-f16.gguf`
- Start: `& "D:\llama\llama-b9434\llama-server.exe" -m ... --mmproj ... --port 8080`
- OCR server: `cd E:\healthcare\OCR_HealthCare\OCR_server && uvicorn web_app:app --port 8502`

---

## Kế hoạch train model nhỏ from scratch

### Mục tiêu
Model ~1-5MB, chạy on-device, không cần VLM, inference < 50ms.

### Nguồn data
- Roboflow dataset: ~2100 ảnh BP monitor, labels = digit bounding boxes (class 0-9)
- Fork community đã lên ~2100 ảnh + augmentation
- Extract từ labels: ~12,000+ digit crop images (64×64)

### Option A — Tiny CNN digit classifier (~500KB) ⭐ Recommended first
```
Pipeline:
  Ảnh → localize_lcd_frame → enhance
      → connected components / contours → digit regions
      → sort by Y (rows) → crop từng digit (64×64)
      → CNN classify (0-9)
      → group by row → "97" "56" "77"
      → map row 0=SYS, row 1=DIA, row 2=PUL

Architecture:
  Input: 64×64 grayscale
  Conv(32, 3×3) → BN → ReLU → MaxPool
  Conv(64, 3×3) → BN → ReLU → MaxPool
  Conv(128, 3×3) → BN → ReLU → MaxPool
  Flatten → FC(256) → Dropout(0.3) → FC(10)
  Output: class 0-9

Training:
  - Extract digit crops từ YOLO annotations
  - Augment: rotation ±10°, brightness ±30%, noise, blur
  - ~12k crops → split 80/20
  - 20-30 epochs, ~15 phút trên CPU
  - Target accuracy: >95%
  - Model size: ~500KB
```

### Option B — CRNN LCD sequence recognition (~2MB)
```
Pipeline:
  Ảnh → localize_lcd_frame → resize (128×64)
      → CRNN → CTC decode → "97/56/77"
      → parse → JSON

Architecture:
  Input: 128×64 grayscale
  CNN backbone: 3 VGG-style blocks → feature map (32×1×512)
  BiLSTM(256) × 2 layers
  FC → CTC loss
  Output: sequence "97 56 77"

Training:
  - Generate sequence labels từ YOLO bbox annotations
    (sort by Y → rows, sort by X → digits, concatenate)
  - ~2100 images, augment → ~8000 samples
  - 50 epochs, ~1h trên GPU
  - Model size: ~2MB
```

### Option C — End-to-end regression (~1MB) — harder
```
Input: 224×224
→ Tiny CNN backbone (MobileNet-like, 8 channels only)
→ 3 regression heads: SYS, DIA, PUL
Output: 3 float values

Cần nhiều data hơn, khó converge với 2100 ảnh.
Không recommend cho MVP.
```

### Kế hoạch thực hiện
1. Download dataset từ Roboflow (YOLO format)
2. Script extract digit crops từ bounding box annotations
3. Train Option A (CNN classifier) trên Colab (free GPU)
4. Evaluate trên test set
5. Export ONNX → tích hợp vào bp_detector/predict.py thay thế YOLOv8
6. Nếu Option A accuracy > 92% → ship. Nếu không → thử Option B.

---

## API Packaging (PHR Integration)

### Flow mục tiêu
```
PHR App (mobile/web)
    ↓  POST /v1/extract  (multipart image)
OCR Module API
    ↓  YOLO + VLM pipeline
    ↓
{
  "success": true,
  "data": {
    "device": "blood_pressure_monitor",
    "readings": {
      "systolic":  {"value": 97,   "unit": "mmHg"},
      "diastolic": {"value": 56,   "unit": "mmHg"},
      "pulse":     {"value": 77,   "unit": "bpm"}
    },
    "confidence": "high"
  },
  "warnings": [],
  "meta": {
    "engine": "yolo+vlm",
    "processing_time_s": 1.5,
    "model_version": "1.0.0"
  }
}
    ↓  User confirm/edit in PHR UI
    ↓
{
  "resourceType": "Observation",
  "code": {"coding": [{"system": "http://loinc.org", "code": "55284-4"}]},
  "component": [
    {"code": {"coding": [{"code": "8480-6"}]}, "valueQuantity": {"value": 97, "unit": "mmHg"}},
    {"code": {"coding": [{"code": "8462-4"}]}, "valueQuantity": {"value": 56, "unit": "mmHg"}}
  ]
}
    ↓
HIS (FHIR R4)
```

### API versioning
- `/v1/extract` — current stable
- Response format cố định, không đổi khi swap model
- Model swap: Qwen2-VL → SmolVLM → custom model → app không cần sửa

---

## References

- [Automated Digit Recognition for BP Monitor - MDPI 2025](https://www.mdpi.com/1999-4893/18/7/377)
- [OCR of Home Blood Pressure - AJH 2025](https://academic.oup.com/ajh/advance-article/doi/10.1093/ajh/hpaf227/8328023)
- [CNN-Based LCD Transcription - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8177819/)
- [SmolVLM2 fine-tune for OCR - Roboflow](https://blog.roboflow.com/base-vs-fine-tuned-smolvlm2-ocr/)
- [Roboflow dataset: blood-pressure-monitor-display](https://universe.roboflow.com/final-project-cwtfb/blood-pressure-monitor-display)
- [bartowski/Qwen2-VL-2B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF)
