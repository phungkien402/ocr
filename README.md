# OCR HealthCare — Vital Signs Extraction

> Trích xuất 7 chỉ số sinh tồn (mạch, nhiệt độ, huyết áp, nhịp thở, cân nặng, chiều cao, SpO2) từ ảnh thiết bị y tế bằng pipeline hybrid YOLO + VLM. Dùng cho PHR mobile app của bác sĩ đi buồng.

---

## Tính năng

- **Hybrid pipeline:** YOLO custom-trained cho BP monitor (~0.4s) + fallback Qwen2.5-VL-3B (~3s) cho ảnh khác (sổ tay, phiếu kết quả, thiết bị lạ).
- **Production-ready:** API key auth, rate limit, retry tự động + circuit breaker khi VLM crash, monitor Telegram, logs PHI-safe.
- **Dockerized:** 3 container stack (FastAPI + ngrok + monitor), 1 lệnh `docker compose up -d` là chạy.
- **HITL-friendly:** mọi trường có thể null, PHR app hiện form xác nhận cho bác sĩ kiểm tra mắt trước khi ghi EMR.
- **Storage tự động:** ảnh + metadata lưu `YYYY/MM/DD/{uuid}` phục vụ fine-tune sau.

## Kiến trúc

```
PHR mobile app
    │ POST /v1/extract (multipart image + X-API-Key)
    ▼
[ngrok HTTPS] → [ocr-server :8502]
                     │
                     ├── YOLO (best.pt) ─┐
                     │   nếu đủ SYS+DIA+PUL → skip VLM (0.4s)
                     │
                     └── fallback ─► [vLLM :8080]
                                     Qwen2.5-VL-3B (3s)
                                          │
                            parse → validate → 7-field JSON
                                          │
                            save image + metadata
                                          │
                              Response → PHR (HITL) → EMR
```

## Stack

| Component | Tech |
|---|---|
| Web framework | FastAPI + uvicorn + slowapi |
| VLM | vLLM serving Qwen/Qwen2.5-VL-3B-Instruct (FP16, 1× V100) |
| YOLO | Ultralytics + custom `best.pt` (Roboflow trained) |
| HTTP client | httpx async với retry + circuit breaker |
| Storage | Filesystem `YYYY/MM/DD/{uuid}.{jpg,json}` |
| Container | Docker Compose (pytorch CUDA 12.1 + Alpine monitor) |
| Tunnel | ngrok HTTPS (free tier) |
| Monitoring | Telegram bot alert + journalctl |

## Quick start

Yêu cầu: Ubuntu 22.04/24.04, NVIDIA GPU ≥16GB VRAM, Docker + NVIDIA Container Toolkit.

```bash
git clone -b sub_main https://github.com/phungkien402/ocr.git OCR_PHR
cd OCR_PHR

# Copy YOLO model (không có trong repo, ~22MB)
scp <source>:best.pt bp_detector/

# Setup secrets
cp .env.example .env                          # NGROK_AUTHTOKEN, TG_*
cp OCR_server/.env.example OCR_server/.env    # OCR_API_KEY (generate)

# Build + up
OCR_UID=$(id -u) OCR_GID=$(id -g) docker compose build
docker compose up -d

# Verify
curl -s http://localhost:8502/health | python3 -m json.tool
```

Chi tiết: xem [`DEPLOY_DOCKER.md`](DEPLOY_DOCKER.md).  
Deploy server mới hoặc bệnh viện khác: xem [`MIGRATE.md`](MIGRATE.md).

## API

### `POST /v1/extract`

```http
POST /v1/extract HTTP/1.1
Host: <endpoint>
X-API-Key: <64-hex-key>
Content-Type: multipart/form-data

file=<JPG/PNG ≤10MB>
```

**Response 200:**

```json
{
  "request_id": "uuid-v4",
  "mach": 70,
  "nhiet_do": 36.5,
  "huyet_ap": {"tam_thu": 120, "tam_truong": 80},
  "nhip_tho": 16,
  "can_nang": 67,
  "chieu_cao": 170,
  "spo2": 98
}
```

Mọi trường có thể `null`. Client BẮT BUỘC cho bác sĩ xác nhận trước khi ghi EMR.

**Error codes:**

| Code | Ý nghĩa | Action |
|---|---|---|
| 400 | File sai format / quá lớn | Compress + retry |
| 401 | API key sai / thiếu | Check config |
| 429 | Rate limit (300/min) | Backoff + retry |
| 500 | Pipeline crash | Hiện form trống cho gõ tay |

### Endpoints khác

- `GET /health` — public, trả status + circuit breaker state
- `GET /demo` — HTML demo flow PHR↔OCR↔HIS + sample code 4 ngôn ngữ
- `GET /test` — minimal test UI cho `/v1/extract`
- `POST /process` — legacy debug endpoint, full internal result (raw VLM text, debug steps)

## Cấu trúc repo

```
OCR_HealthCare/
├── OCR_server/
│   ├── web_app.py                # FastAPI app + auth + rate limit + storage
│   ├── ocr_vitals/
│   │   ├── main.py               # Pipeline orchestrator (YOLO → VLM)
│   │   ├── ocr_engine.py         # VLM client wrapper
│   │   ├── vlm_client.py         # Retry + circuit breaker
│   │   ├── preprocessor.py       # Image resize
│   │   ├── parser.py             # 3-tier parser (JSON / label / regex)
│   │   ├── validator.py          # Range validation
│   │   ├── storage.py            # Image + metadata persistence
│   │   └── config.py             # Field constants
│   ├── static/                   # /, /test, /demo HTML
│   └── tests/                    # 64 unit tests
├── bp_detector/
│   ├── best.pt                   # YOLO weights (gitignored)
│   ├── predict.py                # YOLO inference + preprocessing
│   └── app.py                    # Standalone YOLO UI (port 8503)
├── docker/
│   ├── Dockerfile.monitor
│   └── monitor.sh
├── Dockerfile                    # ocr-server (CUDA + YOLO)
├── docker-compose.yml            # 3-service stack
├── loadtest.py                   # Async load test
├── DEPLOY_DOCKER.md              # Setup + troubleshoot guide
├── DEPLOY_HTTPS.md               # Nginx + Let's Encrypt
├── MIGRATE.md                    # Server migration playbook
└── PROJECT_STATE.md              # Detailed state snapshot
```

## Development

```bash
# Local run (không Docker, cần torch + GPU)
pip install -r OCR_server/requirements.txt
cd OCR_server
uvicorn web_app:app --reload --port 8502

# Run tests
python -m pytest tests/ -v

# Load test
python loadtest.py --url http://localhost:8502 --image test_images/bb.jpg --c 20 --duration 30
```

### Environment variables

Xem [`OCR_server/.env.example`](OCR_server/.env.example) cho full list. Key vars:

- `OCR_API_KEY` — 64-hex auth token
- `STORAGE_PATH` — image + metadata directory
- `VLM_ENDPOINT` / `VLM_MODEL` / `VLM_TIMEOUT_SECONDS`
- `VLM_MAX_RETRIES` / `VLM_CB_THRESHOLD` / `VLM_CB_COOLDOWN` — retry + circuit breaker tuning
- `MAX_UPLOAD_BYTES` / `RATE_LIMIT`

## Performance (1× V100)

| Path | p50 | p95 | Throughput |
|---|---|---|---|
| YOLO fast (BP monitor) | 0.4s | 0.5s | ~12 req/s |
| VLM (ảnh khác) | 3.0s | 4.0s | ~0.33 req/s |
| Mixed 80/20 BP/VLM | ~1s | ~3s | **3,600 req/giờ sustainable** |

Chi tiết: [`PROJECT_STATE.md`](PROJECT_STATE.md) section 10.


**Pending (priority):**
1. Benchmark accuracy trên 50-100 ảnh thật từ phòng khám
2. Cloudflare Tunnel thay ngrok (URL cố định + HTTPS)
3. HTTPS production (nginx + Let's Encrypt)
4. Disclaimer pháp lý review với lawyer VN
5. GitHub Actions CI build/push image

## Known limitations

1. Single API key — chưa multi-tenant cho nhiều BV
2. URL ngrok đổi mỗi lần restart (sẽ chuyển Cloudflare Tunnel)
3. Storage không có retention policy — disk sẽ đầy dần
4. Wrist BP monitor chưa detect được (YOLO chỉ train arm monitor)
5. Vietnamese handwriting accuracy thấp với VLM 3B
6. vLLM single GPU sequential — peak throughput cap ~0.33 req/s

