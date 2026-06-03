# Deploy OCR HealthCare bằng Docker

Stack: `ocr-server` (FastAPI + YOLO, CUDA GPU 1) + `ngrok` + `monitor`.
**vLLM vẫn chạy systemd** trên host port 8080 — container gọi qua `host.docker.internal`.

## 1. Prereq trên server

```bash
# Docker 20.10+
docker --version

# NVIDIA Container Toolkit (cho --gpus)
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Phải in ra cả 2 V100. Nếu lỗi:
# sudo apt-get install -y nvidia-container-toolkit
# sudo nvidia-ctk runtime configure --runtime=docker
# sudo systemctl restart docker

# vLLM systemd vẫn phải up (8080)
curl -s http://localhost:8080/v1/models | jq .
```

## 2. Setup files

```bash
cd /home/phungkien/OCR_PHR    # working dir = repo root

# Top-level .env cho compose (ngrok + telegram)
cp .env.example .env
nano .env                      # điền NGROK_AUTHTOKEN, TG_BOT_TOKEN, TG_CHAT_ID

# Server runtime env (giữ file .env hiện tại — đã có OCR_API_KEY, RATE_LIMIT…)
ls -la OCR_server/.env

# YOLO model (mount read-only, không bake vào image)
ls -lh bp_detector/best.pt     # phải ~22MB
```

## 3. Build + run

```bash
# Build với UID/GID match host (đảm bảo container ghi được storage volume)
OCR_UID=$(id -u) OCR_GID=$(id -g) docker compose build

# Hoặc dùng default 1000:1000 nếu host user là UID 1000
# docker compose build

# Start nền
docker compose up -d

# Theo dõi
docker compose ps
docker compose logs -f ocr-server   # Ctrl+C để thoát
```

Healthcheck của ocr-server sẽ `up (healthy)` sau ~30 giây. ngrok chờ healthy mới khởi.

## 4. Verify

```bash
# Health
curl -s http://localhost:8502/health | python3 -m json.tool

# Extract qua API key (lấy từ OCR_server/.env)
curl -X POST -H "X-API-Key: <YOUR_KEY>" \
  -F "file=@test_images/sample_bp.jpg" \
  http://localhost:8502/v1/extract | python3 -m json.tool

# Lấy ngrok public URL
curl -s http://localhost:4040/api/tunnels | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

# GPU usage (container dùng GPU 1)
nvidia-smi
```

## 5. Migrate từ systemd

```bash
# Tắt 3 systemd units (vLLM giữ nguyên)
sudo systemctl stop ocr-server ngrok-ocr
sudo systemctl disable ocr-server ngrok-ocr

# Xóa cron monitor cũ (giờ chạy trong container)
crontab -e   # comment dòng */1 * * * * monitor.sh

# Test stack mới
docker compose up -d
docker compose ps
```

Rollback nhanh nếu cần:
```bash
docker compose down
sudo systemctl start ocr-server ngrok-ocr
```

## 6. Operations cheatsheet

| Việc | Lệnh |
|---|---|
| Restart sau khi sửa code | `docker compose up -d --build ocr-server` |
| Restart sau khi sửa `.env` | `docker compose restart ocr-server` |
| Xem log realtime | `docker compose logs -f ocr-server` |
| Vào shell trong container | `docker compose exec ocr-server bash` |
| Chạy pytest | `docker compose exec ocr-server python -m pytest tests/ -v` |
| Stop tất cả | `docker compose down` |
| Stop + xóa volume monitor state | `docker compose down -v` |
| Backup storage | `tar czf storage-$(date +%F).tgz OCR_server/data/ocr_storage` |

## 7. Storage layout

`STORAGE_PATH` mount từ host vào container:

```
Host (persists)                                  Container
./OCR_server/data/ocr_storage/  ─────────────►   /app/OCR_server/data/ocr_storage/
   2026/06/03/                                      2026/06/03/
       <uuid>.jpg                                       <uuid>.jpg
       <uuid>.json                                      <uuid>.json
```

## 8. Network

```
   PHR app
      │ HTTPS
      ▼
[ngrok :443] ──tunnel──► [ocr-ngrok container :4040]
                              │ HTTP qua compose network (ocrnet)
                              ▼
                         [ocr-server :8502]   (GPU 1)
                              │ HTTP host.docker.internal:8080
                              ▼
                         [vLLM systemd :8080] (GPU 0)
```

- `ocrnet` (bridge): ocr-server ↔ ngrok ↔ monitor giao tiếp qua DNS service-name.
- `host.docker.internal:host-gateway`: ocr-server + monitor truy cập host (vLLM).
- Chỉ port 8502 và 4040 expose ra host. 4040 bind `127.0.0.1` only.

## 9. Troubleshoot

**`docker compose up` lỗi `could not select device driver "nvidia"`**
→ Thiếu NVIDIA Container Toolkit. Xem section 1.

**Container ocr-server crash loop, log có `OSError: libGL.so.1`**
→ Dockerfile đã cài `libgl1`. Rebuild: `docker compose build --no-cache ocr-server`.

**Container `up` nhưng `/health` báo `storage_enabled: false`**
→ Volume `./OCR_server/data/ocr_storage` không tồn tại / không write được. Tạo + chown:
```bash
mkdir -p OCR_server/data/ocr_storage && chmod 755 $_
```

**Container gọi vLLM timeout**
→ Test từ trong container: `docker compose exec ocr-server curl -s http://host.docker.internal:8080/v1/models`. Nếu fail, kiểm tra vLLM systemd + firewall localhost.

**ngrok không khởi**
→ `NGROK_AUTHTOKEN` chưa set trong `.env`. Check: `docker compose config | grep NGROK`.

**Image quá to (~6GB)**
→ Đúng — torch+cuda runtime. Để gọn hơn: chuyển sang CPU base (xem option lúc setup) → ~1.5GB nhưng YOLO chậm hơn ~5x.

## 10. CI hint

GitHub Actions có thể build image, push lên GHCR, deploy bằng `docker compose pull && up -d`. Để sau nếu cần.
