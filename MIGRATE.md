# Migrate OCR HealthCare sang server mới

Playbook chuyển toàn bộ stack (OCR + vLLM + ngrok + monitor) từ server hiện tại sang server khác.

**Thời gian:** ~1.5–2h (bao gồm download Qwen 6GB).

---

## 1. Cần move những gì

| Loại | Cách | Note |
|---|---|---|
| **Code** | `git clone` từ branch `sub_main` | Đơn giản nhất, có versioning |
| **Secrets** (`.env`, `OCR_server/.env`) | scp / VS Code copy, **hoặc generate mới** | Recommend generate mới — coi key cũ đã compromised |
| **YOLO model** `bp_detector/best.pt` | scp (22MB) | Train offline, KHÔNG trong repo |
| **Qwen2.5-VL-3B** | Auto-download lần đầu | ~6GB từ HuggingFace, mất 10–15 phút |
| **Image storage** (`OCR_server/data/ocr_storage/`) | rsync nếu cần history | Có thể start fresh nếu chỉ test |
| **systemd `vllm-qwen.service`** | scp + edit paths | User / path khác phải sửa |
| **monitor state** (`docker volume`) | KHÔNG cần | State machine tự rebuild |

---

## 2. Prereq trên server mới

Ubuntu 22.04 / 24.04, ít nhất 1 NVIDIA GPU ≥16GB VRAM (Qwen2.5-VL-3B FP16 cần ~10GB).

### 2.1 NVIDIA driver

```bash
nvidia-smi    # nếu in GPU info thì skip
# Nếu chưa có:
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
nvidia-smi    # phải in tên GPU
```

Driver tối thiểu **≥ 525.60.13** (CUDA 12.1 runtime của image cần).

### 2.2 Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# Logout/login lại để áp dụng group, hoặc:
newgrp docker
docker --version
```

### 2.3 NVIDIA Container Toolkit

Ubuntu **không** có sẵn package này, phải add NVIDIA repo:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# → phải in bảng GPU info từ trong container
```

### 2.4 Python cho vLLM (chạy systemd, KHÔNG dockerize)

```bash
sudo apt-get install -y python3 python3-pip python3-venv git
pip install --user vllm    # hoặc dùng venv
# Verify
python3 -c "from vllm.entrypoints.openai import api_server; print('vllm OK')"
```

---

## 3. Migration sequence

### 3.1 Clone repo + lấy YOLO model

```bash
cd ~
git clone -b sub_main https://github.com/phungkien402/ocr.git OCR_PHR
cd OCR_PHR

# Copy best.pt từ server cũ (không có trong git, ~22MB)
scp <old_user>@<old_server>:/home/<old_user>/OCR_PHR/bp_detector/best.pt bp_detector/
ls -lh bp_detector/best.pt    # phải ~22MB
```

### 3.2 Generate secrets mới

**KHÔNG copy key từ server cũ** — coi như đã compromised, đặc biệt nếu trước đó từng lộ qua git.

```bash
# OCR_API_KEY (cho FastAPI)
echo "OCR_API_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

# Telegram bot: tạo bot mới qua @BotFather hoặc revoke + regen token cũ
# ngrok authtoken: lấy từ https://dashboard.ngrok.com (free hoặc upgrade)
```

Tạo 2 file env từ template:

```bash
# Top-level .env (cho docker compose)
cp .env.example .env
nano .env
# Điền: NGROK_AUTHTOKEN, TG_BOT_TOKEN, TG_CHAT_ID

# OCR_server/.env (cho FastAPI runtime)
cp OCR_server/.env.example OCR_server/.env
nano OCR_server/.env
# Điền: OCR_API_KEY (mới), giữ default phần khác
# Sửa STORAGE_PATH=/app/OCR_server/data/ocr_storage (path trong container)
```

### 3.3 Setup vLLM systemd

```bash
sudo tee /etc/systemd/system/vllm-qwen.service > /dev/null <<EOF
[Unit]
Description=vLLM serving Qwen2.5-VL-3B
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/tmp
Environment="PATH=/home/$(whoami)/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="HF_HOME=/home/$(whoami)/.cache/huggingface"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --host 0.0.0.0 --port 8080 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=1
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen

# Đợi load model — lần đầu phải download 6GB từ HuggingFace
sudo journalctl -u vllm-qwen -f
# Ctrl+C khi thấy "Application startup complete" hoặc "Uvicorn running on http://0.0.0.0:8080"

# Verify
curl -s http://localhost:8080/v1/models | python3 -m json.tool
# → phải trả {"object":"list","data":[{"id":"Qwen/Qwen2.5-VL-3B-Instruct",...
```

### 3.4 Adjust docker-compose.yml nếu GPU layout khác

Nếu server mới chỉ có **1 GPU**, sửa `device_ids` trong `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]    # ← từ "1" thành "0" nếu chỉ có 1 GPU
          capabilities: [gpu]
```

Cùng GPU với vLLM cũng OK (Qwen ~10GB + YOLO ~2GB, fit V100 16GB).

### 3.5 Build + run Docker stack

```bash
# Build với UID match host (storage volume write OK)
OCR_UID=$(id -u) OCR_GID=$(id -g) docker compose build

# Start
docker compose up -d
docker compose ps
```

Lần đầu build mất ~5–10 phút (download pytorch CUDA image ~3.6GB).

### 3.6 Smoke test

```bash
# Health
curl -s http://localhost:8502/health | python3 -m json.tool
# → phải có "vlm_circuit": {"state":"closed",...}

# Extract YOLO fast path
KEY=$(grep OCR_API_KEY OCR_server/.env | cut -d= -f2)
curl -s -X POST -H "X-API-Key: $KEY" -F "file=@test_images/bb.jpg" \
  http://localhost:8502/v1/extract | python3 -m json.tool
# → mach=70, BP=118/78

# Extract VLM path (ảnh non-BP)
curl -s -X POST -H "X-API-Key: $KEY" -F "file=@test_images/test_08_handwritten.png" \
  http://localhost:8502/v1/extract | python3 -m json.tool
# → có vài chỉ số non-null

# ngrok public URL
curl -s http://localhost:4040/api/tunnels | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

### 3.7 (Optional) Migrate storage history

Chỉ làm nếu cần giữ ảnh + metadata cũ cho fine-tune sau này. Có thể tốn vài GB.

```bash
rsync -avz --progress \
  <old_user>@<old_server>:/home/<old_user>/OCR_PHR/OCR_server/data/ocr_storage/ \
  ./OCR_server/data/ocr_storage/

# Verify
du -sh OCR_server/data/ocr_storage
ls OCR_server/data/ocr_storage/2026/ | head
```

### 3.8 Update PHR client

- Đổi endpoint URL (URL ngrok mới hoặc Cloudflare Tunnel).
- Cập nhật `OCR_API_KEY` mới qua kênh bảo mật (KHÔNG Slack/email công khai).

---

## 4. Things to watch out for

### GPU layout
- 1 GPU vs 2 GPU: sửa `device_ids` trong compose.
- Card khác V100: kiểm tra `gpu-memory-utilization` trong systemd unit (Qwen-3B FP16 cần ~10GB).
- Driver < 525: image CUDA 12.1 fail. Upgrade driver hoặc đổi base image.

### Path hard-code
- systemd `vllm-qwen.service` có `User=`, `WorkingDirectory=`, `HF_HOME=` chứa user name → script ở 3.3 dùng `$(whoami)` để auto-detect.
- `docker-compose.yml` dùng path tương đối (`./OCR_server/data/...`) nên OK.

### HF model cache
- Lần đầu `vllm-qwen` tải Qwen 6GB vào `$HF_HOME` (default `~/.cache/huggingface`).
- Phải có ≥10GB free trên disk chứa cache.
- Nếu disk nhỏ: đổi `HF_HOME` sang disk khác trong systemd unit.

### Firewall
- Port 8502 (ocr-server): chỉ expose nếu test trực tiếp; production qua ngrok/Cloudflare Tunnel.
- Port 8080 (vLLM): KHÔNG expose ra ngoài, chỉ localhost.
- Port 4040 (ngrok admin): bind 127.0.0.1.

### Conflict port 8080
- Nếu server mới đã có service khác chiếm 8080 (xảy ra ở server cũ với `ehc-helpdesk`):
  - Stop service kia, hoặc
  - Đổi vLLM sang port khác (vd 8000): sửa `--port` trong systemd unit + `VLM_ENDPOINT` trong `OCR_server/.env` + override trong `docker-compose.yml`.

### CUDA driver mismatch
- Verify: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Nếu fail: cần upgrade driver hoặc đổi `Dockerfile` base image sang CUDA version match driver.

---

## 5. Verify checklist sau khi migrate

```bash
# 1. Tất cả container healthy
docker compose ps | grep -E "healthy|Up"

# 2. Health endpoint OK
curl -s http://localhost:8502/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok' and d['storage_enabled'] and d['auth_enabled']; print('health OK')"

# 3. vLLM model loaded
curl -s http://localhost:8080/v1/models | grep -q "Qwen2.5-VL-3B" && echo "vLLM OK"

# 4. YOLO + VLM extract đều work (test 2 loại ảnh)

# 5. ngrok tunnel up
curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

# 6. Monitor container running
docker compose logs --tail=5 monitor | grep -E "ocr=up|vlm=up"

# 7. Tests pass (nếu add pytest vào image)
docker compose exec ocr-server python -m pytest tests/ -v
```

---

## 6. Rollback nếu fail

Server cũ chưa tắt đến khi migrate xong:

```bash
# Trên server CŨ (đảm bảo vẫn serve):
docker compose ps
curl -s http://localhost:8502/health

# PHR client tạm thời point ngược về URL ngrok của server cũ.
```

Sau khi confirm server mới chạy ổn 24-48h, tắt server cũ:

```bash
# Server cũ
docker compose down
sudo systemctl stop vllm-qwen
sudo systemctl disable vllm-qwen
```

---

## 7. Future: tự động hóa

- **bootstrap.sh:** 1 script chạy hết 2.x + 3.x (prereq + clone + build). Phù hợp khi setup nhiều server.
- **GitHub Actions:** build Docker image, push lên GHCR. Server mới chỉ `docker compose pull && up -d`, không cần build local.
- **Ansible / Terraform:** infrastructure-as-code cho nhiều môi trường (staging / prod).
- **Cloudflare Tunnel:** thay ngrok, URL cố định, không lộ public IP (item #2 trong roadmap).

---

## 8. Quick checklist (in & dán bên cạnh)

- [ ] Driver NVIDIA ≥ 525, `nvidia-smi` OK
- [ ] Docker + NVIDIA Container Toolkit OK
- [ ] Python + vllm package OK
- [ ] `git clone` repo branch `sub_main`
- [ ] `bp_detector/best.pt` scp xong (~22MB)
- [ ] `OCR_server/.env` + `.env` đã điền (generate key mới)
- [ ] `systemd vllm-qwen.service` chạy, Qwen model loaded
- [ ] `docker compose build` không lỗi
- [ ] `docker compose up -d` → 3 container healthy
- [ ] Health endpoint trả `vlm_circuit` field
- [ ] Smoke test YOLO (bb.jpg) → mach=70, BP=118/78
- [ ] Smoke test VLM (handwritten/table) → có chỉ số non-null
- [ ] ngrok public URL hoạt động
- [ ] PHR client đổi sang URL + key mới
- [ ] Server cũ giữ chạy 24-48h trước khi tắt
