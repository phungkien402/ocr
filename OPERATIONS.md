# OCR HealthCare — Sổ tay vận hành

> Hướng dẫn day-to-day cho người quản trị server + dịch vụ. KHÔNG phải tài liệu deploy lần đầu (xem `DEPLOY_DOCKER.md`) hay migrate (xem `MIGRATE.md`).

---

## 1. Kiến trúc 5 giây

```
PHR mobile → ngrok HTTPS → ocr-server :8502 ──► YOLO (best.pt) → fallback ──► vLLM :8080
                                │                                                 (systemd, GPU 0)
                                ├── auth + rate limit + retry + circuit breaker
                                ├── storage YYYY/MM/DD/{uuid}.{jpg,json}
                                └── logs PHI-safe

ocr-monitor → check 3 service mỗi 60s → Telegram alert khi đổi state
```

**Service đang chạy:**

| Service | Loại | Port | Mục đích |
|---|---|---|---|
| `vllm-qwen.service` | systemd | 8080 | VLM serving Qwen2.5-VL-3B |
| `ocr-server` | docker container | 8502 | FastAPI + YOLO |
| `ocr-ngrok` | docker container (host network) | 4040 | HTTPS tunnel |
| `ocr-monitor` | docker container | — | Health check + Telegram |

**File quan trọng:**
- `/home/phungkien/OCR_PHR/` — repo + compose
- `/home/phungkien/OCR_PHR/.env` — NGROK + Telegram secrets
- `/home/phungkien/OCR_PHR/OCR_server/.env` — OCR_API_KEY + VLM config
- `/home/phungkien/OCR_PHR/OCR_server/data/ocr_storage/` — ảnh + metadata
- `/etc/systemd/system/vllm-qwen.service` — vLLM unit
- `/home/phungkien/.cache/huggingface/` — Qwen model cache (~6GB)

---

## 2. Routine checks

### Mỗi ngày (5 phút)

```bash
# 1. Tất cả service up không
docker compose ps
sudo systemctl is-active vllm-qwen

# 2. Health endpoint OK
curl -s http://localhost:8502/health | python3 -m json.tool
#    Verify: storage_enabled=true, auth_enabled=true, vlm_circuit.state="closed"

# 3. ngrok URL còn không
curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url'

# 4. Disk còn trống bao nhiêu
df -h /home

# 5. GPU usage
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

### Mỗi tuần (15 phút)

```bash
# 1. Backup storage (xem section 6)
# 2. Cleanup ảnh cũ > 90 ngày (xem section 8)
# 3. Check log size + rotate
sudo du -sh /var/log/journal/
sudo journalctl --vacuum-time=14d   # giữ log 14 ngày

# 4. Update OS security patches
sudo apt-get update && sudo apt-get upgrade -y --only-upgrade

# 5. Check ngrok session quota (free tier limit)
#    Login dashboard.ngrok.com → Usage tab
```

### Mỗi tháng (30 phút)

```bash
# 1. Restart toàn bộ stack (clear memory leak nếu có)
sudo systemctl restart vllm-qwen
sleep 60
docker compose restart

# 2. Test smoke end-to-end
KEY=$(grep OCR_API_KEY OCR_server/.env | cut -d= -f2)
curl -s -X POST -H "X-API-Key: $KEY" -F "file=@test_images/bb.jpg" \
  http://localhost:8502/v1/extract | python3 -m json.tool

# 3. Check storage growth rate — dự đoán khi nào đầy disk
du -sh OCR_server/data/ocr_storage/$(date +%Y/%m)/

# 4. Review Telegram alert log — có pattern lặp lại không
```

---

## 3. Operations cheatsheet

### Restart service

```bash
# Restart 1 container
docker compose restart ocr-server
docker compose restart ngrok
docker compose restart monitor

# Restart toàn bộ
docker compose down && docker compose up -d

# Restart vLLM (systemd)
sudo systemctl restart vllm-qwen
sleep 60                                    # đợi load model
curl -s http://localhost:8080/v1/models     # verify
```

### Deploy code mới

```bash
cd /home/phungkien/OCR_PHR

# 1. Pull code
git pull origin sub_main

# 2. Build lại image
OCR_UID=$(id -u) OCR_GID=$(id -g) docker compose build ocr-server

# 3. Rolling restart (không downtime nếu đúng cách — đợi healthcheck mới đổi)
docker compose up -d ocr-server

# 4. Verify
sleep 10
curl -s http://localhost:8502/health | python3 -m json.tool
docker compose logs --tail=20 ocr-server
```

### Update Docker compose config

```bash
nano docker-compose.yml
# Sau khi sửa:
docker compose config              # verify YAML valid
docker compose up -d --remove-orphans
```

### Update systemd vLLM args

```bash
sudo systemctl edit --full vllm-qwen
# Sửa args (vd --max-model-len, --gpu-memory-utilization)
sudo systemctl daemon-reload
sudo systemctl restart vllm-qwen
sudo journalctl -u vllm-qwen -f    # theo dõi log đến khi "Application startup complete"
```

### Vào shell trong container

```bash
docker compose exec ocr-server bash
docker compose exec monitor sh
docker compose exec ngrok sh
```

### Chạy lệnh trong container

```bash
# Run pytest (nếu image có pytest)
docker compose exec ocr-server python -m pytest tests/ -v

# Inspect env trong container
docker compose exec ocr-server printenv | grep -E "VLM_|STORAGE|OCR_"

# Reset circuit breaker (chỉ cần restart container)
docker compose restart ocr-server
```

---

## 4. Logs — tìm ở đâu

| Log | Lệnh |
|---|---|
| ocr-server (FastAPI app) | `docker compose logs -f ocr-server` |
| ngrok | `docker compose logs -f ngrok` |
| monitor | `docker compose logs -f monitor` hoặc `docker compose exec monitor cat /var/log/ocr_monitor.log` |
| vLLM | `sudo journalctl -u vllm-qwen -f` |
| Telegram alert history | Telegram bot chat |
| System | `sudo journalctl -f` |

### Tìm 1 request cụ thể

PHR báo lỗi với `request_id`:

```bash
REQ_ID="787ed55d-39cc-42c6-b841-41837130f95d"

# 1. Tìm log line
docker compose logs ocr-server | grep $REQ_ID

# 2. Tìm metadata + image trên disk
find OCR_server/data/ocr_storage/ -name "$REQ_ID*"

# 3. Inspect metadata
cat OCR_server/data/ocr_storage/2026/06/03/$REQ_ID.json | python3 -m json.tool

# 4. Xem ảnh gốc
display OCR_server/data/ocr_storage/2026/06/03/$REQ_ID.jpg  # hoặc copy về local
```

### Filter log theo loại

```bash
# Chỉ extract requests
docker compose logs ocr-server | grep '\[extract\]'

# Chỉ VLM errors
docker compose logs ocr-server | grep -E 'VLM.*error|ConnectError|CircuitOpen'

# Chỉ 5xx
docker compose logs ocr-server | grep '500\|503'
```

---

## 5. Monitoring + alerts

### Telegram bot

Bot gửi alert khi state đổi (up→down hoặc down→up). Không spam — chỉ khi transition.

**Alert thường gặp:**

| Message | Ý nghĩa | Action |
|---|---|---|
| `⚠️ DOWN: ocr=down vlm=up ngrok=up` | ocr-server container crash | `docker compose logs ocr-server` rồi `docker compose up -d ocr-server` |
| `⚠️ DOWN: ocr=up vlm=down ngrok=up` | vLLM systemd crash | `sudo systemctl status vllm-qwen` + `sudo systemctl restart vllm-qwen` |
| `⚠️ DOWN: ocr=up vlm=up ngrok=down` | ngrok mất tunnel | Check authtoken còn valid không + `docker compose restart ngrok` |
| `✅ Recovered: ... URL: https://...` | All up — URL ngrok mới (báo PHR team) |

### Disable alert tạm thời (lúc maintenance)

```bash
docker compose stop monitor
# Sau khi xong:
docker compose start monitor
```

### Thêm người nhận alert

Telegram bot gửi tới 1 `CHAT_ID` duy nhất. Để nhiều người nhận:
1. Tạo Telegram group, add bot vào
2. Lấy chat_id của group (negative number, vd `-1001234567890`)
3. Update `TG_CHAT_ID` trong `.env`
4. `docker compose restart monitor`

---

## 6. Backup + restore

### Backup storage (ảnh + metadata)

```bash
# Daily backup script — đặt vào crontab
BACKUP_DIR=/backup/ocr
DATE=$(date +%Y-%m-%d)
mkdir -p $BACKUP_DIR

# Incremental rsync (chỉ copy file mới)
rsync -az --delete \
  /home/phungkien/OCR_PHR/OCR_server/data/ocr_storage/ \
  $BACKUP_DIR/ocr_storage/

# Backup config (lần đầu + khi thay đổi)
tar czf $BACKUP_DIR/config-$DATE.tgz \
  /home/phungkien/OCR_PHR/.env \
  /home/phungkien/OCR_PHR/OCR_server/.env \
  /home/phungkien/OCR_PHR/docker-compose.yml \
  /etc/systemd/system/vllm-qwen.service

# Giữ 30 backup gần nhất
find $BACKUP_DIR -name "config-*.tgz" -mtime +30 -delete
```

Crontab:
```cron
0 2 * * * /home/phungkien/scripts/backup.sh >> /var/log/backup.log 2>&1
```

### Off-site backup (recommend cho production)

```bash
# Push lên S3 / R2 / Backblaze hằng ngày
aws s3 sync $BACKUP_DIR/ s3://my-backup-bucket/ocr/ --delete
# Hoặc rclone sync $BACKUP_DIR/ remote:ocr-backup/
```

### Restore

```bash
# Restore storage
rsync -az /backup/ocr/ocr_storage/ \
  /home/phungkien/OCR_PHR/OCR_server/data/ocr_storage/
sudo chown -R 1000:1000 /home/phungkien/OCR_PHR/OCR_server/data/ocr_storage/

# Restore config
cd /tmp && tar xzf /backup/ocr/config-2026-06-03.tgz
# Copy file vào đúng vị trí, rồi restart
```

---

## 7. Disaster recovery

### Server chết hoàn toàn

Theo `MIGRATE.md` — tổng setup ~1.5-2h trên server mới + restore từ backup.

### vLLM crash loop

```bash
# 1. Check journal
sudo journalctl -u vllm-qwen -n 100 --no-pager

# Common causes:
#   - Out of VRAM: giảm --gpu-memory-utilization hoặc --max-model-len
#   - GPU driver fail: nvidia-smi check, có thể cần reboot
#   - HF cache corrupt: rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B*

# 2. Workaround tạm: tắt vLLM, OCR vẫn chạy YOLO fast path
sudo systemctl stop vllm-qwen
# CB sẽ tự open sau 5 fail, request không phải BP monitor trả null
# PHR app phải fallback gõ tay

# 3. Fix
sudo systemctl restart vllm-qwen
```

### Container ocr-server crash loop

```bash
# 1. Check log
docker compose logs --tail=100 ocr-server

# 2. Rebuild nếu cần
OCR_UID=$(id -u) OCR_GID=$(id -g) docker compose build --no-cache ocr-server
docker compose up -d --force-recreate ocr-server

# 3. Worst case: rollback về image cũ
docker images | grep ocr-server
docker tag ocr-server:<old-sha> ocr-server:latest
docker compose up -d --force-recreate ocr-server
```

### Disk full

```bash
# 1. Cleanup ngay
docker system prune -af --volumes        # XÓA image/container không dùng
docker compose logs --tail=0             # truncate log container
sudo journalctl --vacuum-size=500M       # truncate journal

# 2. Cleanup ảnh cũ
find OCR_server/data/ocr_storage/ -name "*.jpg" -mtime +90 -delete
find OCR_server/data/ocr_storage/ -name "*.json" -mtime +90 -delete

# 3. Nếu vẫn đầy: move storage sang disk khác
mv OCR_server/data/ocr_storage /mnt/bigdisk/ocr_storage
ln -s /mnt/bigdisk/ocr_storage OCR_server/data/ocr_storage
docker compose restart ocr-server
```

### ngrok session bị revoke

Triệu chứng: `ERR_NGROK_107` trong log. Xảy ra khi authtoken lộ public.

```bash
# 1. Vào https://dashboard.ngrok.com/get-started/your-authtoken → revoke + new
# 2. Update .env
nano .env
# 3. Restart
docker compose restart ngrok
# 4. URL mới — báo PHR team
```

---

## 8. Storage management

### Theo dõi growth rate

```bash
# Tổng size
du -sh OCR_server/data/ocr_storage/

# Per tháng
du -sh OCR_server/data/ocr_storage/2026/*/

# Số request / ngày
ls OCR_server/data/ocr_storage/2026/06/03/*.jpg | wc -l
```

### Retention policy (recommend)

Theo PROJECT_STATE limit #3, hiện chưa có cron auto cleanup. Implement:

```bash
sudo tee /etc/cron.daily/ocr-storage-cleanup > /dev/null <<'EOF'
#!/bin/bash
# Xóa ảnh + metadata > 90 ngày
STORAGE=/home/phungkien/OCR_PHR/OCR_server/data/ocr_storage
find $STORAGE -type f \( -name "*.jpg" -o -name "*.json" \) -mtime +90 -delete

# Xóa thư mục rỗng
find $STORAGE -type d -empty -delete

# Log
echo "$(date '+%F %T') cleanup done, current size: $(du -sh $STORAGE | cut -f1)" >> /var/log/ocr-cleanup.log
EOF
sudo chmod +x /etc/cron.daily/ocr-storage-cleanup
```

**Lưu ý compliance:** trước khi enable retention, confirm với pháp chế bệnh viện về thời gian giữ data (luật VN thường ≥10 năm với hồ sơ chính, ảnh tạm có thể ngắn hơn).

---

## 9. Security operations

### Rotate OCR_API_KEY

```bash
# 1. Generate key mới
NEW=$(python3 -c "import secrets; print(secrets.token_hex(32))")
echo $NEW

# 2. Update file
sed -i "s/^OCR_API_KEY=.*/OCR_API_KEY=$NEW/" OCR_server/.env

# 3. Restart container đọc env mới
docker compose restart ocr-server

# 4. Báo PHR team key mới (qua kênh bảo mật, KHÔNG Slack public)
#    Old key vẫn dùng được tới khi container restart — coordinate timing với PHR
```

### Rotate NGROK_AUTHTOKEN

```bash
# 1. Vào dashboard.ngrok.com → revoke cũ → create new
# 2. Update .env
sed -i "s/^NGROK_AUTHTOKEN=.*/NGROK_AUTHTOKEN=<NEW>/" .env
docker compose restart ngrok
```

### Audit "ai dùng API"

Hiện log đã có request_id + IP per request:

```bash
# Số request hôm nay
docker compose logs ocr-server --since 24h | grep '\[extract\]' | wc -l

# Per IP
docker compose logs ocr-server --since 24h | grep 'POST /v1/extract' | \
  awk '{print $3}' | sort | uniq -c | sort -rn

# Failed auth (401)
docker compose logs ocr-server --since 24h | grep '401'
```

### Tăng cường security tương lai

- [ ] Multi API key (per BV) — phân biệt được ai gọi
- [ ] Rate limit per IP riêng (slowapi đã support)
- [ ] IP whitelist nếu PHR có range IP cố định
- [ ] Audit endpoint admin xem history requests

---

## 10. Troubleshooting

### Triệu chứng: PHR báo extract trả all null

```bash
# 1. Check log ocr-server
docker compose logs --tail=50 ocr-server | grep -E "extract|VLM|YOLO"

# Đoán nguyên nhân theo log:
#   "ConnectError" → vLLM down → systemctl restart vllm-qwen
#   "CircuitOpen" → đợi 60s hoặc restart ocr-server reset CB
#   "engine=yolo+vlm" + log VLM OK → ảnh khó, model không nhận ra, không phải lỗi server
```

### Triệu chứng: extract chậm bất thường

```bash
# 1. Check GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 2

# 2. Check vLLM queue (nếu p95 latency tăng = queue dài)
sudo journalctl -u vllm-qwen --since "1 hour ago" | grep -i "waiting\|queue"

# 3. Check số request đồng thời
docker compose logs --since 5m ocr-server | grep extract | wc -l

# Fix:
#   - Nếu queue dài → tăng rate limit hoặc bật tensor-parallel-size 2
#   - Nếu GPU idle → check vLLM bị stuck, restart vllm-qwen
```

### Triệu chứng: 429 rate limit

Hiện default 300/min per key. Nếu bị thường xuyên:
```bash
sed -i "s/^RATE_LIMIT=.*/RATE_LIMIT=600\/minute/" OCR_server/.env
docker compose restart ocr-server
```

### Triệu chứng: container không start

```bash
# 1. Verbose log
docker compose up ocr-server                    # foreground, không -d

# 2. Common issue:
#   - "permission denied" mount storage → chown 1000:1000
#   - "address already in use" → port 8502 bị chiếm: sudo ss -tlnp | grep 8502
#   - "could not select device driver" → NVIDIA toolkit broken: sudo nvidia-ctk runtime configure --runtime=docker
```

### Triệu chứng: ngrok bị "Too many connections"

ngrok free tier có limit ~1 simultaneous tunnel. Nếu để 2 instance song song:

```bash
# Verify chỉ 1 tunnel active
docker compose logs ngrok | tail -20
# Kill instance dư (vd systemd ngrok-ocr cũ)
sudo systemctl stop ngrok-ocr 2>/dev/null
```

---

## 11. Update + release workflow

```
DEV (local laptop)
   ├── Sửa code OCR_server/ocr_vitals/*.py
   ├── Chạy pytest local
   ├── git commit + push sub_main
   └── ──┐
         ▼
SERVER PROD
   ├── SSH vào server
   ├── cd OCR_PHR && git pull origin sub_main
   ├── docker compose build ocr-server          # ~30s với cache
   ├── docker compose up -d ocr-server          # rolling, ~10s downtime
   ├── Verify /health
   └── Smoke test extract
```

**Trước khi deploy bản lớn:**

```bash
# 1. Backup config + storage hiện tại
tar czf /tmp/pre-deploy-$(date +%F).tgz \
  OCR_server/.env .env docker-compose.yml
rsync -az OCR_server/data/ocr_storage/ /backup/pre-deploy/

# 2. Tag image cũ làm rollback point
docker tag ocr-server:latest ocr-server:stable-$(date +%F)

# 3. Deploy
docker compose build && docker compose up -d ocr-server

# 4. Verify 10 phút
for i in {1..30}; do
  curl -s http://localhost:8502/health | jq -r '.status'
  sleep 20
done

# 5. Nếu xấu: rollback
docker tag ocr-server:stable-2026-06-03 ocr-server:latest
docker compose up -d --force-recreate ocr-server
```

---

## 12. Handover notes (lúc nghỉ phép)

Người tiếp quản cần:

1. **Credentials**
   - SSH key vào `phungkien@ehcaihelpdeskserver`
   - Sudo password
   - Telegram bot admin access (để check alert)
   - ngrok account access
   - GitHub repo access

2. **Document đọc trước**
   - `README.md` — kiến trúc tổng
   - `PROJECT_STATE.md` — state chi tiết
   - `OPERATIONS.md` (file này) — vận hành
   - `DEPLOY_DOCKER.md` — setup lại nếu cần
   - `MIGRATE.md` — chuyển server

3. **Test sống**
   - SSH thử, chạy `docker compose ps`
   - `curl /health`
   - Smoke test extract qua URL ngrok
   - Verify nhận được Telegram alert (tự stop container test)

4. **Liên hệ khẩn cấp**
   - Nội bộ team OCR: <số điện thoại>
   - Team PHR mobile: <contact>
   - Sếp chính: <Trungnt>
   - NCC vLLM/Anthropic support nếu cần: (không có)

5. **Việc đang dang dở** (cập nhật khi handover):
   - Cloudflare Tunnel chưa setup
   - Benchmark accuracy chưa làm
   - HTTPS production chưa deploy

---

## 13. Quick reference — lệnh hay dùng nhất

```bash
# Status check
docker compose ps && sudo systemctl is-active vllm-qwen
curl -s http://localhost:8502/health | jq

# Restart 1 cái
docker compose restart ocr-server
sudo systemctl restart vllm-qwen

# Logs
docker compose logs -f ocr-server
sudo journalctl -u vllm-qwen -f

# Smoke test
KEY=$(grep OCR_API_KEY OCR_server/.env | cut -d= -f2)
curl -s -X POST -H "X-API-Key: $KEY" -F "file=@test_images/bb.jpg" \
  http://localhost:8502/v1/extract | jq

# ngrok URL
curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url'

# Disk
df -h /home
du -sh OCR_server/data/ocr_storage/

# GPU
nvidia-smi
```

---

## 14. Checklist 30s sau khi vào ca trực

- [ ] `docker compose ps` — cả 3 service Up + healthy
- [ ] `systemctl is-active vllm-qwen` — active
- [ ] `curl /health` — `status: ok`, `vlm_circuit.state: closed`
- [ ] ngrok URL ping được từ ngoài
- [ ] Disk free > 10GB
- [ ] Không có Telegram alert đỏ chưa giải quyết
- [ ] Smoke test 1 ảnh bb.jpg → trả `mach=70, BP=118/78`

Nếu mọi gạch đầu dòng ✅ → ca trực bình thường.
Nếu có dấu hỏi → vào section troubleshoot tương ứng.
