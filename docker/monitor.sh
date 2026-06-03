#!/bin/bash
# OCR stack health monitor (container version) — alert Telegram on state change.
#
# Targets resolved via Docker compose DNS:
#   - ocr-server:8502  (compose service name)
#   - host.docker.internal:8080  (vLLM on host, systemd)
#   - ngrok:4040  (compose service)
#
# Env vars (set in docker-compose):
#   BOT_TOKEN, CHAT_ID, HOST_LABEL (default = hostname)

BOT_TOKEN="${BOT_TOKEN:-}"
CHAT_ID="${CHAT_ID:-}"
HOST_LABEL="${HOST_LABEL:-$(hostname -s)}"

OCR_URL="http://${OCR_HOST:-ocr-server}:8502/health"
VLM_URL="http://${VLM_HOST:-host.docker.internal}:8080/v1/models"
NGROK_URL="http://${NGROK_HOST:-ngrok}:4040/api/tunnels"

STATE_FILE="/var/lib/monitor/state"
LOG_FILE="/var/log/ocr_monitor.log"

notify() {
    [ -z "$BOT_TOKEN" ] && { echo "[notify] no BOT_TOKEN, skip"; return; }
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${CHAT_ID}" \
        --data-urlencode "text=🩺 [${HOST_LABEL}] ${msg}" \
        --data-urlencode "parse_mode=Markdown" >/dev/null
}

check() {
    if curl -s -f -m 5 "$1" >/dev/null 2>&1; then echo "up"; else echo "down"; fi
}

ocr_status=$(check "$OCR_URL")
vlm_status=$(check "$VLM_URL")
ngrok_status=$(check "$NGROK_URL")

ngrok_url=""
if [ "$ngrok_status" = "up" ]; then
    ngrok_url=$(curl -s -m 5 "$NGROK_URL" 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'] if d.get('tunnels') else '')" 2>/dev/null)
fi

current="ocr=${ocr_status} vlm=${vlm_status} ngrok=${ngrok_status}"
last=$(cat "$STATE_FILE" 2>/dev/null || echo "init")

if [ "$current" != "$last" ]; then
    if [ "$last" = "init" ]; then
        msg="Monitor started. ${current}"
        [ -n "$ngrok_url" ] && msg="${msg}"$'\n'"URL: ${ngrok_url}"
        notify "$msg"
    elif [[ "$current" == *"down"* ]]; then
        notify "⚠️ *DOWN*: ${current}"
    else
        msg="✅ Recovered: ${current}"
        [ -n "$ngrok_url" ] && msg="${msg}"$'\n'"URL: ${ngrok_url}"
        notify "$msg"
    fi
    echo "$current" > "$STATE_FILE"
fi

# Append + truncate log to last 1000 lines
{ echo "$(date '+%F %T') ${current}"; tail -1000 "$LOG_FILE" 2>/dev/null; } \
    > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
