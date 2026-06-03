#!/bin/bash
# OCR server health monitor — alert Telegram if /health fails or vLLM down.
# Run via cron every minute. Only alerts on state change (down→up or up→down)
# to avoid spam.
#
# Setup:
#   1. chmod +x monitor.sh
#   2. Edit BOT_TOKEN + CHAT_ID below
#   3. crontab -e:
#        * * * * * /home/phungkien/OCR_PHR/monitor.sh

# ─── Config ───────────────────────────────────────────────────────────────
BOT_TOKEN="8967449057:AAHC5n3LzM-5H6aifESdpn7wFq9QSdoqegA"
CHAT_ID="5770498222"

OCR_URL="http://localhost:8502/health"
VLM_URL="http://localhost:8080/v1/models"
STATE_FILE="/tmp/ocr_monitor_state"
HOSTNAME=$(hostname -s)
# ──────────────────────────────────────────────────────────────────────────

notify() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${CHAT_ID}" \
        --data-urlencode "text=🩺 [${HOSTNAME}] ${msg}" \
        --data-urlencode "parse_mode=Markdown" >/dev/null
}

# Check OCR server
ocr_status="down"
if curl -s -f -m 5 "$OCR_URL" >/dev/null 2>&1; then
    ocr_status="up"
fi

# Check vLLM
vlm_status="down"
if curl -s -f -m 5 "$VLM_URL" >/dev/null 2>&1; then
    vlm_status="up"
fi

ngrok_status="down"
if curl -s -f -m 5 http://localhost:4040/api/tunnels >/dev/null 2>&1; then
    ngrok_status="up"
fi

# Update combined state
current="ocr=${ocr_status} vlm=${vlm_status} ngrok=${ngrok_status}"

last=$(cat "$STATE_FILE" 2>/dev/null || echo "init")

# Notify only on change
if [ "$current" != "$last" ]; then
    if [ "$last" = "init" ]; then
        notify "Monitor started. Current: ${current}"
    elif [ "$ocr_status" = "down" ] || [ "$vlm_status" = "down" ] || [ "$ngrok_status" = "down" ]; then
        notify "⚠️ *DOWN*: ${current}"
    else
        notify "✅ Recovered: ${current}"
    fi
    echo "$current" > "$STATE_FILE"
fi

# Log every check (for trail), keep last 1000 lines
{ echo "$(date '+%F %T') ${current}"; tail -1000 /var/log/ocr_monitor.log 2>/dev/null; } > /tmp/.mon.tmp && mv /tmp/.mon.tmp /var/log/ocr_monitor.log 2>/dev/null
