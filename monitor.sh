#!/bin/bash
# OCR stack health monitor — alert Telegram if any component fails.
# Checks: OCR API (8502), vLLM (8080), ngrok tunnel (4040 admin API).
# Notifies only on state change (down→up or up→down) to avoid spam.
#
# Setup:
#   1. chmod +x monitor.sh
#   2. Edit BOT_TOKEN + CHAT_ID below
#   3. crontab -e:
#        * * * * * /home/phungkien/OCR_PHR/monitor.sh

# ─── Config ───────────────────────────────────────────────────────────────
BOT_TOKEN="REPLACE_WITH_BOT_TOKEN"
CHAT_ID="REPLACE_WITH_CHAT_ID"

OCR_URL="http://localhost:8502/health"
VLM_URL="http://localhost:8080/v1/models"
NGROK_URL="http://localhost:4040/api/tunnels"
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

check() {
    if curl -s -f -m 5 "$1" >/dev/null 2>&1; then echo "up"; else echo "down"; fi
}

ocr_status=$(check "$OCR_URL")
vlm_status=$(check "$VLM_URL")
ngrok_status=$(check "$NGROK_URL")

# Extract ngrok public URL if up (for visibility)
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

# Trail log (last 1000 lines)
{ echo "$(date '+%F %T') ${current}"; tail -1000 /var/log/ocr_monitor.log 2>/dev/null; } > /tmp/.mon.tmp && mv /tmp/.mon.tmp /var/log/ocr_monitor.log 2>/dev/null
