"""Blood Pressure Monitor OCR — local web server using Roboflow hosted API.

Cách chạy:
    pip install inference-sdk fastapi uvicorn python-multipart
    python app.py

Mở trình duyệt: http://localhost:8503
"""

import base64
import io
import tempfile
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from predict import run_prediction

MODEL_PATH = str(Path(__file__).parent / "best.pt")

app = FastAPI(title="BP Monitor OCR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Post-processing ─────────────────────────────

def parse_bp_predictions(predictions: list, img_height: int) -> dict:
    """Convert Roboflow digit detections → vital signs JSON.

    Groups detections by Y position (SYS / DIA / PULSE row),
    sorts within each row by X, concatenates digits → number.
    """
    if not predictions:
        return {"error": "Không detect được chữ số nào"}

    # Sort all detections by Y (top to bottom)
    dets = sorted(predictions, key=lambda d: d["y"])

    # Group into rows: digits within 40px of each other vertically = same row
    rows = []
    current_row = [dets[0]]
    for d in dets[1:]:
        if abs(d["y"] - current_row[-1]["y"]) < 40:
            current_row.append(d)
        else:
            rows.append(current_row)
            current_row = [d]
    rows.append(current_row)

    # Filter: keep rows with at least 1 digit, confidence > 0.5
    rows = [
        sorted([d for d in row if d["confidence"] > 0.25], key=lambda d: d["x"])
        for row in rows
        if any(d["confidence"] > 0.25 for d in row)
    ]

    # Remove rows with fewer than 1 detection
    rows = [r for r in rows if len(r) >= 1]

    # Reconstruct number from digit classes
    def class_to_digit(cls):
        """Convert class label to single digit string.
        Class "10" = duplicate "0" label (7-segment slash-zero) → skip.
        """
        if str(cls) == '10':
            return None  # skip
        try:
            return str(int(cls))
        except ValueError:
            return None

    def row_to_number(row):
        parts = [class_to_digit(d["class"]) for d in row]
        digits = "".join(p for p in parts if p is not None)
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    # BP monitor layout: row 0 = SYS, row 1 = DIA, row 2 = PULSE
    # Filter out time display (22:08) by skipping rows where digits form >3 chars
    data_rows = [r for r in rows if len(r) <= 3]

    result = {
        "raw_rows": [[{"digit": d["class"], "x": round(d["x"]), "y": round(d["y"]), "conf": round(d["confidence"], 2)} for d in row] for row in data_rows],
        "huyet_ap": None,
        "mach": None,
        "all_detections": len(predictions),
        "rows_detected": len(data_rows),
    }

    if len(data_rows) >= 1:
        sys_val = row_to_number(data_rows[0])
        result["sys_raw"] = sys_val

    if len(data_rows) >= 2:
        dia_val = row_to_number(data_rows[1])
        result["dia_raw"] = dia_val

    if len(data_rows) >= 3:
        pulse_val = row_to_number(data_rows[2])
        result["mach"] = pulse_val

    # Build huyet_ap
    sys_v = result.get("sys_raw")
    dia_v = result.get("dia_raw")
    if sys_v or dia_v:
        result["huyet_ap"] = {"tam_thu": sys_v, "tam_truong": dia_v}

    # Clean up internal keys
    result.pop("sys_raw", None)
    result.pop("dia_raw", None)

    return result


# ─── API endpoints ───────────────────────────────

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Upload ảnh → trả về vital signs JSON."""
    try:
        contents = await file.read()

        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Dùng đúng logic từ predict.py (đã test chuẩn)
        result = run_prediction(tmp_path, MODEL_PATH)
        os.unlink(tmp_path)

        # Map sang format web UI
        vitals = {
            "huyet_ap": None,
            "mach": None,
            "all_detections": len(result.get("raw_rows", [])),
            "rows_detected": len(result.get("raw_rows", [])),
            "raw_rows": result.get("raw_rows", []),
        }
        v = result.get("vitals", {})
        sys_v = _to_int(v.get("systolic"))
        dia_v = _to_int(v.get("diastolic"))
        if sys_v or dia_v:
            vitals["huyet_ap"] = {"tam_thu": sys_v, "tam_truong": dia_v}
        vitals["mach"] = _to_int(v.get("pulse"))

        img_b64 = base64.b64encode(contents).decode()
        vitals["image_b64"] = f"data:image/jpeg;base64,{img_b64}"
        vitals["debug_steps"] = result.get("debug_steps", {})

        return JSONResponse(vitals)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _to_int(val):
    try:
        return int(str(val)) if val is not None else None
    except (ValueError, TypeError):
        return None


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# ─── Web UI ──────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BP Monitor OCR</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #f0f4f8; color: #1a1a1a; }

.header { background: #1e3a5f; color: white; padding: 16px 24px; }
.header h1 { font-size: 18px; font-weight: 700; }
.header p { font-size: 12px; color: #94a3b8; margin-top: 2px; }

.container { max-width: 960px; margin: 24px auto; padding: 0 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.steps-card { max-width: 960px; margin: 0 auto 24px; padding: 0 16px; }
.steps-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 12px; }
.step-item { text-align: center; }
.step-item img { width: 100%; border-radius: 6px; border: 1px solid #e2e8f0; cursor: pointer; transition: transform .15s; }
.step-item img:hover { transform: scale(1.05); }
.step-item .step-label { font-size: 11px; color: #64748b; margin-top: 4px; font-weight: 600; }
.step-item .step-desc { font-size: 10px; color: #94a3b8; }

.card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.card h2 { font-size: 14px; font-weight: 700; color: #555; margin-bottom: 14px; text-transform: uppercase; letter-spacing: .5px; }

/* Upload */
.drop-zone { border: 2px dashed #cbd5e1; border-radius: 10px; padding: 32px 16px; text-align: center; cursor: pointer; transition: all .2s; }
.drop-zone:hover, .drop-zone.drag { border-color: #3b82f6; background: #eff6ff; }
.drop-zone p { font-size: 13px; color: #64748b; margin-top: 8px; }
.drop-zone .icon { font-size: 36px; }
input[type=file] { display: none; }

.btn { width: 100%; margin-top: 14px; padding: 11px; background: #2563eb; color: white; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background .2s; }
.btn:hover { background: #1d4ed8; }
.btn:disabled { background: #93c5fd; cursor: not-allowed; }

/* Preview */
#preview { display: none; width: 100%; border-radius: 8px; margin-top: 12px; max-height: 280px; object-fit: contain; }

/* Results */
.vital { display: flex; align-items: center; justify-content: space-between; padding: 12px 14px; border-radius: 8px; margin-bottom: 10px; }
.vital .label { font-size: 13px; color: #555; }
.vital .label span { display: block; font-size: 11px; color: #94a3b8; margin-top: 2px; }
.vital .value { font-size: 28px; font-weight: 800; }
.vital.sys { background: #fef2f2; }
.vital.sys .value { color: #dc2626; }
.vital.dia { background: #fff7ed; }
.vital.dia .value { color: #ea580c; }
.vital.pulse { background: #f0fdf4; }
.vital.pulse .value { color: #16a34a; }
.vital.empty { background: #f8fafc; }
.vital.empty .value { color: #94a3b8; font-size: 22px; }

.badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.badge.ok { background: #dcfce7; color: #166534; }
.badge.warn { background: #fef3c7; color: #92400e; }

.debug { font-size: 11px; color: #888; background: #f8fafc; border-radius: 6px; padding: 10px; margin-top: 12px; overflow-x: auto; white-space: pre; }
.spinner { display: none; text-align: center; padding: 20px; color: #64748b; font-size: 13px; }
.err { color: #dc2626; font-size: 13px; padding: 10px; background: #fef2f2; border-radius: 8px; }
</style>
</head>
<body>

<div class="header">
  <h1>🩺 Blood Pressure Monitor OCR</h1>
  <p>Roboflow YOLOv8 · blood-pressure-monitor-display/1 · mAP 95.9%</p>
</div>

<div class="steps-card" id="stepsContainer" style="display:none"></div>
<div class="container">
  <!-- Upload card -->
  <div class="card">
    <h2>Ảnh đầu vào</h2>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">📷</div>
      <p>Click hoặc kéo thả ảnh máy đo huyết áp vào đây</p>
      <p style="font-size:11px;margin-top:4px;">JPG, PNG — máy Omron, A&D, Microlife...</p>
    </div>
    <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
    <img id="preview">
    <button class="btn" id="detectBtn" onclick="detect()" disabled>Detect chỉ số</button>
  </div>

  <!-- Results card -->
  <div class="card">
    <h2>Kết quả</h2>
    <div id="placeholder" style="text-align:center;padding:40px;color:#94a3b8;font-size:13px;">
      Chưa có ảnh — upload ảnh để bắt đầu
    </div>
    <div class="spinner" id="spinner">⏳ Đang phân tích...</div>
    <div id="results" style="display:none"></div>
  </div>
</div>

<script>
let selectedFile = null;

// Drag & drop
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('preview');
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
  document.getElementById('detectBtn').disabled = false;
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('results').style.display = 'none';
}

async function detect() {
  if (!selectedFile) return;
  document.getElementById('detectBtn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/detect', { method: 'POST', body: formData });
    const data = await resp.json();
    renderResults(data);
  } catch(e) {
    document.getElementById('results').innerHTML = `<div class="err">Lỗi: ${e.message}</div>`;
    document.getElementById('results').style.display = 'block';
  }

  document.getElementById('spinner').style.display = 'none';
  document.getElementById('detectBtn').disabled = false;
}

function renderResults(data) {
  const el = document.getElementById('results');
  if (data.error) {
    el.innerHTML = `<div class="err">❌ ${data.error}</div>`;
    el.style.display = 'block';
    return;
  }

  const bp = data.huyet_ap || {};
  const sys = bp.tam_thu;
  const dia = bp.tam_truong;
  const pulse = data.mach;

  const sysStatus = sys ? (sys >= 90 && sys <= 120 ? 'ok' : 'warn') : null;
  const diaStatus = dia ? (dia >= 60 && dia <= 80 ? 'ok' : 'warn') : null;
  const pulseStatus = pulse ? (pulse >= 60 && pulse <= 100 ? 'ok' : 'warn') : null;

  el.innerHTML = `
    <div class="vital sys">
      <div class="label">Huyết áp tâm thu (SYS)<span>mmHg · Bình thường: 90–120</span></div>
      <div>
        <div class="value">${sys ?? '—'}</div>
        ${sysStatus ? `<span class="badge ${sysStatus}">${sysStatus === 'ok' ? 'Bình thường' : 'Bất thường'}</span>` : ''}
      </div>
    </div>
    <div class="vital dia">
      <div class="label">Huyết áp tâm trương (DIA)<span>mmHg · Bình thường: 60–80</span></div>
      <div>
        <div class="value">${dia ?? '—'}</div>
        ${diaStatus ? `<span class="badge ${diaStatus}">${diaStatus === 'ok' ? 'Bình thường' : 'Bất thường'}</span>` : ''}
      </div>
    </div>
    <div class="vital pulse">
      <div class="label">Nhịp tim (PULSE)<span>lần/phút · Bình thường: 60–100</span></div>
      <div>
        <div class="value">${pulse ?? '—'}</div>
        ${pulseStatus ? `<span class="badge ${pulseStatus}">${pulseStatus === 'ok' ? 'Bình thường' : 'Bất thường'}</span>` : ''}
      </div>
    </div>
    <div style="font-size:11px;color:#94a3b8;margin-top:8px;">
      Detect được ${data.all_detections} digits · ${data.rows_detected} hàng · model YOLOv8
    </div>
    <details style="margin-top:10px;">
      <summary style="font-size:12px;color:#64748b;cursor:pointer;">Debug — raw rows</summary>
      <div class="debug">${JSON.stringify(data.raw_rows, null, 2)}</div>
    </details>
  `;
  el.style.display = 'block';

  // Hiển thị preprocessing steps
  renderSteps(data.debug_steps || {});
}

const STEP_LABELS = {
  '1_original':   ['Gốc',         'Ảnh đầu vào'],
  '2_perspective':['Perspective', 'Chỉnh nghiêng'],
  '3_lcd_crop':   ['LCD Crop',    'Cắt vùng LCD'],
  '4_enhanced':   ['Enhanced',    'Bilateral+Gamma'],
  '5_final':      ['Final',       'Upscale → YOLO'],
};

function renderSteps(steps) {
  const container = document.getElementById('stepsContainer');
  const keys = Object.keys(steps).sort();
  if (!keys.length) { container.style.display='none'; return; }

  const items = keys.map(k => {
    const [label, desc] = STEP_LABELS[k] || [k, ''];
    return `
      <div class="step-item">
        <img src="${steps[k]}" title="${label}" onclick="window.open(this.src)"/>
        <div class="step-label">${label}</div>
        <div class="step-desc">${desc}</div>
      </div>`;
  }).join('');

  container.innerHTML = `
    <div class="card">
      <h2>Preprocessing Pipeline</h2>
      <div class="steps-grid">${items}</div>
    </div>`;
  container.style.display = 'block';
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8503)
