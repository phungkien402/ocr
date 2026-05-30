"""FastAPI server for OCR Vital Signs."""
import os, tempfile, time, pathlib
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="OCR Vital Signs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

_HERE = pathlib.Path(__file__).parent
_HTML_PATH = _HERE / "static" / "index.html"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/process")
async def process(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg","image/png","image/jpg"):
        return JSONResponse(status_code=400, content={"detail":"Only JPG and PNG images are supported."})
    suffix = ".png" if "png" in (file.content_type or "") else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    t0 = time.perf_counter()
    try:
        from ocr_vitals.main import process_image_async
        result = await process_image_async(tmp_path, device="cuda:0")
        result["processing_time_s"] = round(time.perf_counter() - t0, 2)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Processing error: {str(e)}"})
    finally:
        os.unlink(tmp_path)

@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML_PATH.read_text(encoding="utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
