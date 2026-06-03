# ocr-server — FastAPI + YOLO (CUDA) + httpx client to vLLM
#
# Base: pytorch CUDA runtime (cuDNN 9, CUDA 12.1). Đã có Python 3.11 + torch GPU,
# nên ultralytics chỉ cần pip thêm (không kéo torch lần 2). Image ~6GB.
#
# Build:
#   docker build -t ocr-server:latest .
# Run (single-container, host network):
#   docker run --gpus all --rm -p 8502:8502 --env-file OCR_server/.env \
#       -v $(pwd)/OCR_server/data/ocr_storage:/app/OCR_server/data/ocr_storage \
#       -v $(pwd)/bp_detector/best.pt:/app/bp_detector/best.pt:ro \
#       ocr-server:latest

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2

# System deps for opencv-python-headless (libGL, libglib2.0) + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first — leverage layer cache when only code changes
COPY OCR_server/requirements.txt /tmp/req-server.txt
COPY bp_detector/requirements.txt /tmp/req-bp.txt

# Headless opencv (no GUI libs needed in container) + slowapi for rate limit +
# python-multipart for FastAPI UploadFile. opencv-python-headless replaces
# opencv-python listed in requirements (cv2 API identical).
RUN sed -i 's/^opencv-python\b/opencv-python-headless/' /tmp/req-server.txt /tmp/req-bp.txt \
    && pip install --no-cache-dir \
        -r /tmp/req-server.txt \
        -r /tmp/req-bp.txt \
        slowapi>=0.1.9 \
        python-multipart>=0.0.9

# Copy code (best.pt mount as volume in compose to keep image lean)
COPY OCR_server /app/OCR_server
COPY bp_detector /app/bp_detector

# Drop bundled best.pt if it sneaked in via COPY — mounted at runtime
RUN rm -f /app/bp_detector/best.pt

# Non-root user. UID/GID matchable với host owner của bind-mounted storage dir
# qua build-arg để tránh permission denied khi write metadata JSON.
ARG OCR_UID=1000
ARG OCR_GID=1000
RUN groupadd -r -g ${OCR_GID} ocr \
    && useradd -r -u ${OCR_UID} -g ocr -d /app -s /sbin/nologin ocr \
    && mkdir -p /app/OCR_server/data/ocr_storage \
    && chown -R ocr:ocr /app
USER ocr

WORKDIR /app/OCR_server

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8502/health || exit 1

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8502", "--workers", "1"]
