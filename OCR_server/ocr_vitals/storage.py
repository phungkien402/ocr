"""Image + metadata storage for future fine-tune dataset.

Layout:
    $STORAGE_PATH/
        2026/06/02/
            8f3a1e2b-....jpg     # original image
            8f3a1e2b-....json    # metadata (extracted vitals, raw VLM output, etc.)

Set STORAGE_PATH env to root dir. If unset → storage disabled (logs warning, no-op).
This keeps the service runnable for dev/test without disk persistence.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STORAGE_ROOT: Optional[Path] = None


def is_enabled() -> bool:
    """True iff STORAGE_PATH is set AND directory is writable."""
    return _root() is not None


def _root() -> Optional[Path]:
    """Lazy-resolve STORAGE_PATH env. Returns None if storage disabled."""
    global _STORAGE_ROOT
    if _STORAGE_ROOT is not None:
        return _STORAGE_ROOT if _STORAGE_ROOT != Path("/dev/null") else None
    raw = os.environ.get("STORAGE_PATH", "").strip()
    if not raw:
        logger.warning("STORAGE_PATH unset — image/metadata storage disabled")
        _STORAGE_ROOT = Path("/dev/null")
        return None
    p = Path(raw).expanduser().resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
        _STORAGE_ROOT = p
        logger.info("Storage enabled at %s", p)
        return p
    except OSError as e:
        logger.error("Cannot create STORAGE_PATH=%s: %s — storage disabled", p, e)
        _STORAGE_ROOT = Path("/dev/null")
        return None


def new_request_id() -> str:
    """Generate a fresh UUID4 request_id."""
    return str(_uuid.uuid4())


def _date_dir(root: Path, ts: datetime) -> Path:
    sub = root / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def save_image(request_id: str, image_bytes: bytes, suffix: str = ".jpg") -> Optional[str]:
    """Save raw image bytes. Returns relative path (YYYY/MM/DD/uuid.jpg) or None if disabled.

    Storage failures log a warning but never raise — extraction must not be blocked
    by disk issues.
    """
    root = _root()
    if root is None:
        return None
    ts = datetime.now()
    try:
        dir_ = _date_dir(root, ts)
        fname = f"{request_id}{suffix}"
        (dir_ / fname).write_bytes(image_bytes)
        return f"{ts:%Y/%m/%d}/{fname}"
    except OSError as e:
        logger.warning("save_image failed for %s: %s", request_id, e)
        return None


def save_metadata(request_id: str, metadata: dict) -> None:
    """Save metadata JSON next to the image. metadata must contain timestamp."""
    root = _root()
    if root is None:
        return
    ts_str = metadata.get("timestamp") or datetime.now().isoformat()
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        ts = datetime.now()
    try:
        dir_ = _date_dir(root, ts)
        (dir_ / f"{request_id}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("save_metadata failed for %s: %s", request_id, e)


def image_sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()
