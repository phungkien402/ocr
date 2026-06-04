"""Observability helpers — phase timing + structured (JSON) logging.

Why:
- Pilot cần log chi tiết để debug + đo per-phase latency.
- Format kép: plain text → stdout (xem `docker logs`), JSON line → file (parse với jq, ship lên log aggregator sau).
- PHI-safe: KHÔNG log raw vitals values. Chỉ field NAMES detected.

Architecture:
    Per-request:
      1. web_app handler: Timings() + set_timings(t)
      2. Pipeline gọi t.phase("yolo") quanh từng đoạn nặng
      3. handler emit 1 log line tổng kết với t.to_dict()

Contextvar đảm bảo Timings không nhầm giữa các request đồng thời (asyncio).
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional

# ─── Timings (phase-level latency capture) ──────────────────────────────────

_current_timings: ContextVar[Optional["Timings"]] = ContextVar("timings", default=None)


class Timings:
    """Record per-phase duration trong 1 request.

    Usage:
        t = Timings()
        set_timings(t)
        with t.phase("yolo"):
            run_yolo()
        with t.phase("vlm"):
            run_vlm()
        log_dict = t.to_dict()   # {"yolo": 380, "vlm": 2700}  (ms)

    Nested phase OK — vd "vlm" wrap cả "vlm_http" + "vlm_parse".
    Phase trùng tên → cộng dồn (vd gọi vlm retry nhiều lần).
    """

    def __init__(self) -> None:
        self._phases: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def phase(self, name: str) -> "_PhaseTimer":
        return _PhaseTimer(self, name)

    def add(self, name: str, seconds: float) -> None:
        self._phases[name] = self._phases.get(name, 0.0) + seconds

    def total_s(self) -> float:
        return time.perf_counter() - self._t0

    def to_dict(self) -> dict[str, int]:
        """Round to ms cho log gọn."""
        return {k: round(v * 1000) for k, v in self._phases.items()}


class _PhaseTimer:
    __slots__ = ("_t", "_name", "_start")

    def __init__(self, t: Timings, name: str) -> None:
        self._t = t
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._t.add(self._name, time.perf_counter() - self._start)


def get_timings() -> Optional[Timings]:
    """Trả Timings của request hiện tại nếu có. None nếu gọi ngoài request context."""
    return _current_timings.get()


def set_timings(t: Timings) -> None:
    _current_timings.set(t)


def time_phase(name: str):
    """Decorator + context manager — no-op nếu không có Timings context."""
    t = get_timings()
    if t is None:
        return _Noop()
    return t.phase(name)


class _Noop:
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ─── JSON log formatter ────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """Format LogRecord thành 1 dòng JSON. Field 'extra' dict được merge vào root."""

    # Field LogRecord chuẩn không cần log — chỉ giữ msg, level, time, name
    _SKIP = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":     datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        # Merge extra fields (logger.info("...", extra={...}))
        for k, v in record.__dict__.items():
            if k not in self._SKIP and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# ─── Setup ──────────────────────────────────────────────────────────────────

_LOG_FILE     = os.environ.get("LOG_FILE", "").strip()
_LOG_FILE_MAX = int(os.environ.get("LOG_FILE_MAX_BYTES", str(50 * 1024 * 1024)))   # 50MB
_LOG_FILE_BAK = int(os.environ.get("LOG_FILE_BACKUP_COUNT", "5"))


def configure_logging(level: str = "INFO") -> None:
    """Setup root logger với 2 handler:

    - stdout (plain text, human-readable, capture bởi `docker logs`)
    - LOG_FILE rotating (JSON line) nếu env set

    Idempotent — gọi nhiều lần OK.
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers (tránh duplicate khi reload)
    root.handlers.clear()

    # 1. stdout — plain text
    text_h = logging.StreamHandler()
    text_h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(text_h)

    # 2. file — JSON line (chỉ bật nếu LOG_FILE set)
    if _LOG_FILE:
        try:
            os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
            file_h = logging.handlers.RotatingFileHandler(
                _LOG_FILE, maxBytes=_LOG_FILE_MAX, backupCount=_LOG_FILE_BAK,
                encoding="utf-8",
            )
            file_h.setFormatter(JsonFormatter())
            root.addHandler(file_h)
            logging.getLogger(__name__).info(
                "JSON log → %s (max=%dMB, backup=%d)",
                _LOG_FILE, _LOG_FILE_MAX // 1024 // 1024, _LOG_FILE_BAK,
            )
        except OSError as e:
            logging.getLogger(__name__).warning("Cannot open LOG_FILE %s: %s", _LOG_FILE, e)
