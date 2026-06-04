"""Tests cho obs.Timings + JsonFormatter."""
from __future__ import annotations

import json
import logging
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest

from ocr_vitals.obs import JsonFormatter, Timings, get_timings, set_timings, time_phase


# ─── Timings ──────────────────────────────────────────────────────────────

def test_phase_records_ms():
    t = Timings()
    with t.phase("yolo"):
        time.sleep(0.02)
    d = t.to_dict()
    assert "yolo" in d
    assert 15 <= d["yolo"] <= 200   # generous bounds cho slow CI


def test_same_phase_name_accumulates():
    t = Timings()
    with t.phase("vlm"):
        time.sleep(0.01)
    with t.phase("vlm"):
        time.sleep(0.01)
    d = t.to_dict()
    assert d["vlm"] >= 15           # cộng dồn ≥ 15ms (2 lần 10ms)


def test_nested_phases_both_recorded():
    t = Timings()
    with t.phase("outer"):
        with t.phase("inner"):
            time.sleep(0.01)
        time.sleep(0.01)
    d = t.to_dict()
    assert "outer" in d
    assert "inner" in d
    assert d["outer"] >= d["inner"] # outer wrap inner nên ≥


def test_add_direct():
    t = Timings()
    t.add("manual", 0.123)
    assert t.to_dict()["manual"] == 123


def test_get_set_timings_contextvar():
    assert get_timings() is None
    t = Timings()
    set_timings(t)
    assert get_timings() is t


def test_time_phase_helper_noop_without_context():
    # get_timings() trả None khi ngoài request context
    set_timings(None)            # explicit reset
    # Reset contextvar — set_timings(None) hợp lệ
    with time_phase("foo"):
        pass                     # KHÔNG crash dù không có Timings


def test_time_phase_helper_records_when_in_context():
    t = Timings()
    set_timings(t)
    with time_phase("recorded"):
        time.sleep(0.01)
    assert "recorded" in t.to_dict()


# ─── JsonFormatter ────────────────────────────────────────────────────────

def test_json_format_basic_record():
    rec = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello %s", args=("world",), exc_info=None,
    )
    out = JsonFormatter().format(rec)
    obj = json.loads(out)
    assert obj["msg"] == "hello world"
    assert obj["level"] == "INFO"
    assert obj["logger"] == "test"
    assert "ts" in obj


def test_json_format_includes_extra():
    rec = logging.LogRecord(
        name="x", level=logging.INFO, pathname="", lineno=0,
        msg="evt", args=(), exc_info=None,
    )
    rec.request_id = "abc-123"
    rec.fields_count = 5
    out = JsonFormatter().format(rec)
    obj = json.loads(out)
    assert obj["request_id"] == "abc-123"
    assert obj["fields_count"] == 5


def test_json_format_handles_non_serializable():
    """default=str fallback — không crash với object lạ."""
    rec = logging.LogRecord(
        name="x", level=logging.INFO, pathname="", lineno=0,
        msg="evt", args=(), exc_info=None,
    )
    rec.weird = object()
    out = JsonFormatter().format(rec)
    obj = json.loads(out)
    assert "weird" in obj
    assert "object" in obj["weird"]


def test_json_format_with_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.LogRecord(
            name="x", level=logging.ERROR, pathname="", lineno=0,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    out = JsonFormatter().format(rec)
    obj = json.loads(out)
    assert "exc" in obj
    assert "ValueError" in obj["exc"]
