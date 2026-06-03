"""Tests for ocr_vitals.parser."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from ocr_vitals.parser import parse_vitals, _fuzzy_label


# ── _fuzzy_label ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,expected", [
    # Direct match
    ("mạch", "mach"),
    ("Mạch", "mach"),
    ("MẠCH", "mach"),
    ("pulse", "mach"),
    ("Heart Rate", "mach"),
    ("HR", "mach"),
    ("nhịp tim", "mach"),     # heart rate distinct from nhịp thở
    ("Nhịp tim", "mach"),
    # Respiratory
    ("nhịp thở", "nhip_tho"),
    ("Respiratory Rate", "nhip_tho"),
    ("RR", "nhip_tho"),
    # Temperature
    ("nhiệt độ", "nhiet_do"),
    ("nhiệtđộ", "nhiet_do"),  # no space (đ→d via NFKD fallback)
    ("Nhiệt Độ", "nhiet_do"),
    # Blood pressure
    ("huyết áp", "huyet_ap"),
    ("BP", "huyet_ap"),
    # Weight / height
    ("cân nặng", "can_nang"),
    ("cânnặng", "can_nang"),
    ("chiều cao", "chieu_cao"),
    ("Chỉu cao", "chieu_cao"),  # OCR mis-read
    # SpO2
    ("SpO2", "spo2"),
    ("Sp02", "spo2"),  # OCR confusion 'O' vs '0'
    # Markdown noise (stripped before lookup)
    ("**mạch**", "mach"),
    ("- mạch", "mach"),
    ("### mạch", "mach"),
])
def test_fuzzy_label_direct(label, expected):
    assert _fuzzy_label(label) == expected


@pytest.mark.parametrize("label,expected", [
    # VLM typos (difflib ratio fallback)
    ("Chỉnh cao", "chieu_cao"),
    ("Chiêu cao", "chieu_cao"),
    ("Chiều cau", "chieu_cao"),
    ("Nhip thơ", "nhip_tho"),
])
def test_fuzzy_label_typo_fallback(label, expected):
    assert _fuzzy_label(label) == expected


@pytest.mark.parametrize("label", [
    "xyz", "hello world", "abc123",
    "tableinfo", "random string here",
])
def test_fuzzy_label_no_match(label):
    assert _fuzzy_label(label) is None


# ── parse_vitals: JSON parser ──────────────────────────────────────────────

def test_parse_json_complete():
    raw = '{"mach": 75, "nhiet_do": 36.5, "huyet_ap": {"tam_thu": 120, "tam_truong": 80}, "nhip_tho": 16, "can_nang": 67, "chieu_cao": 170, "spo2": 98}'
    v = parse_vitals(raw)
    assert v["mach"] == 75
    assert v["nhiet_do"] == 36.5
    assert v["huyet_ap"] == {"tam_thu": 120, "tam_truong": 80}
    assert v["spo2"] == 98


def test_parse_json_bp_as_string():
    raw = '{"huyet_ap": "120/80", "mach": 75}'
    v = parse_vitals(raw)
    assert v["huyet_ap"] == {"tam_thu": 120, "tam_truong": 80}


def test_parse_json_with_null_values():
    raw = '{"mach": null, "huyet_ap": null, "spo2": 98}'
    v = parse_vitals(raw)
    assert v["mach"] is None
    assert v["huyet_ap"] is None
    assert v["spo2"] == 98


# ── parse_vitals: label:value parser ───────────────────────────────────────

def test_parse_label_value_plain():
    raw = "mạch: 75\nnhiệt độ: 36.5\nhuyết áp: 120/80\nspo2: 98"
    v = parse_vitals(raw)
    assert v["mach"] == 75
    assert v["nhiet_do"] == 36.5
    assert v["huyet_ap"] == {"tam_thu": 120, "tam_truong": 80}
    assert v["spo2"] == 98


def test_parse_label_value_numbered():
    raw = "1. Mạch: 75\n2. Nhiệt độ: 36.5\n3. Huyết áp: 120/80"
    v = parse_vitals(raw)
    assert v["mach"] == 75
    assert v["nhiet_do"] == 36.5
    assert v["huyet_ap"] == {"tam_thu": 120, "tam_truong": 80}


def test_parse_label_value_markdown_bullets():
    raw = "- **mạch:** 75\n- **nhiệt độ:** 36.5\n- **huyết áp:** 120/80"
    v = parse_vitals(raw)
    assert v["mach"] == 75
    assert v["huyet_ap"] == {"tam_thu": 120, "tam_truong": 80}


def test_parse_label_value_with_units():
    raw = "mạch: 75 lần/phút\nnhiệt độ: 36.5°C\ncân nặng: 67 kg\nspo2: 98%"
    v = parse_vitals(raw)
    assert v["mach"] == 75
    assert v["nhiet_do"] == 36.5
    assert v["can_nang"] == 67.0
    assert v["spo2"] == 98


def test_parse_label_value_null_handling():
    raw = "mạch: null\nnhiệt độ: -\nhuyết áp: 120/80\nspo2: không có"
    v = parse_vitals(raw)
    assert v["mach"] is None
    assert v["nhiet_do"] is None
    assert v["huyet_ap"] == {"tam_thu": 120, "tam_truong": 80}
    assert v["spo2"] is None


def test_parse_vlm_relabel_nhiptim():
    """VLM hallucinates 'Nhịp tim' but parser maps to mach (semantically correct)."""
    raw = "Mạch: 75\nNhịp tim: 72"  # VLM put 2 different things
    v = parse_vitals(raw)
    # Last "mach"-matching line wins (Nhịp tim → mach overrides)
    assert v["mach"] == 72


def test_parse_label_typo_chinhcao():
    """User-reported case: VLM wrote 'Chỉnh cao' instead of 'chiều cao'."""
    raw = "1. Mạch: 75\n6. Chỉnh cao: 160"
    v = parse_vitals(raw)
    assert v["mach"] == 75
    assert v["chieu_cao"] == 160


# ── No-match / fallback ────────────────────────────────────────────────────

def test_parse_empty_input():
    v = parse_vitals("")
    assert all(v[k] is None for k in ("mach", "nhiet_do", "huyet_ap"))


def test_parse_unrelated_garbage():
    v = parse_vitals("hello world this is not vitals")
    assert all(v[k] is None for k in ("mach", "nhiet_do", "huyet_ap", "spo2"))


def test_parse_lcd_format():
    """LCD-style monitor output: SYS/DIA/PUL keywords."""
    raw = "SYS 120\nDIA 80\nPUL 75"
    v = parse_vitals(raw)
    assert v["huyet_ap"] == {"tam_thu": 120, "tam_truong": 80}
    assert v["mach"] == 75
