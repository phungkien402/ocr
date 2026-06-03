"""Tests for ocr_vitals.validator."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from ocr_vitals.validator import validate_vitals


def test_all_normal():
    vitals = {"mach": 75, "nhiet_do": 36.5,
              "huyet_ap": {"tam_thu": 120, "tam_truong": 80},
              "nhip_tho": 16, "spo2": 98}
    val, missing = validate_vitals(vitals)
    assert missing == ["can_nang", "chieu_cao"]
    assert val["mach"]["status"] == "normal"
    assert val["nhiet_do"]["status"] == "normal"
    assert val["huyet_ap"]["status"] == "normal"
    assert val["spo2"]["status"] == "normal"


def test_bp_abnormal_systolic():
    vitals = {"huyet_ap": {"tam_thu": 180, "tam_truong": 80}}
    val, _ = validate_vitals(vitals)
    assert val["huyet_ap"]["status"] == "abnormal"


def test_bp_abnormal_diastolic():
    vitals = {"huyet_ap": {"tam_thu": 120, "tam_truong": 110}}
    val, _ = validate_vitals(vitals)
    assert val["huyet_ap"]["status"] == "abnormal"


def test_pulse_boundary():
    """Range 60-100. Test exact boundaries."""
    assert validate_vitals({"mach": 60})[0]["mach"]["status"] == "normal"
    assert validate_vitals({"mach": 100})[0]["mach"]["status"] == "normal"
    assert validate_vitals({"mach": 59})[0]["mach"]["status"] == "abnormal"
    assert validate_vitals({"mach": 101})[0]["mach"]["status"] == "abnormal"


def test_temp_float():
    assert validate_vitals({"nhiet_do": 36.5})[0]["nhiet_do"]["status"] == "normal"
    assert validate_vitals({"nhiet_do": 38.0})[0]["nhiet_do"]["status"] == "abnormal"


def test_spo2_low():
    assert validate_vitals({"spo2": 94})[0]["spo2"]["status"] == "abnormal"
    assert validate_vitals({"spo2": 95})[0]["spo2"]["status"] == "normal"


def test_missing_field():
    val, missing = validate_vitals({"mach": 75})
    assert "nhiet_do" in missing
    assert "huyet_ap" in missing
    assert val["nhiet_do"]["status"] == "missing"


def test_bp_partial_only_systolic():
    """BP dict with only tam_thu set, tam_truong None."""
    vitals = {"huyet_ap": {"tam_thu": 120, "tam_truong": None}}
    val, _ = validate_vitals(vitals)
    # Should still validate (dia None is treated as ok=True)
    assert val["huyet_ap"]["status"] == "normal"


def test_height_no_range():
    """can_nang and chieu_cao have no normal_range — always 'ok' if present."""
    val, _ = validate_vitals({"chieu_cao": 170, "can_nang": 67})
    assert val["chieu_cao"]["status"] == "ok"
    assert val["can_nang"]["status"] == "ok"
