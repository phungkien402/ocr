"""Parser for vital signs extracted by OCR engine.

Priority order:
  1. JSON parse (new primary path — Qwen3-VL outputs JSON directly)
  2. Simple "Label: value" line-by-line (legacy Qwen3 text format)
  3. LCD label parsing (SYS/DIA/PUL patterns)
  4. Vietnamese keyword regex (last resort)
"""

from __future__ import annotations  # Python 3.9 compatibility

import json
import logging
import re
import unicodedata

from .config import FIELD_KEYWORDS

logger = logging.getLogger(__name__)

# English LCD labels
LCD_LABELS = {
    "sys": "huyet_ap.tam_thu", "systolic": "huyet_ap.tam_thu",
    "dia": "huyet_ap.tam_truong", "diastolic": "huyet_ap.tam_truong",
    "pul": "mach", "pulse": "mach", "pul/min": "mach", "pulse/min": "mach",
}
UNIT_LABELS = {"mmhg", "kpa", "bpm", "/min", "min"}

# Fuzzy label → field map
_LABEL_MAP = {
    "mach": "mach", "mạch": "mach", "pulse": "mach", "hr": "mach", "heart rate": "mach",
    "nhiet do": "nhiet_do", "nhiệt độ": "nhiet_do", "temp": "nhiet_do", "temperature": "nhiet_do",
    "huyet ap": "huyet_ap", "huyết áp": "huyet_ap", "blood pressure": "huyet_ap", "bp": "huyet_ap", "ha": "huyet_ap",
    "nhip tho": "nhip_tho", "nhịp thở": "nhip_tho", "respiratory rate": "nhip_tho", "rr": "nhip_tho",
    "can nang": "can_nang", "cân nặng": "can_nang", "weight": "can_nang",
    "chieu cao": "chieu_cao", "chiều cao": "chieu_cao", "height": "chieu_cao",
    "spo2": "spo2", "sp02": "spo2", "o2": "spo2",
}

_EMPTY_VITALS = {
    "mach": None, "nhiet_do": None, "huyet_ap": None,
    "nhip_tho": None, "can_nang": None, "chieu_cao": None, "spo2": None,
}


def parse_vitals(raw_text: str) -> dict:
    """Parse vital signs from raw OCR text.

    Tries parsers in order of reliability; returns first successful result
    with at least one non-null field.
    """
    # 1. JSON (Qwen3-VL primary output)
    result = _parse_json(raw_text)
    if result and _count_found(result) >= 1:
        logger.info("JSON parser: %d field(s) found", _count_found(result))
        return result

    # 2. Simple "Label: value" lines
    result = _parse_label_value(raw_text)
    if result and _count_found(result) >= 1:
        logger.info("Label:value parser: %d field(s) found", _count_found(result))
        return result

    # 3. LCD (SYS/DIA/PUL) + Vietnamese keyword regex
    norm = raw_text.lower().strip()
    lcd = _parse_lcd(norm)
    kw = {
        "mach":     _extract_int(norm, "mach"),
        "nhiet_do": _extract_float(norm, "nhiet_do"),
        "huyet_ap": _extract_bp(norm),
        "nhip_tho": _extract_int(norm, "nhip_tho"),
        "can_nang": _extract_float(norm, "can_nang"),
        "chieu_cao": _extract_int(norm, "chieu_cao"),
        "spo2":     _extract_int(norm, "spo2"),
    }
    result = _merge(kw, lcd)
    logger.info("Regex parser: %d field(s) found", _count_found(result))
    return result


# ─────────────────────────────────────────────
# 1. JSON parser
# ─────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    """Extract JSON from model output and map to vitals dict."""
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    vitals = dict(_EMPTY_VITALS)

    def _int(key):
        v = data.get(key)
        if v is None or str(v).lower() in ("null", "none", ""):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    def _float(key):
        v = data.get(key)
        if v is None or str(v).lower() in ("null", "none", ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    vitals["mach"] = _int("mach")
    vitals["nhiet_do"] = _float("nhiet_do")
    vitals["nhip_tho"] = _int("nhip_tho")
    vitals["can_nang"] = _float("can_nang")
    vitals["chieu_cao"] = _float("chieu_cao")
    vitals["spo2"] = _int("spo2")

    # Blood pressure: {"tam_thu": 120, "tam_truong": 80} or null
    bp = data.get("huyet_ap")
    if isinstance(bp, dict):
        sys = _safe_int(bp.get("tam_thu"))
        dia = _safe_int(bp.get("tam_truong"))
        if sys is not None or dia is not None:
            vitals["huyet_ap"] = {"tam_thu": sys, "tam_truong": dia}
    elif bp is None:
        vitals["huyet_ap"] = None

    # Capture device type if present
    device = data.get("device")
    if device and isinstance(device, str):
        vitals["_device"] = device

    return vitals


def _safe_int(v):
    if v is None or str(v).lower() in ("null", "none"):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# 2. Label: value parser
# ─────────────────────────────────────────────

def _parse_label_value(text: str) -> dict | None:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    vitals = dict(_EMPTY_VITALS)
    found = False
    for line in lines:
        m = re.match(r"^(.+?)\s*[:：]\s*(.+)$", line)
        if not m:
            continue
        label = m.group(1).strip().rstrip("-*•")
        value = m.group(2).strip()
        if value.lower() in ("null", "none", "n/a", "-", "không có"):
            continue
        field = _fuzzy_label(label)
        if field is None:
            continue
        parsed = _parse_value(field, value)
        if parsed is not None:
            vitals[field] = parsed
            found = True
    return vitals if found else None


# ─────────────────────────────────────────────
# 3. LCD + Keyword regex (legacy)
# ─────────────────────────────────────────────

def _parse_lcd(text: str) -> dict:
    result = {"mach": None, "huyet_ap": None}
    sys_v = dia_v = None
    for label, field in LCD_LABELS.items():
        m = re.search(re.escape(label) + r"[\s.:;=]*(\d{2,3})", text)
        if m:
            v = int(m.group(1))
            if field == "huyet_ap.tam_thu":
                sys_v = v
            elif field == "huyet_ap.tam_truong":
                dia_v = v
            elif field == "mach":
                result["mach"] = v
    # Line-by-line fallback
    if None in (sys_v, dia_v, result["mach"]):
        lb = _lcd_lines(text)
        if sys_v is None:
            sys_v = lb.get("tam_thu")
        if dia_v is None:
            dia_v = lb.get("tam_truong")
        if result["mach"] is None:
            result["mach"] = lb.get("mach")
    if sys_v is not None or dia_v is not None:
        result["huyet_ap"] = {"tam_thu": sys_v, "tam_truong": dia_v}
    return result


def _lcd_lines(text: str) -> dict:
    lines = text.split("\n")
    out = {"tam_thu": None, "tam_truong": None, "mach": None}
    for i, line in enumerate(lines):
        field = _lcd_label(line)
        if field is None:
            continue
        val = _digit_from_line(line)
        if val is None:
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j].strip()
                if not nxt or _lcd_label(nxt):
                    break
                m = re.search(r"^(\d{2,3})$", nxt) or re.search(r"(\d{2,3})", nxt)
                if m:
                    val = int(m.group(1))
                    break
        if val is not None:
            out[field] = val
    return out


def _lcd_label(line: str):
    line = line.strip().lower()
    if re.search(r"\bsys\b|systolic", line):
        return "tam_thu"
    if re.search(r"\bdia\b|diastolic", line):
        return "tam_truong"
    if re.search(r"\bpul\b|\bpulse\b", line):
        return "mach"
    return None


def _digit_from_line(line: str):
    cleaned = re.sub(r"(sys|dia|pul|pulse|systolic|diastolic|/min)", "", line, flags=re.IGNORECASE)
    m = re.search(r"(\d{2,3})", cleaned)
    return int(m.group(1)) if m else None


def _merge(kw: dict, lcd: dict) -> dict:
    v = kw.copy()
    if lcd.get("mach") is not None:
        v["mach"] = lcd["mach"]
    lcd_bp = lcd.get("huyet_ap")
    if lcd_bp:
        if v["huyet_ap"] is None:
            v["huyet_ap"] = {}
        for k in ("tam_thu", "tam_truong"):
            if lcd_bp.get(k) is not None:
                v["huyet_ap"][k] = lcd_bp[k]
        if v["huyet_ap"].get("tam_thu") is None and v["huyet_ap"].get("tam_truong") is None:
            v["huyet_ap"] = None
    return v


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _count_found(vitals: dict) -> int:
    return sum(1 for k, v in vitals.items() if k != "_units" and v is not None)


def _fuzzy_label(label: str) -> str | None:
    lo = label.lower().strip()
    if lo in _LABEL_MAP:
        return _LABEL_MAP[lo]
    nd = _rm_diac(lo)
    if nd in _LABEL_MAP:
        return _LABEL_MAP[nd]
    for k, f in _LABEL_MAP.items():
        if _rm_diac(k) == nd:
            return f
    for k, f in _LABEL_MAP.items():
        knd = _rm_diac(k)
        if len(knd) >= 4 and knd in nd:
            return f
    return None


def _parse_value(field: str, s: str):
    s = s.strip()
    if field == "huyet_ap":
        m = re.search(r"(\d{2,3})\s*[/\\-]\s*(\d{2,3})", s)
        if m:
            return {"tam_thu": int(m.group(1)), "tam_truong": int(m.group(2))}
        return None
    if field in ("nhiet_do", "can_nang", "chieu_cao"):
        m = re.search(r"(\d+[.,]?\d*)", s)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                return None
        return None
    # integer fields
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _rm_diac(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def _kw_pos(text: str, field: str) -> int:
    keywords = FIELD_KEYWORDS.get(field, [])
    txt_nd = _rm_diac(text)
    for kw in keywords:
        pos = text.find(kw.lower())
        if pos != -1:
            return pos + len(kw)
        nd = _rm_diac(kw.lower())
        pos = txt_nd.find(nd)
        if pos != -1:
            return pos + len(nd)
    return -1


def _nearest_num(text: str, pos: int, is_float: bool = False):
    window = re.sub(r"^[\s.:,;=]+", "", text[pos:pos + 50])
    pat = r"(\d+[.,]\d+|\d+)" if is_float else r"(\d+)"
    m = re.search(pat, window)
    if not m:
        return None
    try:
        s = m.group(1).replace(",", ".")
        return float(s) if is_float else int(s)
    except ValueError:
        return None


def _extract_int(text: str, field: str):
    pos = _kw_pos(text, field)
    return _nearest_num(text, pos) if pos != -1 else None


def _extract_float(text: str, field: str):
    pos = _kw_pos(text, field)
    return _nearest_num(text, pos, is_float=True) if pos != -1 else None


def _extract_bp(text: str):
    pos = _kw_pos(text, "huyet_ap")
    areas = ([text[pos:pos + 50]] if pos != -1 else []) + [text]
    for area in areas:
        m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", area)
        if m:
            return {"tam_thu": int(m.group(1)), "tam_truong": int(m.group(2))}
        m = re.search(r"(\d{2,3})\s*[-\\]\s*(\d{2,3})", area)
        if m:
            return {"tam_thu": int(m.group(1)), "tam_truong": int(m.group(2))}
    return None
