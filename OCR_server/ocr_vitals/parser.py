"""Parser for vital signs extracted by OCR engine."""
from __future__ import annotations
import json, logging, re, unicodedata
from .config import FIELD_KEYWORDS

logger = logging.getLogger(__name__)

LCD_LABELS = {
    "sys": "huyet_ap.tam_thu", "systolic": "huyet_ap.tam_thu",
    "dia": "huyet_ap.tam_truong", "diastolic": "huyet_ap.tam_truong",
    "pul": "mach", "pulse": "mach", "pul/min": "mach", "pulse/min": "mach",
}

_LABEL_MAP = {
    "mach": "mach", "mạch": "mach", "pulse": "mach", "hr": "mach", "heart rate": "mach",
    "nhiet do": "nhiet_do", "nhiệt độ": "nhiet_do", "temp": "nhiet_do", "temperature": "nhiet_do",
    "huyet ap": "huyet_ap", "huyết áp": "huyet_ap", "blood pressure": "huyet_ap", "bp": "huyet_ap",
    "nhip tho": "nhip_tho", "nhịp thở": "nhip_tho", "respiratory rate": "nhip_tho", "rr": "nhip_tho",
    "can nang": "can_nang", "cân nặng": "can_nang", "weight": "can_nang",
    "chieu cao": "chieu_cao", "chiều cao": "chieu_cao", "chỉu cao": "chieu_cao",
    "chiu cao": "chieu_cao", "height": "chieu_cao",
    "spo2": "spo2", "sp02": "spo2", "o2": "spo2",
}

# JSON key aliases: model may output Vietnamese keys or wrong casing
_KEY_ALIASES = {
    "mach": "mach", "mạch": "mach", "mac": "mach",
    "nhiet_do": "nhiet_do", "nhiệt_độ": "nhiet_do", "nhietdo": "nhiet_do",
    "nhip_tho": "nhip_tho", "nhịp_thở": "nhip_tho", "nhiptho": "nhip_tho",
    "can_nang": "can_nang", "cân_nặng": "can_nang", "cannang": "can_nang",
    "chieu_cao": "chieu_cao", "chiều_cao": "chieu_cao", "chieucao": "chieu_cao",
    "huyet_ap": "huyet_ap", "huyết_áp": "huyet_ap", "huyetap": "huyet_ap",
    "spo2": "spo2", "sp02": "spo2", "spo₂": "spo2", "sao2": "spo2",
    "o2 sat": "spo2", "o2sat": "spo2", "oxygen": "spo2",
}

_EMPTY_VITALS = {
    "mach": None, "nhiet_do": None, "huyet_ap": None,
    "nhip_tho": None, "can_nang": None, "chieu_cao": None, "spo2": None,
}


def _normalize_key(key: str) -> str:
    """Lowercase + strip diacritics + check aliases."""
    k = unicodedata.normalize("NFC", key).lower().replace(" ", "_")
    k_ascii = "".join(
        c for c in unicodedata.normalize("NFD", k)
        if unicodedata.category(c) != "Mn"
    )
    return _KEY_ALIASES.get(k, _KEY_ALIASES.get(k_ascii, k_ascii))


def parse_vitals(raw_text: str) -> dict:
    """Parse vital signs from raw OCR text. Tries parsers in order."""
    result = _parse_json(raw_text)
    if result and _count_found(result) >= 1:
        logger.info("JSON parser: %d field(s) found", _count_found(result))
        return result

    result = _parse_label_value(raw_text)
    if result and _count_found(result) >= 1:
        logger.info("Label:value parser: %d field(s) found", _count_found(result))
        return result

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


# ── 1. JSON parser ──────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Fix "huyet_ap": 110/65  →  nested dict
    text = re.sub(
        r'"huyet_ap"\s*:\s*(\d+)\s*/\s*(\d+)',
        r'"huyet_ap": {"tam_thu": \1, "tam_truong": \2}',
        text,
    )

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    # Normalize all keys
    data = {_normalize_key(k): v for k, v in data.items()}

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

    vitals["mach"]     = _int("mach")
    vitals["nhiet_do"] = _float("nhiet_do")
    vitals["nhip_tho"] = _int("nhip_tho")
    vitals["can_nang"] = _float("can_nang")
    vitals["chieu_cao"] = _float("chieu_cao")
    vitals["spo2"]     = _int("spo2")

    # Blood pressure
    bp = data.get("huyet_ap")
    if isinstance(bp, dict):
        bp = {_normalize_key(k): v for k, v in bp.items()}
        sys_v = _safe_int(bp.get("tam_thu"))
        dia_v = _safe_int(bp.get("tam_truong"))
        if sys_v is not None or dia_v is not None:
            vitals["huyet_ap"] = {"tam_thu": sys_v, "tam_truong": dia_v}
    elif isinstance(bp, str):
        m = re.match(r"(\d+)\s*/\s*(\d+)", bp.strip())
        if m:
            vitals["huyet_ap"] = {"tam_thu": int(m.group(1)), "tam_truong": int(m.group(2))}

    # Glucometer
    dh = data.get("duong_huyet")
    if dh is not None:
        try:
            vitals["duong_huyet"] = float(dh)
        except (TypeError, ValueError):
            pass
    don_vi = data.get("don_vi")
    if don_vi and isinstance(don_vi, str):
        vitals["don_vi"] = don_vi

    # Lab report
    lab = data.get("lab_results") or data.get("results")
    if isinstance(lab, dict) and lab:
        vitals["lab_results"] = lab

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


# ── 2. Label: value parser ──────────────────────────────────────────────────

def _parse_label_value(text: str) -> dict | None:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    vitals = dict(_EMPTY_VITALS)
    found = False
    for line in lines:
        m = re.match(r"^(.+?)\s*[:：]\s*(.+)$", line)
        if not m:
            continue
        label = m.group(1).strip().rstrip("-*•").strip()
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


# ── 3. LCD + Keyword regex ──────────────────────────────────────────────────

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
    if sys_v is not None or dia_v is not None:
        result["huyet_ap"] = {"tam_thu": sys_v, "tam_truong": dia_v}
    return result


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fuzzy_label(label: str) -> str | None:
    normalized = label.lower().strip()
    # Direct match
    if normalized in _LABEL_MAP:
        return _LABEL_MAP[normalized]
    # Strip diacritics and try again
    nd = "".join(c for c in unicodedata.normalize("NFKD", normalized)
                 if not unicodedata.combining(c))
    if nd in _LABEL_MAP:
        return _LABEL_MAP[nd]
    # Partial match
    for key, field in _LABEL_MAP.items():
        if key in normalized or normalized in key:
            return field
    return None


def _parse_value(field: str, value: str):
    if field == "huyet_ap":
        m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", value)
        if m:
            return {"tam_thu": int(m.group(1)), "tam_truong": int(m.group(2))}
        return None
    if field in ("nhiet_do", "can_nang", "chieu_cao"):
        m = re.search(r"(\d+[.,]\d+|\d+)", value)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                return None
        return None
    m = re.search(r"(\d+)", value)
    return int(m.group(1)) if m else None


def _extract_bp(text: str) -> dict | None:
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if m:
        return {"tam_thu": int(m.group(1)), "tam_truong": int(m.group(2))}
    return None


def _extract_int(text: str, field: str) -> int | None:
    pos = _kw_pos(text, field)
    if pos == -1:
        return None
    m = re.search(r"(\d+)", text[pos:pos+30])
    return int(m.group(1)) if m else None


def _extract_float(text: str, field: str) -> float | None:
    pos = _kw_pos(text, field)
    if pos == -1:
        return None
    m = re.search(r"(\d+[.,]\d+|\d+)", text[pos:pos+30])
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def _rm_diac(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def _kw_pos(text: str, field: str) -> int:
    keywords = FIELD_KEYWORDS.get(field, [])
    txt_nd = _rm_diac(text)
    for kw in keywords:
        pos = text.find(kw.lower())
        if pos != -1:
            return pos
        pos = txt_nd.find(_rm_diac(kw.lower()))
        if pos != -1:
            return pos
    return -1


def _merge(a: dict, b: dict) -> dict:
    result = dict(a)
    for k, v in b.items():
        if result.get(k) is None and v is not None:
            result[k] = v
    return result


def _count_found(vitals: dict) -> int:
    return sum(1 for k, v in vitals.items()
               if not k.startswith("_") and v is not None)
