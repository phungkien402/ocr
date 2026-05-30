"""Validator for extracted vital signs."""

from .config import VITALS_INFO


def validate_vitals(vitals: dict) -> tuple:
    """Validate vitals against normal ranges.

    Returns:
        (validation_dict, missing_fields_list)
    """
    validation = {}
    missing_fields = []

    for field, info in VITALS_INFO.items():
        value = vitals.get(field)

        if value is None:
            missing_fields.append(field)
            validation[field] = {"status": "missing", "value": None}
            continue

        normal_range = info.get("normal_range")
        if normal_range is None:
            validation[field] = {"status": "ok", "value": value}
            continue

        # Blood pressure special case
        if field == "huyet_ap" and isinstance(value, dict):
            sys_val = value.get("tam_thu")
            dia_val = value.get("tam_truong")
            sys_range = normal_range.get("tam_thu", [0, 999])
            dia_range = normal_range.get("tam_truong", [0, 999])
            sys_ok = sys_val is None or (sys_range[0] <= sys_val <= sys_range[1])
            dia_ok = dia_val is None or (dia_range[0] <= dia_val <= dia_range[1])
            status = "normal" if (sys_ok and dia_ok) else "abnormal"
            validation[field] = {"status": status, "value": value}
            continue

        # Numeric range check
        if isinstance(normal_range, list) and len(normal_range) == 2:
            lo, hi = normal_range
            if isinstance(value, (int, float)):
                status = "normal" if lo <= value <= hi else "abnormal"
            else:
                status = "ok"
            validation[field] = {"status": status, "value": value}
        else:
            validation[field] = {"status": "ok", "value": value}

    return validation, missing_fields
