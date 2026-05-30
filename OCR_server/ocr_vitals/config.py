"""Configuration for OCR vital signs pipeline."""

try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

VITALS_INFO = {
    "mach": {
        "label_vn": "Mạch",
        "label_en": "Pulse / Heart Rate",
        "unit": "lần/phút",
        "normal_range": [60, 100],
    },
    "nhiet_do": {
        "label_vn": "Nhiệt độ",
        "label_en": "Temperature",
        "unit": "°C",
        "normal_range": [36.1, 37.2],
    },
    "huyet_ap": {
        "label_vn": "Huyết áp",
        "label_en": "Blood Pressure (SYS/DIA)",
        "unit": "mmHg",
        "normal_range": {"tam_thu": [90, 120], "tam_truong": [60, 80]},
    },
    "nhip_tho": {
        "label_vn": "Nhịp thở",
        "label_en": "Respiratory Rate",
        "unit": "lần/phút",
        "normal_range": [12, 20],
    },
    "can_nang": {
        "label_vn": "Cân nặng",
        "label_en": "Weight",
        "unit": "kg",
        "normal_range": None,
    },
    "chieu_cao": {
        "label_vn": "Chiều cao",
        "label_en": "Height",
        "unit": "cm",
        "normal_range": None,
    },
    "spo2": {
        "label_vn": "SpO2",
        "label_en": "Oxygen Saturation",
        "unit": "%",
        "normal_range": [95, 100],
    },
}

FIELD_KEYWORDS = {
    "mach":     ["mạch", "mach", "pulse", "pul", "hr", "heart rate"],
    "nhiet_do": ["nhiệt độ", "nhiet do", "temp", "temperature", "nhiệt"],
    "huyet_ap": ["huyết áp", "huyet ap", "blood pressure", "bp", "ha", "sys", "dia"],
    "nhip_tho": ["nhịp thở", "nhip tho", "respiratory", "rr", "nhịp"],
    "can_nang": ["cân nặng", "can nang", "weight"],
    "chieu_cao": ["chiều cao", "chieu cao", "height"],
    "spo2":     ["spo2", "sp02", "spо2", "o2", "oxygen"],
}
