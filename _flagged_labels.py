# _flagged_labels.py — DeepScanAI
# ---------------------------------
# Content policy label definitions for the Nnet classifier.
# Kept in a separate module so content_detector.py stays clean.
# Imported by content_detector.py at runtime.

# label_key → (display_category, subcategory, base_score)
LABEL_MAP = {
    "MALE_GENITALIA_EXPOSED":   ("Content Flag", "Policy Violation — Class A",         100.0),
    "FEMALE_GENITALIA_EXPOSED": ("Content Flag", "Policy Violation — Class A",         100.0),
    "ANUS_EXPOSED":             ("Content Flag", "Policy Violation — Class A",         100.0),
    "FEMALE_BREAST_EXPOSED":    ("Content Flag", "Policy Violation — Class A",         100.0),
    "BUTTOCKS_EXPOSED":         ("Content Flag", "Policy Violation — Class A",         100.0),
    "FEMALE_BREAST_COVERED":    ("Content Flag", "Dress Code Advisory — Moderate",      65.0),
    "MALE_BREAST_EXPOSED":      ("Content Flag", "Dress Code Advisory — Low Severity",  65.0),
    "BELLY_EXPOSED":            ("Content Flag", "Dress Code Advisory — Low Severity",  70.0),
    "BELLY_COVERED":            ("Content Flag", "Dress Code Advisory — Minimal",       50.0),
}

# ── Content Safety Classifier (loaded here to keep content_detector.py clean) ──
try:
    from nudenet import NudeDetector as _NnetDetector
    nnet_instance = _NnetDetector()
    NNET_AVAILABLE = True
except Exception as _e:
    print(f"[_flagged_labels] Content classifier unavailable: {_e}")
    nnet_instance = None
    NNET_AVAILABLE = False

# label_key → human-readable reason string
REASON_MAP = {
    "MALE_GENITALIA_EXPOSED":   "Class-A content policy violation detected",
    "FEMALE_GENITALIA_EXPOSED": "Class-A content policy violation detected",
    "ANUS_EXPOSED":             "Class-A content policy violation detected",
    "FEMALE_BREAST_EXPOSED":    "Class-A content policy violation detected",
    "BUTTOCKS_EXPOSED":         "Class-A content policy violation detected",
    "FEMALE_BREAST_COVERED":    "Dress code advisory — moderate severity",
    "MALE_BREAST_EXPOSED":      "Dress code advisory — low severity",
    "BELLY_EXPOSED":            "Dress code advisory — low severity",
    "BELLY_COVERED":            "Dress code advisory — minimal severity",
}