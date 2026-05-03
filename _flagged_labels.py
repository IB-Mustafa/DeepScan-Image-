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
    "FEMALE_BREAST_EXPOSED":    ("Content Flag", "Policy Violation — Class B",          85.0),
    "MALE_BREAST_EXPOSED":      ("Content Flag", "Dress Code Advisory — Low Severity",  30.0),
    "BUTTOCKS_EXPOSED":         ("Content Flag", "Policy Violation — Class B",          85.0),
    "FEMALE_BREAST_COVERED":    ("Content Flag", "Dress Code Advisory — Moderate",      45.0),
    "BELLY_EXPOSED":            ("Content Flag", "Dress Code Advisory — Low Severity",  30.0),
    "BELLY_COVERED":            ("Content Flag", "Dress Code Advisory — Minimal",       15.0),
}

# label_key → human-readable reason string
REASON_MAP = {
    "MALE_GENITALIA_EXPOSED":   "Class-A content policy violation detected",
    "FEMALE_GENITALIA_EXPOSED": "Class-A content policy violation detected",
    "ANUS_EXPOSED":             "Class-A content policy violation detected",
    "FEMALE_BREAST_EXPOSED":    "Class-B content policy violation detected",
    "MALE_BREAST_EXPOSED":      "Dress code advisory — low severity",
    "BUTTOCKS_EXPOSED":         "Class-B content policy violation detected",
    "FEMALE_BREAST_COVERED":    "Dress code advisory — moderate severity",
    "BELLY_EXPOSED":            "Dress code advisory — low severity",
    "BELLY_COVERED":            "Dress code advisory — minimal severity",
}