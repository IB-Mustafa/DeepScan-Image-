# _flagged_labels.py — DeepScanAI
# ---------------------------------
# Content policy label definitions for the Nnet classifier.
# Kept in a separate module so content_detector.py stays clean.
# Imported by content_detector.py at runtime.

# ── Classifier label aliases ──────────────────────────────────────────────────
# Clean internal names mapped to classifier's expected label strings.
# This keeps the codebase readable while staying compatible with the model output.

# label_key → (display_category, subcategory, base_score)
LABEL_MAP = {}

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
REASON_MAP = {}