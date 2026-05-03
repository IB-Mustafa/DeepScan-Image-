"""
analyzer.py — DeepScanAI
=========================
Thin orchestrator. Delegates to two specialist modules:

  ai_detector.py       — AI generation & deepfake forensics (5 signals)
  content_detector.py  — Sensitive content (6 detection layers)

This file contains NO detection logic of its own.

Public API
----------
  analyze_image(path: str) -> dict
    {
        "deepfake":     { ai_score, deepfake_score, reasoning, signal_scores },
        "sensitive":    { category: { subcategory, score, reason } },
        "overall_risk": "Low" | "Medium" | "High",
        "authentic":    bool,
    }
"""

from ai_detector      import run_ai_detection
from content_detector import run_content_detection


def analyze_image(path: str) -> dict:
    """
    Analyze a single image for both AI generation and sensitive content.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    dict with keys:
        deepfake      — AI / deepfake scores and signal breakdown
        sensitive     — detected sensitive categories with scores and reasons
        overall_risk  — "Low" | "Medium" | "High"
        authentic     — True only if AI score is low AND no sensitive content found
    """
    deepfake_result  = run_ai_detection(path)
    sensitive_result = run_content_detection(path)

    ai_score      = deepfake_result.get("ai_score") or 0.0
    sensitive_max = max((v["score"] for v in sensitive_result.values()), default=0.0)
    combined_max  = max(ai_score, sensitive_max)

    if   combined_max >= 70: risk = "High"
    elif combined_max >= 40: risk = "Medium"
    else:                    risk = "Low"

    authentic = (ai_score < 40) and (len(sensitive_result) == 0)

    return {
        "deepfake":     deepfake_result,
        "sensitive":    sensitive_result,
        "overall_risk": risk,
        "authentic":    authentic,
    }