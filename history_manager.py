"""
history_manager.py
------------------
Handles reading / writing scan history to a local JSON file.
No database needed — flat list stored in history.json.
"""

import json
import os
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")


def _load_raw() -> list:
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_raw(records: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def add_record(image_path: str, results: dict) -> None:
    """
    Append one scan result to history.json.
    Stored fields: name, path, timestamp, overall_risk, authentic,
                   ai_score, deepfake_score, sensitive
    """
    records = _load_raw()

    deepfake       = results.get("deepfake", {})
    ai_score       = deepfake.get("ai_score")
    deepfake_score = deepfake.get("deepfake_score")

    # Strip 'reason' from sensitive dict before saving (keeps file clean)
    sensitive_clean = {}
    for cat, info in results.get("sensitive", {}).items():
        sensitive_clean[cat] = {
            "subcategory": info.get("subcategory", ""),
            "score":       info.get("score", 0.0),
        }

    record = {
        "name":           os.path.basename(image_path),
        "path":           image_path,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_risk":   results.get("overall_risk", "Low"),
        "authentic":      results.get("authentic", True),
        "ai_score":       ai_score,
        "deepfake_score": deepfake_score,
        "sensitive":      sensitive_clean,
    }

    records.insert(0, record)   # newest first
    records = records[:200]     # keep at most 200 entries
    _save_raw(records)


def get_all_records() -> list:
    """Return all history records, newest first."""
    return _load_raw()


def get_stats() -> dict:
    """
    Aggregate stats for the Dashboard cards.
    Returns { total, ai_detected, sensitive, clean }
    """
    records      = _load_raw()
    total        = len(records)
    ai_detected  = sum(1 for r in records
                       if r.get("ai_score") is not None and r["ai_score"] >= 50)
    sensitive    = sum(1 for r in records if r.get("sensitive"))
    clean        = sum(1 for r in records if r.get("authentic", False))

    return {
        "total":       total,
        "ai_detected": ai_detected,
        "sensitive":   sensitive,
        "clean":       clean,
    }


def clear_history() -> None:
    """Wipe all history."""
    _save_raw([])