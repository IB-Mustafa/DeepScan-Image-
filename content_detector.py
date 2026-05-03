"""
content_detector.py — DeepScanAI
==================================
Sensitive content detection. No AI-generation logic.

Detection layers:
  1. Nnet (restricted scope) — weapons, blood, injury, medical, content policy flags
  2. Blood & Hemorrhage      — vivid red HSV + morphological analysis
                               Sub-types: splatter, pooling, smear patterns
  3. Scene Threat Analysis   — dark environments, looming shapes, posture cues
                               Sub-types: low-visibility, shadow threat, cowering
  4. Weapon-in-Hand          — power-grip + elongated rigid protrusion
                               Sub-types: bladed, blunt, firearm-shaped
  5. Smoke & Fire Hazard     — orange/yellow flame + grey smoke HSV zones

Public API
----------
  run_content_detection(path: str) -> dict
    { category: { subcategory: str, score: float, reason: str } }
"""

import numpy as np
import _flagged_labels as _FL

# ── OpenCV ────────────────────────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("[content_detector] OpenCV not available — visual detection disabled")

# ── Nnet — content safety classifier ─────────────────────────────────────────
try:
    from nudenet import NudeDetector as _NnetDetector
    _nnet = _NnetDetector()
    NNET_OK = True
    print("[content_detector] ✓ Nnet loaded")
except Exception as e:
    print(f"[content_detector] Nnet unavailable: {e}")
    _nnet = None
    NNET_OK = False

_NNET_MAP = {
    "GUN_POINTED":              ("Restricted Item",  "Firearm — Directed at Subject",     65.0),
    "GUN_NOT_POINTED":          ("Restricted Item",  "Firearm — Present in Scene",        55.0),
    "KNIFE":                    ("Restricted Item",  "Bladed Object Detected",            52.0),
    "WEAPON":                   ("Restricted Item",  "Unclassified Prohibited Object",    50.0),
    "BLOOD":                    ("Hemorrhage Alert", "Blood — Confirmed by Classifier",   68.0),
    "INJURY":                   ("Physical Trauma",  "Visible Injury Detected",           62.0),
    "MEDICAL":                  ("Clinical Content", "Clinical / Surgical Scene",         42.0),
    "MEDICAL_EQUIPMENT":        ("Clinical Content", "Medical Equipment in Frame",        32.0),
    **_FL.LABEL_MAP,
}

_NNET_THRESHOLD = 0.22

_NNET_REASONS = {
    "GUN_POINTED":              "Firearm identified in directed/aimed orientation",
    "GUN_NOT_POINTED":          "Firearm present but not actively directed",
    "KNIFE":                    "Bladed object identified with moderate-to-high confidence",
    "WEAPON":                   "Unclassified prohibited object flagged by classifier",
    "BLOOD":                    "Blood confirmed by Nnet classifier",
    "INJURY":                   "Physical injury markers identified in frame",
    "MEDICAL":                  "Clinical or surgical scene context detected",
    "MEDICAL_EQUIPMENT":        "Medical device or equipment visible in scene",
    **_FL.REASON_MAP,
}


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _load_image(path: str):
    if not CV2_OK:
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (512, 512))
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _zone_ratio(mask, y0: int, y1: int) -> float:
    zone = mask[y0:y1, :]
    return float(zone.sum()) / 255.0 / max(zone.size, 1)

def _has_frontal_face(gray) -> bool:
    try:
        fc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return len(fc.detectMultiScale(gray, 1.1, 4)) > 0
    except Exception:
        return False

# Shared face cascade instance
_face_cascade = None
def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None and CV2_OK:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _face_cascade


# =============================================================================
# LAYER 1 — Nnet content safety classifier
# =============================================================================
def _detect_nnet(path: str) -> dict:
    if not NNET_OK or _nnet is None:
        return {}
    try:
        raw = _nnet.detect(path)
    except Exception as e:
        print(f"[content_detector] Nnet error: {e}")
        return {}

    aggregated: dict = {}
    for det in raw:
        label_key = det.get("class", "").upper()
        conf = float(det.get("score", 0.0))
        if conf < _NNET_THRESHOLD or label_key not in _NNET_MAP:
            continue
        display_label, subcategory, base_score = _NNET_MAP[label_key]
        effective = round(min(base_score * (0.7 + conf * 0.3) + conf * 15, 100.0), 1)
        effective = max(effective, round(base_score * 0.90, 1))
        reason    = _NNET_REASONS.get(label_key, f"Flagged with {conf*100:.0f}% confidence")
        if display_label not in aggregated or effective > aggregated[display_label]["score"]:
            aggregated[display_label] = {
                "subcategory": subcategory,
                "score":       effective,
                "reason":      reason,
            }
    return aggregated


# =============================================================================
# LAYER 2 — BLOOD & HEMORRHAGE
# =============================================================================
def _detect_blood(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        img_bgr, hsv, gray = loaded
        face_cascade = _get_face_cascade()

        # Face zone mask — exclude lipstick/makeup
        face_zone_mask = np.ones((512, 512), dtype=np.uint8) * 255
        if _has_frontal_face(gray) and face_cascade is not None:
            for (fx, fy, fw, fh) in face_cascade.detectMultiScale(gray, 1.1, 4):
                pad_x = int(fw * 0.30)
                pad_y = int(fh * 0.30)
                x1 = max(0, fx - pad_x);  x2 = min(511, fx + fw + pad_x)
                y1 = max(0, fy - pad_y);  y2 = min(511, fy + fh + pad_y)
                face_zone_mask[y1:y2, x1:x2] = 0

        # Broad red mask
        r1 = cv2.inRange(hsv, np.array([0,   70, 40]), np.array([10,  255, 220]))
        r2 = cv2.inRange(hsv, np.array([165, 70, 40]), np.array([180, 255, 220]))
        blood_mask = cv2.bitwise_or(r1, r2)

        # Fire/ketchup exclusion
        excl_a     = cv2.inRange(hsv, np.array([5,  100,  80]), np.array([30, 255, 255]))
        excl_b     = cv2.inRange(hsv, np.array([0,  160,  40]), np.array([5,  255, 139]))
        flame_zone = cv2.dilate(cv2.bitwise_or(excl_a, excl_b),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        blood_mask = cv2.bitwise_and(blood_mask, cv2.bitwise_not(flame_zone))
        blood_mask = cv2.bitwise_and(blood_mask, face_zone_mask)

        # Real face detection (sat guard filters fire false positives)
        real_faces = []
        if face_cascade is not None:
            for (fx, fy, fw, fh) in face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50)):
                if float(hsv[fy:fy+fh, fx:fx+fw][:, :, 1].mean()) < 160:
                    real_faces.append((fx, fy, fw, fh))

        # Body region from faces
        body_region = np.zeros((512, 512), dtype=np.uint8)
        for (fx, fy, fw, fh) in real_faces:
            pad_x = int(fw * 2.0)
            body_region[max(0, fy-int(fh*0.5)):min(511, fy+fh+int(fh*2.5)),
                        max(0, fx-pad_x):min(511, fx+fw+pad_x)] = 255

        # Large skin region = hand/arm context
        skin_a = cv2.inRange(hsv, np.array([0,  15, 150]), np.array([20, 170, 255]))
        skin_b = cv2.inRange(hsv, np.array([160,15, 150]), np.array([180,170, 255]))
        skin   = cv2.bitwise_or(skin_a, skin_b)
        if cv2.countNonZero(skin) / (512 * 512) > 0.20:
            skin_dilated  = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30)))
            combined_body = cv2.bitwise_or(body_region, skin_dilated)
        else:
            combined_body = body_region

        has_body   = cv2.countNonZero(combined_body) > 0
        blood_zone = cv2.bitwise_and(blood_mask, combined_body) if has_body else blood_mask
        blood_clean = cv2.morphologyEx(blood_zone,
                      cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        ratio = cv2.countNonZero(blood_clean) / (512 * 512)

        # Uniform object guard
        cnts_b, _ = cv2.findContours(blood_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_b:
            b_areas = [cv2.contourArea(c) for c in cnts_b]
            if max(b_areas) / max(sum(b_areas), 1) > 0.55 and sum(1 for a in b_areas if a > 200) <= 3:
                return {}

        threshold = 0.006 if has_body else 0.150
        if ratio < threshold:
            return {}

        cnts, _ = cv2.findContours(blood_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {}

        total_area   = sum(cv2.contourArea(c) for c in cnts)
        largest_area = max(cv2.contourArea(c) for c in cnts)
        count        = len(cnts)

        if largest_area / max(total_area, 1) > 0.70 and count <= 4:
            sub    = "Hemorrhage Pooling Pattern"
            reason = "Large contiguous blood-colored region — consistent with pooling"
            score  = round(min(55.0 + ratio * 1800, 95.0), 1)
        elif count > 12 and largest_area / max(total_area, 1) < 0.35:
            sub    = "Hemorrhage Splatter Pattern"
            reason = "Multiple dispersed red blobs — consistent with splatter"
            score  = round(min(50.0 + ratio * 1600, 92.0), 1)
        else:
            elongated = sum(1 for c in cnts
                           if cv2.contourArea(c) >= 300 and
                           max(*cv2.boundingRect(c)[2:]) / max(min(*cv2.boundingRect(c)[2:]), 1) > 3.5)
            if elongated >= 2:
                sub    = "Hemorrhage Smear Pattern"
                reason = "Elongated red streaks — consistent with smear or drag pattern"
                score  = round(min(48.0 + ratio * 1500, 88.0), 1)
            else:
                sub    = "Blood Presence Detected"
                reason = "Blood-consistent color and morphology detected"
                score  = round(min(45.0 + ratio * 1400, 90.0), 1)

        return {"Hemorrhage Alert": {"subcategory": sub, "score": score, "reason": reason}}
    except Exception as e:
        print(f"[content_detector] Blood detection error: {e}")
        return {}


# =============================================================================
# LAYER 3 — SCENE THREAT ANALYSIS
# =============================================================================
def _detect_distress(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        img_bgr, hsv, gray = loaded
        H = 512
        face_cascade = _get_face_cascade()

        if _has_frontal_face(gray):
            return {}

        if face_cascade is not None:
            if len(face_cascade.detectMultiScale(gray, 1.05, 3)) >= 2:
                return {}

        warm = cv2.inRange(hsv, np.array([5, 80, 180]), np.array([35, 255, 255]))
        if cv2.countNonZero(warm) / (H * H) > 0.005:
            return {}

        # Bright background guard — well-lit scenes (offices, studios, daylight)
        # are never threatening. Skip if >40% pixels are bright.
        bright_ratio = float((gray > 180).sum()) / (H * H)
        if bright_ratio > 0.40:
            return {}

        dark_ratio = float((gray < 40).sum()) / (H * H)
        if dark_ratio > 0.72:
            return {
                "Scene Threat": {
                    "subcategory": "Extreme Low-Visibility Environment",
                    "score":       round(60.0 + dark_ratio * 20, 1),
                    "reason":      f"Scene is {dark_ratio*100:.0f}% dark — no identifiable subject detected",
                }
            }

        upper_gray = gray[:int(H * 0.65), :]
        _, thresh  = cv2.threshold(upper_gray, 80, 255, cv2.THRESH_BINARY_INV)
        cnts, _    = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shadow_score  = 0.0
        dominant_area = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if area > 5000 or (area > 2000 and aspect > 1.5):
                shadow_score  += min(area / 5000.0, 1.0) * 35.0
                dominant_area  = max(dominant_area, int(area))
        shadow_score = min(shadow_score, 85.0)

        skin1 = cv2.inRange(hsv, np.array([0, 20, 50]),   np.array([25, 180, 255]))
        skin2 = cv2.inRange(hsv, np.array([168, 20, 50]), np.array([180, 180, 255]))
        skin  = cv2.bitwise_or(skin1, skin2)

        lower_center   = skin[int(H*0.55):, int(H*0.25):int(H*0.75)]
        cowering_ratio = lower_center.sum() / 255.0 / max(lower_center.size, 1)
        upper_dark     = float((gray[:int(H*0.50), :] < 80).sum()) / (int(H*0.50) * H)

        cowering_score = 0.0
        if cowering_ratio > 0.25 and upper_dark > 0.25 and shadow_score > 20:
            cowering_score = min(50.0 + cowering_ratio * 60.0, 88.0)

        best_score = max(shadow_score, cowering_score)
        if best_score < 45.0:
            return {}

        if cowering_score >= shadow_score:
            sub    = "Cowering / Submission Posture"
            reason = (f"Skin concentrated in lower-center ({cowering_ratio*100:.0f}%) "
                      f"with {upper_dark*100:.0f}% overhead darkness")
        else:
            sub    = "Looming Shadow Threat"
            reason = (f"Large dark contour (area ≈ {dominant_area}px²) in upper frame "
                      f"without identifiable face")

        return {"Scene Threat": {"subcategory": sub, "score": round(best_score, 1), "reason": reason}}
    except Exception as e:
        print(f"[content_detector] Distress detection error: {e}")
        return {}


# =============================================================================
# LAYER 4 — WEAPON IN HAND
# =============================================================================
def _detect_weapon_in_hand(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        _, hsv, gray = loaded

        if _has_frontal_face(gray):
            return {}

        skin_mask  = cv2.inRange(hsv, np.array([0,  20, 70]), np.array([20,  255, 255]))
        metal_mask = cv2.inRange(hsv, np.array([0,   0, 70]), np.array([180,  30, 220]))
        dark_mask  = cv2.inRange(hsv, np.array([0,   0,  0]), np.array([180,  50,  80]))

        hand_cnts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not hand_cnts:
            return {}

        main_hand = max(hand_cnts, key=cv2.contourArea)
        if cv2.contourArea(main_hand) < 6000:
            return {}

        hull    = cv2.convexHull(main_hand, returnPoints=False)
        defects = cv2.convexityDefects(main_hand, hull)
        gap_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                _, _, _, d = defects[i, 0]
                if d > 12000:
                    gap_count += 1
        if gap_count >= 3:
            return {}

        # Toy / prop guard — colorful scenes (>15% bright non-skin pixels)
        # indicate toys, props, or costumes — not real weapons
        colorful    = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([180, 255, 255]))
        non_skin_cl = cv2.bitwise_and(colorful, cv2.bitwise_not(skin_mask))
        if cv2.countNonZero(non_skin_cl) / (512 * 512) > 0.15:
            return {}   # toy or colorful prop — skip

        best_result = None
        for obj_mask, mask_name in [(metal_mask, "metal"), (dark_mask, "dark")]:
            isolated = cv2.bitwise_and(obj_mask, cv2.bitwise_not(skin_mask))
            if cv2.countNonZero(isolated) < 8000:
                continue
            obj_cnts, _ = cv2.findContours(isolated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not obj_cnts:
                continue
            ox, oy, ow, oh = cv2.boundingRect(max(obj_cnts, key=cv2.contourArea))
            short_side = min(ow, oh)
            long_side  = max(ow, oh)
            if short_side < 12:
                continue
            aspect = long_side / max(short_side, 1)
            if aspect < 3.0 or oy + oh < 200:
                continue

            if mask_name == "metal" and aspect > 5.0:
                sub, score = "Bladed Object in Grip", 85.0
                reason = f"Silver/grey elongated object (aspect {aspect:.1f}:1) in power-grip"
            elif mask_name == "dark" and 3.0 <= aspect <= 5.5:
                sub, score = "Blunt / Club-type Object in Grip", 75.0
                reason = f"Dark elongated object (aspect {aspect:.1f}:1) in closed grip"
            elif mask_name == "metal" and 3.0 <= aspect <= 5.0:
                sub, score = "Firearm-Shaped Object in Grip", 80.0
                reason = f"Metallic object with firearm proportions (aspect {aspect:.1f}:1)"
            else:
                sub, score = "Unclassified Elongated Object in Grip", 70.0
                reason = f"Elongated object (aspect {aspect:.1f}:1) in power-grip"

            if best_result is None or score > best_result["score"]:
                best_result = {"subcategory": sub, "score": score, "reason": reason}

        return {"Restricted Item": best_result} if best_result else {}
    except Exception as e:
        print(f"[content_detector] Weapon-in-hand error: {e}")
        return {}


# =============================================================================
# LAYER 5 — SMOKE & FIRE HAZARD
# =============================================================================
def _detect_fire_smoke(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        _, hsv, gray = loaded
        H = 512

        flame1 = cv2.inRange(hsv, np.array([5,  180, 150]), np.array([30, 255, 255]))
        flame2 = cv2.inRange(hsv, np.array([0,  200, 180]), np.array([8,  255, 255]))
        flame  = cv2.morphologyEx(cv2.bitwise_or(flame1, flame2),
                 cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        flame_ratio = cv2.countNonZero(flame) / (H * H)

        flame_cnts, _ = cv2.findContours(flame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flame_areas   = [cv2.contourArea(c) for c in flame_cnts]
        largest_blob  = max(flame_areas) if flame_areas else 0
        total_blob    = sum(flame_areas)
        concentration = largest_blob / max(total_blob, 1)

        warm_ratio = cv2.countNonZero(
            cv2.inRange(hsv, np.array([5, 80, 180]), np.array([35, 255, 255]))) / (H * H)
        dark_ratio = float((gray < 50).sum()) / (H * H)

        is_candle = (concentration > 0.65 and flame_ratio < 0.08
                     and (warm_ratio > 0.004 or dark_ratio > 0.65))
        if is_candle:
            return {}

        min_flame    = 0.04 if dark_ratio >= 0.60 else 0.13
        large_single = concentration > 0.80 and largest_blob > 8000
        multi_blob   = sum(1 for a in flame_areas if a > 1200) >= 2
        has_flame    = flame_ratio > min_flame and (large_single or multi_blob)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        smoke  = cv2.morphologyEx(
            cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 40, 200])),
            cv2.MORPH_OPEN, kernel)
        smoke_cnts, _ = cv2.findContours(smoke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_smoke   = sum(1 for c in smoke_cnts if cv2.contourArea(c) > 8000)
        smoke_ratio   = cv2.countNonZero(smoke) / (H * H)
        has_smoke     = large_smoke >= 2 and smoke_ratio > 0.08

        if not has_flame and not has_smoke:
            return {}

        if has_flame and has_smoke:
            sub    = "Active Fire with Smoke Plume"
            score  = round(min(60.0 + flame_ratio * 800 + smoke_ratio * 200, 88.0), 1)
            reason = f"Flame ({flame_ratio*100:.1f}%) and smoke ({smoke_ratio*100:.1f}%) detected"
        elif has_flame:
            sub    = "Active Flame Detected"
            score  = round(min(50.0 + flame_ratio * 900, 82.0), 1)
            reason = f"Fire signature covers {flame_ratio*100:.1f}% of frame"
        else:
            sub    = "Smoke / Obscurant Plume"
            score  = round(min(45.0 + smoke_ratio * 300, 70.0), 1)
            reason = f"{large_smoke} large smoke regions covering {smoke_ratio*100:.1f}%"

        return {"Environmental Hazard": {"subcategory": sub, "score": score, "reason": reason}}
    except Exception as e:
        print(f"[content_detector] Fire/smoke detection error: {e}")
        return {}


# =============================================================================
# PUBLIC API
# =============================================================================
def run_content_detection(path: str) -> dict:
    """
    Run all detection layers and return merged results.
    Returns: { category: { subcategory: str, score: float, reason: str } }
    Merge rule: highest score per category wins.
    """
    result: dict = {}
    for detector in (
        _detect_nnet,
        _detect_blood,
        _detect_distress,
        _detect_weapon_in_hand,
        _detect_fire_smoke,
    ):
        for cat, info in detector(path).items():
            if cat not in result or info["score"] > result[cat]["score"]:
                result[cat] = info
    return result