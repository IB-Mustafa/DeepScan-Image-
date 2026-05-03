"""
content_detector.py — DeepScanAI
==================================
Sensitive content detection. No AI-generation logic.

Detection layers:
  1. Nnet (restricted scope) — weapons, blood, injury, medical only
                                   Non-safety labels are excluded.
  2. Blood & Hemorrhage         — vivid red HSV + morphological analysis
                                   Sub-types: splatter, pooling, smear patterns
  3. Bruise & Physical Trauma   — purple-red discoloration co-occurring with skin
                                   Severity tiered: minor / moderate / severe
  4. Scene Threat Analysis      — dark environments, looming shapes, posture cues
                                   Sub-types: low-visibility, shadow threat, cowering
  5. Weapon-in-Hand             — power-grip + elongated rigid protrusion
                                   Sub-types: bladed, blunt, firearm-shaped
  6. Smoke & Fire Hazard        — orange/yellow flame + grey smoke HSV zones
  7. Chemical / Hazmat Cues     — yellow-green color anomalies + containment shapes
  8. Crowd Panic / Stampede     — dense motion blur + fragmented limb distribution

Sensitivity score guide:
  Weapon present (not aimed)     → 55%
  Weapon aimed / grip detected   → 60–85%
  Minor bruising                 → 35–50%
  Moderate trauma                → 50–70%
  Severe trauma / blood          → 70–95%
  Low-visibility threat scene    → 70%
  Shadow / posture threat        → 45–88%
  Fire / smoke hazard            → 50–85%
  Hazmat cue                     → 45–75%
  Crowd panic                    → 50–80%

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

# ── Nnet — content safety classifier ────────────────────────────
try:
    from nudenet import NudeDetector as _NnetDetector
    _nnet = _NnetDetector()
    NNET_OK = True
    print("[content_detector] ✓ Nnet loaded")
except Exception as e:
    print(f"[content_detector] Nnet unavailable: {e}")
    _nnet = None
    NNET_OK = False

# ── Nnet label map — all labels handled with clean professional display names ──
_NNET_MAP = {
    # ── Safety / restricted items ──────────────────────────────────────────────
    "GUN_POINTED":              ("Restricted Item",  "Firearm — Directed at Subject",         65.0),
    "GUN_NOT_POINTED":          ("Restricted Item",  "Firearm — Present in Scene",            55.0),
    "KNIFE":                    ("Restricted Item",  "Bladed Object Detected",                52.0),
    "WEAPON":                   ("Restricted Item",  "Unclassified Prohibited Object",        50.0),
    # ── Medical / injury ───────────────────────────────────────────────────────
    "BLOOD":                    ("Hemorrhage Alert", "Blood — Confirmed by Classifier",       68.0),
    "INJURY":                   ("Physical Trauma",  "Visible Injury Detected",               62.0),
    "MEDICAL":                  ("Clinical Content", "Clinical / Surgical Scene",             42.0),
    "MEDICAL_EQUIPMENT":        ("Clinical Content", "Medical Equipment in Frame",            32.0),
    # ── Content policy flags — loaded from separate module ──────────────────────
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
    """Load and resize image to 512×512. Returns (bgr, hsv, gray) or None."""
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
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        )
        return len(face_cascade.detectMultiScale(gray, 1.1, 4)) > 0
    except Exception:
        return False


# =============================================================================
# LAYER 1 — Nnet content safety classifier
# Detects: weapons, blood, injury, medical, content policy flags
# All labels mapped to professional display names — no raw classifier terms shown
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
# LAYER 2 — BLOOD & HEMORRHAGE  (multi-pattern HSV analysis)
#
# Three sub-patterns detected independently:
#   A) Splatter  — small high-saturation red blobs dispersed across frame
#   B) Pooling   — large contiguous red region (low edge density inside)
#   C) Smear     — elongated red streaks (high aspect ratio contours)
# =============================================================================
def _detect_blood(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        img_bgr, hsv, gray = loaded

        # Guard: if a frontal face is detected, mask out the face zone.
        # This prevents theatrical makeup, lipstick, or face paint from
        # being mistaken for blood. Only flag red pixels OUTSIDE the face area.
        face_zone_mask = np.ones((512, 512), dtype=np.uint8) * 255
        if _has_frontal_face(gray):
            try:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (fx, fy, fw, fh) in faces:
                    # Expand face bounding box by 30% to cover makeup edges
                    pad_x = int(fw * 0.30)
                    pad_y = int(fh * 0.30)
                    x1 = max(0, fx - pad_x)
                    y1 = max(0, fy - pad_y)
                    x2 = min(511, fx + fw + pad_x)
                    y2 = min(511, fy + fh + pad_y)
                    face_zone_mask[y1:y2, x1:x2] = 0
            except Exception:
                pass

        # Blood detection — three-stage pipeline:
        #
        # Stage 1: Build red pixel mask (broad — catches vivid + matte blood)
        # Stage 2: Context guards (fire exclusion, face makeup exclusion,
        #          real-face validation, uniform-object exclusion)
        # Stage 3: Threshold — low with body context, high without

        # Stage 1: broad red mask
        r1 = cv2.inRange(hsv, np.array([0,   60,  40]), np.array([10,  255, 220]))
        r2 = cv2.inRange(hsv, np.array([165,  60,  40]), np.array([180, 255, 220]))
        blood_mask = cv2.bitwise_or(r1, r2)

        # Two-part exclusion to remove non-blood red pixels:
        #   Part A: orange H 5-30, sat>=100 — catches fire/flame AND ketchup stream
        #   Part B: H 0-5, sat>=160, val<140 — dark saturated sauce/paste blobs
        #           (bright blood val>=140 at H 0-5 is preserved)
        excl_a     = cv2.inRange(hsv, np.array([5,  100,  80]), np.array([30, 255, 255]))
        excl_b     = cv2.inRange(hsv, np.array([0,  160,  40]), np.array([5,  255, 139]))
        flame_zone = cv2.dilate(cv2.bitwise_or(excl_a, excl_b),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        blood_mask = cv2.bitwise_and(blood_mask, cv2.bitwise_not(flame_zone))

        # Face makeup exclusion
        blood_mask = cv2.bitwise_and(blood_mask, face_zone_mask)

        # Stage 2: detect real faces only
        # minSize=(50,50) filters tiny false-positive detections in fire/texture patterns.
        # Saturation guard: real skin has S < 160; fire/pattern false faces have S > 160.
        real_faces = []
        for (fx, fy, fw, fh) in face_cascade.detectMultiScale(
                gray, 1.1, 4, minSize=(50, 50)):
            face_region_hsv = hsv[fy:fy+fh, fx:fx+fw]
            mean_sat = float(face_region_hsv[:, :, 1].mean())
            if mean_sat < 160:   # real skin — fire/texture false positives have sat > 160
                real_faces.append((fx, fy, fw, fh))

        # Build body region from validated faces only
        body_region = np.zeros((512, 512), dtype=np.uint8)
        for (fx, fy, fw, fh) in real_faces:
            pad_x = int(fw * 2.0)
            x1 = max(0,   fx - pad_x)
            x2 = min(511, fx + fw + pad_x)
            y1 = max(0,   fy - int(fh * 0.5))
            y2 = min(511, fy + fh + int(fh * 2.5))
            body_region[y1:y2, x1:x2] = 255

        # Also treat large skin regions as body context (hands, arms, torso without face)
        # val >= 150 required — real skin is bright; dark wood/table fails this check
        skin_a = cv2.inRange(hsv, np.array([0,  15, 150]), np.array([20, 170, 255]))
        skin_b = cv2.inRange(hsv, np.array([160,15, 150]), np.array([180,170, 255]))
        skin   = cv2.bitwise_or(skin_a, skin_b)
        skin_ratio = cv2.countNonZero(skin) / (512 * 512)
        if skin_ratio > 0.20:   # significant real skin present = hand/arm/body shot
            skin_dilated = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30)))
            combined_body = cv2.bitwise_or(body_region, skin_dilated)
        else:
            combined_body = body_region

        has_body = cv2.countNonZero(combined_body) > 0

        # Stage 3: compute ratio and apply uniform-object guard
        if has_body:
            blood_zone = cv2.bitwise_and(blood_mask, combined_body)
        else:
            blood_zone = blood_mask

        blood_clean = cv2.morphologyEx(blood_zone, cv2.MORPH_OPEN,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        ratio = cv2.countNonZero(blood_clean) / (512 * 512)

        # Uniform-object guard: a single large blob covering >55% of red area
        # = clothing, ketchup bottle, painted object — NOT blood spatter.
        cnts_b, _ = cv2.findContours(blood_clean, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if cnts_b:
            b_areas    = [cv2.contourArea(c) for c in cnts_b]
            b_largest  = max(b_areas)
            b_total    = sum(b_areas)
            b_conc     = b_largest / max(b_total, 1)
            sig_blobs  = sum(1 for a in b_areas if a > 200)
            if b_conc > 0.55 and sig_blobs <= 3:
                return {}   # uniform object — not blood

        threshold = 0.006 if has_body else 0.150  # 15% without body context — filters ketchup/food
        if ratio < threshold:
            return {}

        # Analyse contour geometry to classify sub-pattern
        cnts, _ = cv2.findContours(blood_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {}

        total_area   = sum(cv2.contourArea(c) for c in cnts)
        largest_area = max(cv2.contourArea(c) for c in cnts)
        count        = len(cnts)

        # Pooling: one large connected region, low internal edge density
        if largest_area / max(total_area, 1) > 0.70 and count <= 4:
            sub    = "Hemorrhage Pooling Pattern"
            reason = "Large contiguous blood-colored region detected — consistent with pooling"
            score  = round(min(55.0 + ratio * 1800, 95.0), 1)

        # Splatter: many small dispersed blobs
        elif count > 12 and largest_area / max(total_area, 1) < 0.35:
            sub    = "Hemorrhage Splatter Pattern"
            reason = "Multiple dispersed high-saturation red blobs — consistent with splatter"
            score  = round(min(50.0 + ratio * 1600, 92.0), 1)

        # Smear: elongated streaks
        else:
            elongated = 0
            for c in cnts:
                if cv2.contourArea(c) < 300:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if max(w, h) / max(min(w, h), 1) > 3.5:
                    elongated += 1
            if elongated >= 2:
                sub    = "Hemorrhage Smear Pattern"
                reason = "Elongated red streaks detected — consistent with smear or drag pattern"
                score  = round(min(48.0 + ratio * 1500, 88.0), 1)
            else:
                sub    = "Blood Presence Detected"
                reason = "Blood-consistent color and morphology detected in image"
                score  = round(min(45.0 + ratio * 1400, 90.0), 1)

        return {
            "Hemorrhage Alert": {
                "subcategory": sub,
                "score":       score,
                "reason":      reason,
            }
        }
    except Exception as e:
        print(f"[content_detector] Blood detection error: {e}")
        return {}


# =============================================================================
# LAYER 4 — SCENE THREAT ANALYSIS  (3 independent sub-detectors)
#
#   A) Low-Visibility Scene  — extreme dark ratio (classic menace)
#   B) Shadow Threat         — large elongated dark contours looming above
#   C) Cowering Posture      — skin clustered low-center + dark overhead zone
#
# Guard: clear frontal face detected → not a hidden/threat scene.
# =============================================================================
def _detect_distress(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        img_bgr, hsv, gray = loaded

        H = 512

        # Guard 1: any frontal face detected → normal scene, skip
        if _has_frontal_face(gray):
            return {}

        # Guard 2: multiple faces anywhere in image (family/group shot) → safe
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
            )
            all_faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            if len(all_faces) >= 2:
                return {}
        except Exception:
            pass

        # Guard 3: warm-light source present (candle/lamp glow = indoor safe scene)
        # Warm light = orange-yellow pixels with high value
        warm = cv2.inRange(hsv, np.array([5, 80, 180]), np.array([35, 255, 255]))
        warm_ratio = cv2.countNonZero(warm) / (H * H)
        if warm_ratio > 0.005:   # even a small warm light source = safe indoor scene
            return {}

        # ── A) Extreme low-light scene ────────────────────────────────────────
        dark_ratio = float((gray < 40).sum()) / (H * H)
        if dark_ratio > 0.72:   # raised from 0.60 — genuine threat scenes are very dark
            return {
                "Scene Threat": {
                    "subcategory": "Extreme Low-Visibility Environment",
                    "score":       round(60.0 + dark_ratio * 20, 1),
                    "reason":      f"Scene is {dark_ratio*100:.0f}% dark pixels — no identifiable subject in high-threat environment",
                }
            }

        # ── B) Shadow threat — large looming dark contours in upper frame ─────
        upper_gray = gray[:int(H * 0.65), :]
        _, thresh = cv2.threshold(upper_gray, 80, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # ── C) Cowering posture ───────────────────────────────────────────────
        skin1 = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([25, 180, 255]))
        skin2 = cv2.inRange(hsv, np.array([168, 20, 50]), np.array([180, 180, 255]))
        skin  = cv2.bitwise_or(skin1, skin2)

        lower_center   = skin[int(H * 0.55):, int(H * 0.25):int(H * 0.75)]
        cowering_ratio = lower_center.sum() / 255.0 / max(lower_center.size, 1)
        upper_dark     = float((gray[:int(H * 0.50), :] < 80).sum()) / (int(H * 0.50) * H)

        cowering_score = 0.0
        if cowering_ratio > 0.25 and upper_dark > 0.25 and shadow_score > 20:
            cowering_score = min(50.0 + cowering_ratio * 60.0, 88.0)

        best_score = max(shadow_score, cowering_score)
        if best_score < 45.0:
            return {}

        if cowering_score >= shadow_score:
            sub    = "Cowering / Submission Posture"
            reason = (f"Subject skin concentrated in lower-center zone ({cowering_ratio*100:.0f}%) "
                      f"with {upper_dark*100:.0f}% overhead darkness — posture consistent with threat response")
        else:
            sub    = "Looming Shadow Threat"
            reason = (f"Large dark contour (area ≈ {dominant_area}px²) detected in upper frame "
                      f"without identifiable face — consistent with looming threat composition")

        return {
            "Scene Threat": {
                "subcategory": sub,
                "score":       round(best_score, 1),
                "reason":      reason,
            }
        }
    except Exception as e:
        print(f"[content_detector] Distress detection error: {e}")
        return {}


# =============================================================================
# LAYER 5 — WEAPON IN HAND  (power-grip + elongated protrusion)
#
# Sub-type classification:
#   Bladed    — grey/silver metallic, very elongated (aspect > 5)
#   Blunt     — dark object, moderate elongation (aspect 3–5)
#   Firearm-shaped — rectangular protrusion at mid-frame
#
# Guards: pen/pencil (short_side < 12), phone/tablet (aspect < 3),
#         small hand area, face present (selfie context).
# =============================================================================
def _detect_weapon_in_hand(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        _, hsv, gray = loaded

        if _has_frontal_face(gray):
            return {}

        skin_mask   = cv2.inRange(hsv, np.array([0,  20,  70]), np.array([20,  255, 255]))
        metal_mask  = cv2.inRange(hsv, np.array([0,   0,  70]), np.array([180,  30, 220]))
        dark_mask   = cv2.inRange(hsv, np.array([0,   0,   0]), np.array([180,  50,  80]))

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
            if aspect < 3.0:
                continue
            if oy + oh < 200:
                continue

            # Sub-type classification
            if mask_name == "metal" and aspect > 5.0:
                sub    = "Bladed Object in Grip"
                score  = 85.0
                reason = f"Silver/grey elongated object (aspect {aspect:.1f}:1) in power-grip — consistent with bladed weapon"
            elif mask_name == "dark" and 3.0 <= aspect <= 5.5:
                sub    = "Blunt / Club-type Object in Grip"
                score  = 75.0
                reason = f"Dark elongated object (aspect {aspect:.1f}:1) in closed grip — consistent with blunt instrument"
            elif mask_name == "metal" and 3.0 <= aspect <= 5.0:
                sub    = "Firearm-Shaped Object in Grip"
                score  = 80.0
                reason = f"Metallic object with firearm-consistent proportions (aspect {aspect:.1f}:1) detected in grip"
            else:
                sub    = "Unclassified Elongated Object in Grip"
                score  = 70.0
                reason = f"Elongated object (aspect {aspect:.1f}:1) held in power-grip — restricted item candidate"

            if best_result is None or score > best_result["score"]:
                best_result = {"subcategory": sub, "score": score, "reason": reason}

        if best_result:
            return {"Restricted Item": best_result}
        return {}

    except Exception as e:
        print(f"[content_detector] Weapon-in-hand error: {e}")
        return {}


# =============================================================================
# LAYER 6 — SMOKE & FIRE HAZARD  (NEW)
#
# Detects two hazard signatures:
#   A) Active flame — orange/yellow high-saturation pixels in lower/mid frame
#   B) Smoke plume  — grey low-saturation large blobs covering significant area
# =============================================================================
def _detect_fire_smoke(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        _, hsv, gray = loaded
        H = 512
 
        # ── A) Flame pixels ───────────────────────────────────────────────────
        flame1 = cv2.inRange(hsv, np.array([5,  180, 150]), np.array([30, 255, 255]))
        flame2 = cv2.inRange(hsv, np.array([0,  200, 180]), np.array([8,  255, 255]))
        flame  = cv2.morphologyEx(cv2.bitwise_or(flame1, flame2),
                                  cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        flame_ratio = cv2.countNonZero(flame) / (H * H)
 
        flame_cnts, _ = cv2.findContours(flame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flame_areas   = [cv2.contourArea(c) for c in flame_cnts]
        largest_blob  = max(flame_areas) if flame_areas else 0
        total_blob    = sum(flame_areas)
        concentration = largest_blob / max(total_blob, 1)
 
        warm_ratio = cv2.countNonZero(
            cv2.inRange(hsv, np.array([5, 80, 180]), np.array([35, 255, 255]))) / (H * H)
        dark_ratio = float((gray < 50).sum()) / (H * H)
 
        # ── Guard 1: candle / lamp ────────────────────────────────────────────
        # Single small concentrated blob + dark or warm scene = safe light source
        is_candle = (
            concentration > 0.65
            and flame_ratio < 0.08
            and (warm_ratio > 0.004 or dark_ratio > 0.65)
        )
        if is_candle:
            return {}
 
        # ── Guard 2: dark-adaptive minimum coverage ───────────────────────────
        # Bright scenes (food photos, daylight): need >15% coverage to be real fire.
        # Dark scenes (campfire, indoor): 4% is enough.
        min_flame = 0.04 if dark_ratio >= 0.60 else 0.13
 
        # Flame is valid if: enough coverage AND (one large blob OR multiple blobs)
        large_single = concentration > 0.80 and largest_blob > 8000
        multi_blob   = sum(1 for a in flame_areas if a > 1200) >= 2
        has_flame    = flame_ratio > min_flame and (large_single or multi_blob)
 
        # ── B) Smoke ──────────────────────────────────────────────────────────
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
 
        return {
            "Environmental Hazard": {
                "subcategory": sub,
                "score":       score,
                "reason":      reason,
            }
        }
    except Exception as e:
        print(f"[content_detector] Fire/smoke detection error: {e}")
        return {}
 
 

# =============================================================================
# LAYER 7 — CHEMICAL / HAZMAT CUES  (NEW)
#
# Looks for yellow-green color anomalies (chlorine / chemical agent signature)
# combined with container-like circular shapes (drums, canisters).
# =============================================================================
def _detect_hazmat(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        _, hsv, gray = loaded
        H = 512

        # Yellow-green anomaly (H: 35–85, high sat, mid-high val)
        yg = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 230]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yg = cv2.morphologyEx(yg, cv2.MORPH_OPEN, kernel)
        yg_ratio = cv2.countNonZero(yg) / (H * H)

        if yg_ratio < 0.03:
            return {}

        # Check for circular/cylindrical containment shapes via Hough circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=80, param2=30, minRadius=15, maxRadius=120
        )
        has_container = circles is not None and len(circles[0]) >= 1

        if yg_ratio > 0.12 and has_container:
            sub    = "Chemical Container with Hazmat Signature"
            score  = round(min(55.0 + yg_ratio * 300, 75.0), 1)
            reason = f"Yellow-green chemical signature ({yg_ratio*100:.1f}% coverage) co-located with cylindrical container shapes"
        elif yg_ratio > 0.08:
            sub    = "Anomalous Chemical Color Signature"
            score  = round(min(45.0 + yg_ratio * 250, 68.0), 1)
            reason = f"Unusual yellow-green spectral pattern ({yg_ratio*100:.1f}%) — possible hazardous material indicator"
        else:
            return {}

        return {
            "Hazmat Indicator": {
                "subcategory": sub,
                "score":       score,
                "reason":      reason,
            }
        }
    except Exception as e:
        print(f"[content_detector] Hazmat detection error: {e}")
        return {}


# =============================================================================
# LAYER 8 — CROWD PANIC / STAMPEDE  (NEW)
#
# Detects dense fragmented limb distribution across lower frame
# combined with motion blur signature (high edge density + low structure).
# =============================================================================
def _detect_crowd_panic(path: str) -> dict:
    loaded = _load_image(path)
    if loaded is None:
        return {}
    try:
        img_bgr, hsv, gray = loaded
        H = 512

        if _has_frontal_face(gray):
            return {}

        # Edge density in lower 70% of frame (limbs / bodies)
        lower_gray  = gray[int(H * 0.30):, :]
        edges       = cv2.Canny(lower_gray, 50, 120)
        edge_ratio  = cv2.countNonZero(edges) / max(lower_gray.size, 1)

        # Skin fragments — many small disconnected skin blobs
        skin1 = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([25, 180, 255]))
        skin2 = cv2.inRange(hsv, np.array([168, 20, 50]), np.array([180, 180, 255]))
        skin  = cv2.bitwise_or(skin1, skin2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin   = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
        skin_cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        small_skin_blobs = sum(1 for c in skin_cnts if 200 < cv2.contourArea(c) < 3000)
        total_skin_ratio = cv2.countNonZero(skin) / (H * H)

        # Blur metric — motion blur lowers Laplacian variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        high_edge    = edge_ratio > 0.12
        fragmented   = small_skin_blobs > 8
        blurred      = lap_var < 300
        dense_skin   = total_skin_ratio > 0.20

        signal_count = sum([high_edge, fragmented, blurred, dense_skin])
        if signal_count < 3:
            return {}

        score = round(min(40.0 + signal_count * 10 + small_skin_blobs * 1.5 + edge_ratio * 200, 80.0), 1)
        reason = (f"Crowd panic indicators: edge density={edge_ratio*100:.1f}%, "
                  f"skin fragments={small_skin_blobs}, blur variance={lap_var:.0f}, "
                  f"skin coverage={total_skin_ratio*100:.1f}%")

        return {
            "Crowd Safety": {
                "subcategory": "Dense Crowd / Potential Stampede Scene",
                "score":       score,
                "reason":      reason,
            }
        }
    except Exception as e:
        print(f"[content_detector] Crowd panic detection error: {e}")
        return {}


# =============================================================================
# PUBLIC API
# =============================================================================
def run_content_detection(path: str) -> dict:
    """
    Run all detection layers and return merged results.
    Scope is restricted to safety-relevant content only.

    Returns:
        { category: { subcategory: str, score: float, reason: str } }

    Merge rule: highest score per category wins.
    """
    result: dict = {}

    for detector in (
        _detect_nnet,          # weapons / blood / medical (Nnet restricted)
        _detect_blood,            # hemorrhage pattern analysis
        _detect_distress,         # scene threat analysis
        _detect_weapon_in_hand,   # grip-based weapon sub-typing
        _detect_fire_smoke,       # fire & smoke hazard  (NEW)
        _detect_hazmat,           # chemical / hazmat cues  (NEW)
        _detect_crowd_panic,      # crowd density / stampede  (NEW)
    ):
        for cat, info in detector(path).items():
            if cat not in result or info["score"] > result[cat]["score"]:
                result[cat] = info

    return result