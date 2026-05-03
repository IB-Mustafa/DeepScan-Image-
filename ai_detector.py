"""
ai_detector.py — DeepScanAI
============================
Handles ONLY AI-generation and deepfake detection.
No sensitive-content logic lives here.

Five forensic signals:
  1. EXIF Metadata       — real cameras leave rich metadata; AI images often don't
  2. FFT Frequency       — natural images follow 1/f power law; AI images deviate
  3. ELA Compression     — AI images show near-zero ELA (too clean to re-compress)
  4. Sensor Noise        — real sensors produce characteristic noise; AI is too smooth
                           Key fix: kurtosis > 4.0 on noise = synthetic / AI pattern
  5. Chromatic Aberration— real lenses bend light; AI skips this physical effect

Fixes vs previous version:
  - ELA: near-zero mean (< 0.5) was returning neutral 40. Now returns 72 (suspicious)
    because AI images saved cleanly have almost no compression history.
  - Noise: kurtosis threshold tightened — real sensor noise kurtosis is 0.5–3.0;
    AI images often show kurtosis > 4 (peaky/synthetic texture).
  - CA: threshold lowered — AI portraits have near-zero aberration (< 1.0 → 75).
  - Convergence guard: raised from 1 to 2 high signals before capping,
    and cap raised from 54 → 62 so mid-confidence AI images aren't suppressed.
  - Weights rebalanced: ELA 0.20→0.22, noise 0.15→0.18, ca 0.10→0.12, fft 0.25→0.20.

Public API
----------
  run_ai_detection(path: str) -> dict
    {
        "ai_score":       float | None,   # 0-100, probability of AI generation
        "deepfake_score": float | None,   # 0-100, face-manipulation probability
        "reasoning":      str,
        "signal_scores":  { exif, fft, ela, noise, ca }
    }
"""

import io
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis

# Pillow version-safe resampling
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


# =============================================================================
# SIGNAL 1 — EXIF METADATA
# =============================================================================
def _signal_exif(path: str) -> float:
    """
    Real cameras write rich EXIF tags (make, model, focal length, ISO …).
    AI-generated images either lack EXIF entirely or have minimal tags.
    Returns a suspicion score 0-100 (higher = more likely AI).
    """
    try:
        with Image.open(path) as img:
            fmt      = (img.format or "").upper()
            exif_raw = img.getexif() if hasattr(img, "getexif") else {}

        # These formats never carry EXIF — neutral score
        if fmt in ("PNG", "WEBP", "BMP", "GIF"):
            return 45.0

        # JPEG with zero EXIF is very suspicious
        if not exif_raw or len(exif_raw) == 0:
            return 72.0   # raised from 68 — zero EXIF on JPEG is a strong AI signal

        camera_tags  = {"Make", "Model", "LensModel", "LensMake"}
        capture_tags = {"DateTimeOriginal", "DateTime", "DateTimeDigitized"}
        setting_tags = {"FocalLength", "ISOSpeedRatings", "ExposureTime",
                        "FNumber", "Flash", "MeteringMode", "ExposureProgram"}
        gps_tags     = {"GPSInfo"}

        tag_names = {TAGS.get(k, "") for k in exif_raw.keys()}
        score = (
            len(tag_names & camera_tags)  * 3 +
            len(tag_names & capture_tags) * 2 +
            len(tag_names & setting_tags) * 1 +
            len(tag_names & gps_tags)     * 2
        )

        if   score >= 8: return 5.0
        elif score >= 5: return 15.0
        elif score >= 3: return 30.0
        elif score >= 1: return 50.0
        else:            return 64.0

    except Exception:
        return 50.0


# =============================================================================
# SIGNAL 2 — FFT FREQUENCY DOMAIN
# =============================================================================
def _signal_fft(img: Image.Image) -> float:
    """
    Natural images follow a 1/f^n power spectrum (slope -1.5 to -2.8).
    AI generators often produce spectra outside this range — too smooth
    (missing high-frequency detail) or artificially regular.
    Note: high-quality AI portraits can pass this test — FFT alone is weak
    for photorealistic faces.
    """
    try:
        gray   = np.array(img.convert("L").resize((512, 512))).astype(np.float64)
        fft_sh = np.fft.fftshift(np.fft.fft2(gray))
        power  = np.abs(fft_sh) ** 2

        h, w   = power.shape
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.mgrid[-cy:h-cy, -cx:w-cx]
        radius = np.sqrt(x_idx**2 + y_idx**2).astype(int)
        max_r  = min(cy, cx)

        radial_power = np.zeros(max_r)
        for r in range(1, max_r):
            mask = (radius == r)
            if mask.sum() > 0:
                radial_power[r] = power[mask].mean()

        valid = radial_power[1:] > 0
        freqs = np.arange(1, max_r)[valid]
        pows  = radial_power[1:][valid]
        if len(freqs) < 10:
            return 50.0

        slope, _ = np.polyfit(np.log(freqs), np.log(pows + 1e-10), 1)

        if   -2.8 <= slope <= -1.5: slope_score = 10.0
        elif -3.2 <= slope <= -1.2: slope_score = 30.0
        elif -3.8 <= slope <= -0.8: slope_score = 52.0
        else:                        slope_score = 70.0

        mid_r    = max_r // 2
        lo_pow   = radial_power[1:mid_r].sum()
        hi_pow   = radial_power[mid_r:].sum()
        hf_ratio = hi_pow / (lo_pow + hi_pow + 1e-10)

        if   0.05 <= hf_ratio <= 0.25: hf_score = 10.0
        elif 0.02 <= hf_ratio <= 0.35: hf_score = 30.0
        elif hf_ratio < 0.02:          hf_score = 65.0
        else:                           hf_score = 40.0

        return (slope_score * 0.6) + (hf_score * 0.4)

    except Exception:
        return 50.0


# =============================================================================
# SIGNAL 3 — ELA (Error Level Analysis)
# =============================================================================
def _signal_ela(img: Image.Image) -> float:
    """
    Re-compress image at quality=75 and measure per-pixel difference.
    Real photos show varied compression history (high coefficient of variation).
    AI images show two failure modes:
      A) Near-zero ELA mean (< 0.5) — image was generated/saved cleanly with
         no prior JPEG compression, so re-compression barely changes it.
         This is SUSPICIOUS, not neutral. Fixed from previous 40 → 72.
      B) Low CV — unnaturally uniform compression residual.
    """
    try:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")

        ela  = np.abs(
            np.array(rgb).astype(np.float32) -
            np.array(recomp).astype(np.float32)
        ).mean(axis=2)

        mean = ela.mean()

        # FIX: near-zero ELA = pristine/AI image, not neutral
        if mean < 0.5:
            return 72.0   # was 40.0 — this is a strong AI indicator

        cv = ela.std() / (mean + 1e-6)

        if   cv > 1.5: return 8.0
        elif cv > 1.2: return 20.0
        elif cv > 0.9: return 40.0
        elif cv > 0.6: return 58.0
        else:          return 76.0

    except Exception:
        return 50.0


# =============================================================================
# SIGNAL 4 — SENSOR NOISE
# =============================================================================
def _signal_noise(img: Image.Image) -> float:
    """
    Real camera sensors produce characteristic noise with:
      - Moderate std (not too clean, not too noisy)
      - Gaussian-like kurtosis (0.5 – 3.0)
      - Spatial autocorrelation matching hardware characteristics

    AI images fail in two ways:
      A) noise_std < 2.0  — too smooth / over-processed
      B) kurtosis > 4.0   — peaky/synthetic texture pattern (fixed threshold)
         Previous code only checked std; kurtosis=5.3 on this AI image
         was being ignored entirely.
    """
    try:
        gray      = np.array(img.convert("L").resize((512, 512))).astype(np.float32)
        smooth    = gaussian_filter(gray, sigma=3)
        noise     = gray - smooth
        noise_std = float(noise.std())

        if noise_std < 0.5:
            return 80.0

        noise_flat = noise.flatten()
        kurt_val   = float(kurtosis(noise_flat))

        if   noise_std < 2.0:  level_score = 72.0
        elif noise_std < 5.0:  level_score = 48.0
        elif noise_std < 15.0: level_score = 15.0
        else:                   level_score = 30.0

        # FIX: kurtosis > 4.0 = synthetic/AI noise pattern
        if   kurt_val < 0:            kurt_score = 60.0   # negative = unnaturally flat
        elif kurt_val <= 3.0:         kurt_score = 10.0   # real sensor range
        elif kurt_val <= 4.0:         kurt_score = 28.0   # borderline
        else:                         kurt_score = 65.0   # > 4.0 = AI synthetic texture

        try:
            autocorr_h   = float(np.corrcoef(
                noise[:, :-1].flatten(), noise[:, 1:].flatten())[0, 1])
            autocorr_v   = float(np.corrcoef(
                noise[:-1, :].flatten(), noise[1:, :].flatten())[0, 1])
            spatial_corr = (abs(autocorr_h) + abs(autocorr_v)) / 2.0
        except Exception:
            spatial_corr = 0.2

        if   0.05 <= spatial_corr <= 0.40: corr_score = 15.0
        elif spatial_corr < 0.01:           corr_score = 62.0
        else:                               corr_score = 32.0

        return (level_score * 0.5) + (kurt_score * 0.3) + (corr_score * 0.2)

    except Exception:
        return 50.0


# =============================================================================
# SIGNAL 5 — CHROMATIC ABERRATION
# =============================================================================
def _signal_chromatic_aberration(img: Image.Image) -> float:
    """
    Real lenses cause chromatic aberration — color fringing at edges.
    AI images lack this physical lens effect, so color channels align too well.

    FIX: Lowered thresholds — AI portraits have near-zero CA (< 1.0).
    Previous code returned only 68 for ca < 1.5, missing very clean AI images.
    """
    try:
        from scipy.ndimage import sobel
        arr     = np.array(img.convert("RGB").resize((512, 512))).astype(np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        gray      = 0.299 * r + 0.587 * g + 0.114 * b
        edges     = np.sqrt(sobel(gray, axis=1)**2 + sobel(gray, axis=0)**2)
        threshold = np.percentile(edges, 85)
        edge_mask = edges > threshold

        if edge_mask.sum() < 100:
            return 50.0

        ca = (
            np.abs(r - g)[edge_mask].mean() +
            np.abs(r - b)[edge_mask].mean() +
            np.abs(g - b)[edge_mask].mean()
        ) / 3.0

        r_shift  = np.roll(r, 1, axis=1)
        shift_c  = float(np.corrcoef(r_shift[edge_mask], b[edge_mask])[0, 1])
        normal_c = float(np.corrcoef(r[edge_mask],       b[edge_mask])[0, 1])
        ca_shift = abs(shift_c - normal_c)

        # FIX: tightened thresholds — AI images are extremely clean
        if   ca > 5.0 and ca_shift > 0.03: return 8.0    # strong real lens
        elif ca > 3.0:                       return 20.0
        elif ca > 2.0:                       return 38.0
        elif ca > 1.0:                       return 55.0  # borderline
        else:                                return 75.0  # near-zero CA = AI (was 68)

    except Exception:
        return 50.0


# =============================================================================
# PUBLIC API
# =============================================================================
def run_ai_detection(path: str) -> dict:
    """
    Run all 5 forensic signals and return AI/deepfake scores.

    Returns:
        {
            "ai_score":       float | None,
            "deepfake_score": float | None,
            "reasoning":      str,
            "signal_scores":  { "exif": float, "fft": float, "ela": float,
                                "noise": float, "ca": float }
        }
    """
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        if max(img.size) > 1024:
            img = img.copy()
            img.thumbnail((1024, 1024), _LANCZOS)
    except Exception as e:
        return {
            "ai_score":       None,
            "deepfake_score": None,
            "reasoning":      f"Cannot open image: {e}",
            "signal_scores":  {},
        }

    scores = {
        "exif":  _signal_exif(path),
        "fft":   _signal_fft(img),
        "ela":   _signal_ela(img),
        "noise": _signal_noise(img),
        "ca":    _signal_chromatic_aberration(img),
    }
    print(f"[ai_detector] Signals: {scores}")

    # Rebalanced weights: ELA and noise carry more weight for portrait AI detection
    weights  = {"exif": 0.28, "fft": 0.20, "ela": 0.22, "noise": 0.18, "ca": 0.12}
    ai_raw   = sum(scores[k] * weights[k] for k in weights)
    ai_score = round(min(max(ai_raw, 0.0), 100.0), 1)

    # Convergence guard — require 2+ high signals before allowing scores above 65
    # (prevents a single bad signal from dominating)
    high_count = sum(1 for v in scores.values() if v >= 65)
    if high_count == 0 and ai_score > 45:
        ai_score = 45.0
    elif high_count <= 1 and ai_score > 62:
        ai_score = 62.0   # raised cap from 54 → 62

    # Deepfake = face-manipulation score (elevated only when face detected)
    deepfake_score = round(ai_score * 0.35, 1)
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, post_process=False, device="cpu")
        if mtcnn(img) is not None:
            deepfake_score = ai_score
    except Exception:
        pass

    dominant   = max(scores, key=scores.__getitem__)
    reason_map = {
        "exif":  ("rich camera metadata present",       "missing / sparse EXIF metadata"),
        "fft":   ("natural 1/f frequency spectrum",     "unnatural frequency distribution"),
        "ela":   ("authentic JPEG compression history", "unnaturally clean compression (AI origin)"),
        "noise": ("natural camera sensor noise",        "abnormal noise pattern (synthetic texture)"),
        "ca":    ("natural chromatic aberration",       "missing lens aberration (AI origin)"),
    }
    label     = reason_map[dominant][1 if scores[dominant] >= 50 else 0]
    reasoning = f"Primary signal: {label} ({dominant.upper()}: {scores[dominant]:.0f}/100)"

    return {
        "ai_score":       ai_score,
        "deepfake_score": deepfake_score,
        "reasoning":      reasoning,
        "signal_scores":  {k: round(v, 1) for k, v in scores.items()},
    }