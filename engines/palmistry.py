"""
ANUPT — Palmistry Engine

Honest design note: true deterministic palm-line detection requires a
trained hand-landmark CV model (the project plan itself defers this to
V2, pending properly licensed training data). Building an untested
"detector" here would silently fabricate results — exactly what the
plan says to avoid.

So this module does two real, deterministic things:
  1. Image quality gate (blur / brightness / resolution) — pass/fail,
     no AI involved.
  2. A conservative skin-tone hand-presence heuristic — a real signal,
     but explicitly scored as a heuristic, not a landmark detector.

If both checks pass, actual line/mount reading is delegated to the AI
vision model (Gemini) — labeled in the UI as "AI-assisted reading,"
never presented as a deterministic measurement. If checks fail, the
module returns a retake request instead of a reading.
"""

from PIL import Image
import numpy as np

MIN_RESOLUTION = 400  # shortest side, px
BLUR_VARIANCE_THRESHOLD = 60.0   # Laplacian variance; lower = blurrier
DARK_THRESHOLD = 40
BRIGHT_THRESHOLD = 235


def _laplacian_variance(gray: np.ndarray) -> float:
    """Simple discrete Laplacian for blur detection (no OpenCV dependency)."""
    kernel_result = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(np.var(kernel_result))


def _skin_presence_ratio(rgb: np.ndarray) -> float:
    """
    Conservative heuristic skin-tone mask in YCbCr space. Returns the
    fraction of pixels falling in a broad human-skin-tone band. This is
    a real but approximate signal, not a hand detector — it cannot
    distinguish a hand from other skin-toned objects.
    """
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    mask = (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173) & (y > 40)
    return float(np.mean(mask))


def assess_image(pil_image: Image.Image) -> dict:
    """
    Deterministic quality + heuristic hand-presence gate.
    Returns pass/fail plus the measurements behind the decision.
    """
    img = pil_image.convert("RGB")
    w, h = img.size
    rgb = np.array(img)
    gray = np.array(img.convert("L"), dtype=float)

    resolution_ok = min(w, h) >= MIN_RESOLUTION
    blur_score = _laplacian_variance(gray)
    blur_ok = blur_score >= BLUR_VARIANCE_THRESHOLD
    mean_brightness = float(np.mean(gray))
    brightness_ok = DARK_THRESHOLD <= mean_brightness <= BRIGHT_THRESHOLD
    skin_ratio = _skin_presence_ratio(rgb)
    # Broad, deliberately lenient band — this heuristic is a coarse filter,
    # not a verdict. Too little skin-tone content likely means no hand at all.
    hand_likely = skin_ratio >= 0.12

    passed = resolution_ok and blur_ok and brightness_ok and hand_likely

    issues = []
    if not resolution_ok:
        issues.append(f"Image resolution too low (shortest side {min(w,h)}px, need ≥{MIN_RESOLUTION}px).")
    if not blur_ok:
        issues.append("Image appears too blurry for line detail.")
    if not brightness_ok:
        issues.append("Lighting is too dark or too overexposed.")
    if not hand_likely:
        issues.append("Couldn't confidently detect a palm in frame — retake with the palm filling the frame.")

    return {
        "passed": passed,
        "resolution": f"{w}x{h}",
        "blur_score": round(blur_score, 1),
        "brightness": round(mean_brightness, 1),
        "skin_tone_ratio": round(skin_ratio, 3),
        "issues": issues,
    }
