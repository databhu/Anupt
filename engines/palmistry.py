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
MIN_SKIN_TONE_RATIO = 0.12  # deliberately lenient — a coarse filter, not a verdict

# ---------------------------------------------------------------------------
# Closed feature vocabulary — deterministic reference data, not AI output.
# This exists specifically to stop the AI from inventing feature names: the
# vision prompt is built by handing it EXACTLY this list and instructing it
# to only ever name a feature that appears here, and only when it's actually
# visible. "Traditional meaning" here is the general, standard association —
# short and neutral on purpose; the AI's job is to connect a SPECIFIC
# observation on THIS photo to this general tradition, not to originate the
# tradition itself.
# ---------------------------------------------------------------------------

PALM_FEATURES = {
    "Life Line": {
        "category": "line",
        "typical_location": "curves around the base of the thumb, between the thumb and the fleshy "
                            "Mount of Venus",
        "traditional_meaning": "Vitality, physical wellbeing, and major life changes — not literally "
                               "lifespan length, a common misconception.",
    },
    "Head Line": {
        "category": "line",
        "typical_location": "runs roughly horizontally across the middle of the palm, usually starting "
                            "near the Life Line",
        "traditional_meaning": "Thinking style, intellectual approach, and decision-making.",
    },
    "Heart Line": {
        "category": "line",
        "typical_location": "runs horizontally near the top of the palm, just below the fingers",
        "traditional_meaning": "Emotional expression and relationship patterns.",
    },
    "Fate Line": {
        "category": "line",
        "typical_location": "runs vertically up the center of the palm, toward the middle finger — "
                            "not present or clearly visible on every hand",
        "traditional_meaning": "Career direction and the degree external circumstances shape one's path.",
    },
    "Sun/Apollo Line": {
        "category": "line",
        "typical_location": "a short vertical line rising toward the ring finger, when present",
        "traditional_meaning": "Creative recognition, reputation, and a sense of fulfillment.",
    },
    "Mercury Line": {
        "category": "line",
        "typical_location": "runs from near the base of the palm up toward the little finger, when present",
        "traditional_meaning": "Communication style and, in some traditions, business acumen.",
    },
    "Marriage/Relationship Lines": {
        "category": "line",
        "typical_location": "short horizontal lines on the outer edge of the palm, just below the little finger",
        "traditional_meaning": "Significant close relationships — not a literal count of marriages, "
                               "a common misconception.",
    },
    "Girdle of Venus": {
        "category": "line",
        "typical_location": "a curved line above the Heart Line, beneath the middle and ring fingers, "
                            "when present",
        "traditional_meaning": "Emotional sensitivity and intensity.",
    },
    "Mount of Venus": {
        "category": "mount",
        "typical_location": "the fleshy pad at the base of the thumb, encircled by the Life Line",
        "traditional_meaning": "Warmth, vitality, and capacity for connection.",
    },
    "Mount of Jupiter": {
        "category": "mount",
        "typical_location": "at the base of the index finger",
        "traditional_meaning": "Ambition, leadership, and self-confidence.",
    },
    "Mount of Saturn": {
        "category": "mount",
        "typical_location": "at the base of the middle finger",
        "traditional_meaning": "Discipline, responsibility, and introspection.",
    },
    "Mount of Apollo": {
        "category": "mount",
        "typical_location": "at the base of the ring finger",
        "traditional_meaning": "Creativity, self-expression, and desire for recognition.",
    },
    "Mount of Mercury": {
        "category": "mount",
        "typical_location": "at the base of the little finger",
        "traditional_meaning": "Communication, wit, and business sense.",
    },
    "Mount of Mars": {
        "category": "mount",
        "typical_location": "two zones — 'upper' between the Heart Line and Mount of Mercury, "
                            "'lower' between the thumb and Life Line",
        "traditional_meaning": "Courage, resilience, and how one handles conflict.",
    },
    "Mount of Luna": {
        "category": "mount",
        "typical_location": "the lower outer edge of the palm, opposite the thumb",
        "traditional_meaning": "Imagination, intuition, and inner life.",
    },
    "Hand Shape": {
        "category": "shape",
        "typical_location": "the overall proportions of the palm and fingers",
        "traditional_meaning": "The classical Earth/Air/Fire/Water hand-shape system, describing a "
                               "general temperament.",
    },
    "Thumb": {
        "category": "shape",
        "typical_location": "size, flexibility, and the angle it sits at relative to the palm",
        "traditional_meaning": "Willpower and reasoning style.",
    },
    "Fingers": {
        "category": "shape",
        "typical_location": "relative length, spacing, and set of the four fingers",
        "traditional_meaning": "Different traits per finger in most traditions — e.g. a long index "
                               "finger with leadership, a long little finger with communication.",
    },
}

# Marking types that can appear ON a line or mount (reported as a modifier
# on a finding, not as their own top-level feature) — also a closed
# vocabulary, for the same reason.
MARKING_TYPES = ["Island", "Fork", "Break", "Cross", "Star", "Triangle", "Chain", "Grille", "Circle"]

LIFE_AREAS = ["personality", "career", "finance", "relationships", "strengths", "challenges", "life_phases"]


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
    hand_likely = skin_ratio >= MIN_SKIN_TONE_RATIO

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
