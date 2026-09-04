"""
ANUPT — Numerology theme scoring (interpretation layer, not calculation).

Deliberately a separate module from engines/numerology.py: that file does
arithmetic on a name/date and nothing else; this file interprets those
already-computed numbers into theme scores. Neither file talks to the AI
or does any UI rendering — see the module docstring in numerology.py for
the full layering rationale.

Every score here is a weighted sum of documented number-trait associations
(NUMBER_THEME_AFFINITY below) applied to numbers engines/numerology.py
already calculated. There is no randomness anywhere in this file — the
same profile always produces the same scores, and every score carries the
specific numbers that produced it, so "why is my Career score 72?" always
has a traceable, arithmetic answer rather than a vibe.
"""

from engines import numerology as num

THEMES = ["career", "finance", "relationships", "leadership", "creativity"]

# How much each core number contributes to a theme score. Life Path carries
# the most weight — most numerology traditions treat it as the primary
# indicator of a person's overall direction; Attitude/Personality carry the
# least, since they're generally read as surface-level/first-impression
# indicators rather than core drivers. Weights sum to 1.0.
NUMBER_WEIGHT_IN_SCORE = {
    "life_path": 0.30,
    "destiny": 0.25,
    "soul_urge": 0.15,
    "personality": 0.10,
    "maturity": 0.10,
    "attitude": 0.10,
}

# Documented association of each number's traditional meaning with each
# theme (0.0-1.0). Hand-authored directly from the same trait words already
# shown in numerology.NUMBER_MEANINGS, so a score's "reason" and the
# number's displayed meaning always tell a consistent story rather than
# two disconnected sets of claims about the same number.
NUMBER_THEME_AFFINITY = {
    1:  {"career": 0.60, "finance": 0.40, "relationships": 0.25, "leadership": 0.90, "creativity": 0.50},
    2:  {"career": 0.30, "finance": 0.25, "relationships": 0.90, "leadership": 0.20, "creativity": 0.30},
    3:  {"career": 0.35, "finance": 0.20, "relationships": 0.45, "leadership": 0.30, "creativity": 0.95},
    4:  {"career": 0.80, "finance": 0.60, "relationships": 0.35, "leadership": 0.40, "creativity": 0.20},
    5:  {"career": 0.40, "finance": 0.30, "relationships": 0.35, "leadership": 0.35, "creativity": 0.55},
    6:  {"career": 0.40, "finance": 0.35, "relationships": 0.85, "leadership": 0.30, "creativity": 0.35},
    7:  {"career": 0.30, "finance": 0.25, "relationships": 0.30, "leadership": 0.20, "creativity": 0.45},
    8:  {"career": 0.80, "finance": 0.95, "relationships": 0.30, "leadership": 0.85, "creativity": 0.25},
    9:  {"career": 0.35, "finance": 0.20, "relationships": 0.60, "leadership": 0.40, "creativity": 0.50},
    11: {"career": 0.45, "finance": 0.30, "relationships": 0.50, "leadership": 0.60, "creativity": 0.70},
    22: {"career": 0.90, "finance": 0.70, "relationships": 0.40, "leadership": 0.90, "creativity": 0.55},
    33: {"career": 0.35, "finance": 0.20, "relationships": 0.90, "leadership": 0.50, "creativity": 0.45},
}

BAND_THRESHOLDS = [(75, "Strong"), (55, "Good"), (35, "Developing"), (0, "Growth Area")]

_NUMBER_LABELS = {
    "life_path": "Life Path", "destiny": "Destiny", "soul_urge": "Soul Urge",
    "personality": "Personality", "maturity": "Maturity", "attitude": "Attitude",
}


def _band(score: int) -> str:
    for threshold, label in BAND_THRESHOLDS:
        if score >= threshold:
            return label
    return "Growth Area"


def _first_trait_word(value: int) -> str:
    """'Power, ambition, material mastery, authority.' -> 'power' — used to
    build a short, readable reason without hand-writing one per number×theme
    combination (which would be 12 numbers × 5 themes = 60 phrases to keep
    in sync by hand)."""
    meaning = num.NUMBER_MEANINGS.get(value, "")
    first = meaning.split(",")[0].strip().rstrip(".")
    # Master Number meanings start with "Master Number — <trait>..."
    if "—" in first:
        first = first.split("—", 1)[1].strip()
    return first.lower() or "its traditional traits"


def theme_scores(profile: dict) -> dict:
    """`profile` is engines.numerology.full_profile()'s output. Returns, for
    each theme in THEMES, a 0-100 score, a plain-language band, the specific
    numbers that drove it, and a one-line reason built from those numbers —
    entirely arithmetic and lookup, nothing generated or random."""
    results = {}
    for theme in THEMES:
        weighted_sum = 0.0
        weight_total = 0.0
        contributions = []
        for number_key, weight in NUMBER_WEIGHT_IN_SCORE.items():
            entry = profile.get(number_key)
            if not entry:
                continue
            value = entry["value"]
            affinity = NUMBER_THEME_AFFINITY.get(value, {}).get(theme, 0.0)
            weighted_sum += weight * affinity
            weight_total += weight
            contributions.append((number_key, value, weight * affinity))

        score = round((weighted_sum / weight_total) * 100) if weight_total else 0
        contributions.sort(key=lambda c: c[2], reverse=True)
        top = [c for c in contributions if c[2] > 0][:2]

        if top:
            phrases = [f"{_NUMBER_LABELS[k]} {v} ({_first_trait_word(v)})" for k, v, _ in top]
            reason = "Driven mainly by your " + " and ".join(phrases) + "."
        else:
            reason = "None of your core numbers show a strong pull toward this area either way."

        results[theme] = {
            "score": score,
            "band": _band(score),
            "top_contributors": [{"number_type": k, "value": v} for k, v, _ in top],
            "reason": reason,
        }
    return results
