"""
ANUPT — Astrology theme scoring (interpretation layer, not calculation).

Separate from engines/astrology.py for the same reason numerology_scoring.py
is separate from numerology.py: that file does astronomical/astrological
calculation and nothing else; this file interprets those already-computed
placements into theme scores. No AI, no randomness — every score is a
documented function of which planets occupy which houses and how
dignified they are, and every score carries the specific placements that
produced it.

Scope note: this uses the classical Vedic significator (karaka) system and
whole-sign houses (engines.astrology.compute_chart()'s output), not the
Western chart — a theme score built from two different house/zodiac
systems at once would conflate two traditions' logic into one number,
which this app is explicit about not doing (see astrology.py's module
docstring on keeping Western and Vedic separate).
"""

from engines import astrology as astro

THEMES = ["career", "finance", "relationships", "communication", "leadership",
          "family", "education", "health", "spirituality", "personal_growth"]

# Which houses traditionally signify each theme, and which planet(s) are
# that theme's classical karaka (significator) regardless of where they sit.
#
# Note on "health": the 6th house is the classical house of disease/
# affliction, where traditional readings treat a WEAK 6th house as the
# favorable sign (fewer health struggles) — the opposite polarity from
# every other theme here, where a strong, well-dignified placement is the
# favorable sign. Mixing that inverted logic into the same scoring formula
# as everything else would silently misrepresent what a high or low score
# means. To keep one consistent "higher score = more favorable" rule across
# every theme, "health" here uses the 1st house (vitality/the physical
# body) and 8th house (deep vitality/resilience) instead — a defensible
# proxy, not a substitute for the 6th house's own, differently-structured
# traditional reading.
THEME_HOUSES = {
    "career": [10, 6], "finance": [2, 11], "relationships": [7, 5],
    "communication": [3], "leadership": [1, 10],
    "family": [4, 2], "education": [4, 5], "health": [1, 8],
    "spirituality": [9, 12], "personal_growth": [1, 9],
}
THEME_KARAKAS = {
    "career": ["Saturn", "Sun"], "finance": ["Jupiter", "Venus"],
    "relationships": ["Venus", "Moon"], "communication": ["Mercury"],
    "leadership": ["Sun", "Mars"],
    "family": ["Moon", "Jupiter"], "education": ["Mercury", "Jupiter"],
    "health": ["Sun", "Mars"], "spirituality": ["Jupiter", "Ketu"],
    "personal_growth": ["Sun", "Jupiter"],
}

# Points added/subtracted per dignity state. House-occupancy counts at full
# weight (a planet actually sitting in the theme's house matters most);
# a karaka's own dignity counts at half weight (it signifies the theme
# wherever it sits, but less directly than actually occupying it).
DIGNITY_POINTS = {"Exalted": 12, "Own Sign": 8, "Moolatrikona": 8, "Neutral": 0, "Debilitated": -12}
KARAKA_WEIGHT = 0.5
BASE_SCORE = 50


def theme_scores(chart: dict) -> dict:
    """`chart` is engines.astrology.compute_chart()'s output. Returns, for
    each theme, a 0-100 score and the specific planets/houses/dignities
    that drove it — arithmetic and lookup only, nothing generated."""
    results = {}
    for theme in THEMES:
        score = BASE_SCORE
        contributions = []  # (description, points) for the reason text

        for house in THEME_HOUSES[theme]:
            for name, data in chart["planets"].items():
                if data["house"] != house:
                    continue
                dignity = astro.vedic_dignity(name, data["sign"], data.get("degree_in_sign"))
                points = DIGNITY_POINTS.get(dignity, 0)
                score += points
                if points != 0:
                    contributions.append((f"{name} ({dignity.lower()}) in house {house}", points))

        for karaka in THEME_KARAKAS[theme]:
            data = chart["planets"].get(karaka)
            if not data:
                continue
            dignity = astro.vedic_dignity(karaka, data["sign"], data.get("degree_in_sign"))
            points = round(DIGNITY_POINTS.get(dignity, 0) * KARAKA_WEIGHT)
            score += points
            if points != 0:
                contributions.append((f"{karaka} (its own dignity: {dignity.lower()})", points))

        score = max(0, min(100, score))
        contributions.sort(key=lambda c: abs(c[1]), reverse=True)
        top = contributions[:2]

        if top:
            reason = "Driven mainly by " + " and ".join(desc for desc, _ in top) + "."
        else:
            reason = "No strongly dignified or afflicted placements in this area — a neutral baseline."

        results[theme] = {
            "score": score,
            "band": _band(score),
            "top_contributors": [{"description": d, "points": p} for d, p in top],
            "reason": reason,
        }
    return results


def _band(score: int) -> str:
    if score >= 70:
        return "Strong"
    if score >= 55:
        return "Good"
    if score >= 40:
        return "Balanced"
    return "Growth Area"
