"""
ANUPT — Unified Insight Engine
The core differentiator: takes the structured, deterministic output of
Numerology, Astrology and Tarot (Palmistry folded in when available),
maps each to a shared theme space, and reports where the systems agree,
where they conflict, and how strongly — with a full evidence trail.

This engine only produces structured data. Turning it into readable
prose is the AI writer's job (ai/gemini_client.py) — and the AI is only
ever given this structured evidence, never asked to invent findings.
"""

THEMES = ["career", "finance", "relationships", "family", "personal_growth", "spirituality"]

NUMEROLOGY_THEME_MAP = {
    1: "career", 2: "relationships", 3: "personal_growth", 4: "career",
    5: "personal_growth", 6: "family", 7: "spirituality", 8: "finance",
    9: "spirituality", 11: "spirituality", 22: "career", 33: "family",
}

HOUSE_THEME_MAP = {
    10: "career", 2: "finance", 11: "finance", 7: "relationships",
    4: "family", 5: "personal_growth", 3: "personal_growth",
    9: "spirituality", 12: "spirituality", 1: "personal_growth",
    6: "career", 8: "spirituality",
}

TAROT_SUIT_THEME_MAP = {
    "Wands": "career", "Cups": "relationships", "Swords": "personal_growth",
    "Pentacles": "finance", None: "spirituality",  # None => Major Arcana
}


def numerology_signal(numerology_profile: dict) -> dict:
    """Weight-1.0 signal on the Life Path's mapped theme, weight-0.5 on Personal Year's."""
    signal = {t: 0.0 for t in THEMES}
    lp_theme = NUMEROLOGY_THEME_MAP.get(numerology_profile["life_path"]["value"])
    py_theme = NUMEROLOGY_THEME_MAP.get(numerology_profile["personal_year"]["value"])
    if lp_theme:
        signal[lp_theme] += 1.0
    if py_theme:
        signal[py_theme] += 0.5
    total = sum(signal.values()) or 1.0
    return {t: round(v / total, 3) for t, v in signal.items()}, {
        "life_path_theme": lp_theme, "personal_year_theme": py_theme,
    }


def astrology_signal(chart: dict) -> dict:
    """Weight-1.0 signal from the current Mahadasha lord's house, weight-0.5 from Antardasha lord's."""
    signal = {t: 0.0 for t in THEMES}
    planets = chart["planets"]
    maha_lord = chart["dasha"]["mahadasha"]
    antar_lord = chart["dasha"]["antardasha"]

    maha_theme = antar_theme = None
    if maha_lord in planets:
        maha_theme = HOUSE_THEME_MAP.get(planets[maha_lord]["house"])
        if maha_theme:
            signal[maha_theme] += 1.0
    if antar_lord in planets and antar_lord != maha_lord:
        antar_theme = HOUSE_THEME_MAP.get(planets[antar_lord]["house"])
        if antar_theme:
            signal[antar_theme] += 0.5

    total = sum(signal.values()) or 1.0
    return {t: round(v / total, 3) for t, v in signal.items()}, {
        "mahadasha_lord": maha_lord, "mahadasha_theme": maha_theme,
        "antardasha_lord": antar_lord, "antardasha_theme": antar_theme,
    }


def tarot_signal(cards: list) -> dict:
    """Each drawn card contributes an equal share of weight to its suit-mapped theme."""
    signal = {t: 0.0 for t in THEMES}
    per_card_weight = 1.0 / max(len(cards), 1)
    contributions = []
    for card in cards:
        theme = TAROT_SUIT_THEME_MAP.get(card["suit"])
        signal[theme] += per_card_weight
        contributions.append({"card": card["name"], "orientation": card["orientation"], "theme": theme})
    total = sum(signal.values()) or 1.0
    return {t: round(v / total, 3) for t, v in signal.items()}, contributions


def synthesize(numerology_profile: dict, chart: dict, cards: list) -> dict:
    """
    Combine the three deterministic engines into a unified, theme-scored,
    evidence-backed structure. This is what gets handed to the AI writer.
    """
    num_sig, num_evidence = numerology_signal(numerology_profile)
    astro_sig, astro_evidence = astrology_signal(chart)
    tarot_sig, tarot_evidence = tarot_signal(cards)

    combined = {}
    for theme in THEMES:
        systems_touching = []
        if num_sig[theme] > 0:
            systems_touching.append("Numerology")
        if astro_sig[theme] > 0:
            systems_touching.append("Astrology")
        if tarot_sig[theme] > 0:
            systems_touching.append("Tarot")

        score = num_sig[theme] + astro_sig[theme] + tarot_sig[theme]
        n = len(systems_touching)
        if n >= 3:
            strength = "strong"
        elif n == 2:
            strength = "moderate"
        elif n == 1:
            strength = "weak"
        else:
            strength = "none"

        combined[theme] = {
            "score": round(score, 3),
            "strength": strength,
            "supporting_systems": systems_touching,
        }

    ranked_themes = sorted(THEMES, key=lambda t: combined[t]["score"], reverse=True)
    top_theme = ranked_themes[0]

    return {
        "theme_scores": combined,
        "ranked_themes": ranked_themes,
        "top_theme": top_theme,
        "evidence": {
            "numerology": num_evidence,
            "astrology": astro_evidence,
            "tarot": tarot_evidence,
        },
    }
