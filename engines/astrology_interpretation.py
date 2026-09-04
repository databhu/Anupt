"""
ANUPT — Astrology rule-based interpretation engine.

This is the "Rule Engine" step of the Rule Engine -> Evidence -> AI ->
Final Reading pipeline. Everything in this file is deterministic: given
the same placement facts, it always produces the same interpretation,
computed from documented rules — never generated, never random, and not
a set of fixed canned sentences either. The output is composed from
modular clauses parametrized by the actual dignity/house/retrograde/
combustion facts of THIS specific placement, so two different placements
of the same planet produce genuinely different text, not the same
template with a noun swapped in.

Where this fits: engines/astrology.py computes WHAT is true (Sun is in
Leo, house 10, exalted). This file computes WHAT THAT MEANS, using
documented astrological rules — genuine interpretation, but still 100%
rule-based, not AI. ai/gemini_client.py's job shrinks accordingly: instead
of interpreting raw placements from scratch, it receives this module's
output as pre-interpreted evidence and its job becomes synthesis and
personalization — weaving already-interpreted findings into a narrative
that answers the user's specific question, not inventing the
interpretation of a placement on its own.
"""

from engines import astrology as astro

PLANET_SIGNIFICATIONS = {
    "Sun": "identity, vitality, and how you express authority",
    "Moon": "emotional instincts and how you process feelings",
    "Mars": "drive, assertion, and how you handle conflict",
    "Mercury": "communication style and analytical thinking",
    "Jupiter": "growth, opportunity, and where you find meaning",
    "Venus": "love, values, and what you find beautiful or worthwhile",
    "Saturn": "discipline, responsibility, and where life asks patience",
    "Uranus": "individuality, sudden change, and where you resist convention",
    "Neptune": "imagination, idealism, and where boundaries blur",
    "Pluto": "transformation, intensity, and where power dynamics play out",
    "Rahu": "unconventional ambition and what you're reaching toward",
    "Ketu": "detachment and what you're instinctively releasing",
}

# 5-tier strength scale, from dignity. Each tier has its own verb/tone —
# genuinely different phrasing, not the same sentence with a number changed.
_DIGNITY_TO_TIER = {
    "Exalted": 5, "Own Sign": 4, "Rulership": 4, "Moolatrikona": 4,
    "Neutral": 3, "Detriment": 2, "Fall": 2, "Debilitated": 1, "Detriment & Fall": 1,
}
STRENGTH_TIER_LANGUAGE = {
    5: {"label": "Very Strong", "verb": "expresses powerfully and naturally here"},
    4: {"label": "Strong", "verb": "expresses with real ease here"},
    3: {"label": "Moderate", "verb": "expresses in a balanced, situation-dependent way here"},
    2: {"label": "Challenged", "verb": "asks for conscious effort to express well here"},
    1: {"label": "Very Challenged", "verb": "tends to express in a blocked or overcompensating way here"},
}

# House-classification manifestation clause: HOW a trait tends to show up,
# not just how strong it is — a genuinely different axis of interpretation
# from dignity strength.
_HOUSE_CLASS_CLAUSE = {
    "Angular": "Angular houses are where things become visible in outer "
               "circumstances, so this tends to show up in concrete, noticeable ways.",
    "Succedent": "Succedent houses are about sustaining and stabilizing, so this "
                "tends to build gradually rather than announce itself suddenly.",
    "Cadent": "Cadent houses relate to learning and adaptation, so this often "
             "operates through reflection or behind-the-scenes effort rather than "
             "outward display.",
}

# Retrograde nuance is planet-specific — the SAME "turns inward" theme reads
# very differently for Mercury (communication) than for Mars (assertion).
_RETROGRADE_CLAUSE = {
    "Sun": "the Sun cannot truly retrograde from Earth's viewpoint in the way "
           "other planets do, so this note is typically not applicable.",
    "Moon": "the Moon does not retrograde in standard practice either.",
    "Mercury": "Retrograde Mercury here suggests communication and decisions "
              "benefit from a second pass — reconsidering before committing tends to serve well.",
    "Venus": "Retrograde Venus here suggests values and relationships may be "
            "revisited or reworked rather than settled quickly.",
    "Mars": "Retrograde Mars here suggests assertiveness turns inward — drive "
           "is present but may show as quiet frustration rather than direct action.",
    "Jupiter": "Retrograde Jupiter here suggests growth comes through "
              "reflection on past opportunities rather than reaching for new ones.",
    "Saturn": "Retrograde Saturn here suggests discipline is self-imposed and "
             "internally reworked rather than externally structured.",
    "Uranus": "Retrograde Uranus here suggests change happens through inner "
             "realization before it shows up outwardly.",
    "Neptune": "Retrograde Neptune here suggests idealism turns inward, "
              "toward personal reflection rather than outward vision.",
    "Pluto": "Retrograde Pluto here suggests transformation happens through "
            "private, internal processing rather than visible upheaval.",
}

# Classical combustion orbs (degrees from the Sun within which a planet is
# considered "burned up" / overshadowed) — commonly-cited approximate
# values; some traditions use slightly different numbers, and the Sun and
# Moon are not themselves subject to combustion.
COMBUSTION_ORBS = {
    "Moon": 12.0, "Mars": 17.0, "Mercury": 14.0, "Jupiter": 11.0,
    "Venus": 10.0, "Saturn": 15.0,
}


def is_combust(planet: str, planet_longitude: float, sun_longitude: float) -> bool:
    """A real, documented exception rule: a planet too close to the Sun is
    traditionally considered combust regardless of how favorable its sign
    dignity otherwise looks."""
    orb = COMBUSTION_ORBS.get(planet)
    if orb is None:
        return False
    return astro.angular_distance(planet_longitude, sun_longitude) <= orb


def interpret_placement(planet: str, sign: str, house: int, dignity: str,
                         retrograde: bool, house_classification: str,
                         planet_longitude: float | None = None,
                         sun_longitude: float | None = None) -> dict:
    """The core rule-based interpretation for one planet's placement.
    Returns {"strength_tier": 1-5, "strength_label": str, "interpretation": str,
    "basis": [str, ...], "exceptions_applied": [str, ...]} — `basis` lists
    exactly which facts drove the interpretation, and `exceptions_applied`
    lists any exception rules (currently: combustion) that modified it, so
    the reasoning is inspectable, not just asserted."""
    tier = _DIGNITY_TO_TIER.get(dignity, 3)
    basis = [f"{planet} is {dignity} in {sign}"]
    exceptions_applied = []

    combust = False
    if planet_longitude is not None and sun_longitude is not None:
        combust = is_combust(planet, planet_longitude, sun_longitude)
    if combust:
        tier = max(1, tier - 1)  # combustion pulls strength down a tier, never below 1
        exceptions_applied.append(
            f"{planet} is combust (within {COMBUSTION_ORBS[planet]}° of the Sun), which "
            "traditionally overshadows a planet's own expression regardless of its sign dignity."
        )
        basis.append("combustion (proximity to the Sun)")

    tier_lang = STRENGTH_TIER_LANGUAGE[tier]
    significations = PLANET_SIGNIFICATIONS.get(planet, "this planet's core themes")
    house_theme = astro.HOUSE_THEMES.get(house, "this area of life")

    sentence = (
        f"{planet} — {significations} — {tier_lang['verb']}, shaping matters of "
        f"{house_theme} (house {house})."
    )
    clauses = [sentence]

    class_clause = _HOUSE_CLASS_CLAUSE.get(house_classification)
    if class_clause:
        clauses.append(class_clause)
        basis.append(f"house {house} is {house_classification}")

    if retrograde:
        retro_clause = _RETROGRADE_CLAUSE.get(planet)
        if retro_clause and planet not in ("Sun", "Moon"):
            clauses.append(retro_clause)
            basis.append(f"{planet} is retrograde")

    if combust:
        clauses.append(
            f"That said, {planet} here is combust — its usual strength is somewhat "
            "muted by closeness to the Sun, even with otherwise favorable dignity."
            if tier >= 3 else
            f"Combustion adds to the challenge here — {planet}'s expression is both "
            "structurally difficult and further muted by closeness to the Sun."
        )

    return {
        "planet": planet, "sign": sign, "house": house, "dignity": dignity,
        "strength_tier": tier, "strength_label": tier_lang["label"],
        "interpretation": " ".join(clauses),
        "basis": basis,
        "exceptions_applied": exceptions_applied,
    }


# Aspect strength: how tight the orb is, not just which aspect it is —
# a 0.2° conjunction is a much stronger statement than a 7.8° one.
def _aspect_strength_tier(orb: float, aspect_name: str) -> int:
    max_orb = dict(astro.MAJOR_ASPECTS).get(aspect_name) or dict(astro.MINOR_ASPECTS).get(aspect_name, 8)
    max_orb = max_orb[1] if isinstance(max_orb, tuple) else max_orb
    ratio = orb / max_orb if max_orb else 1.0
    if ratio <= 0.15:
        return 5
    if ratio <= 0.4:
        return 4
    if ratio <= 0.7:
        return 3
    return 2


_ASPECT_NATURE = {
    "Conjunction": "blends the two planets' energies into one combined expression",
    "Sextile": "offers an easy, cooperative opportunity between these two energies",
    "Square": "creates productive tension that pushes growth through friction",
    "Trine": "flows smoothly, sometimes so easily it goes unused without effort",
    "Opposition": "creates a push-pull between these two energies that asks for balance",
    "Semi-sextile": "creates a subtle, easy-to-miss connection",
    "Semi-square": "creates minor, low-grade friction",
    "Quincunx": "asks for ongoing adjustment between two energies that don't naturally relate",
    "Sesquiquadrate": "creates a nagging, background tension",
}


def interpret_aspect(aspect: dict) -> dict:
    """Rule-based interpretation for one aspect (from
    engines.astrology.calculate_aspects()'s output). Strength comes from
    how exact the orb is, not just which aspect type it is."""
    planet_a, planet_b = aspect["point_a"], aspect["point_b"]
    aspect_name, orb = aspect["aspect"], aspect["orb"]
    tier = _aspect_strength_tier(orb, aspect_name)
    nature = _ASPECT_NATURE.get(aspect_name, "connects these two energies")
    sig_a = PLANET_SIGNIFICATIONS.get(planet_a, planet_a)
    sig_b = PLANET_SIGNIFICATIONS.get(planet_b, planet_b)

    tightness = "an exact, highly significant" if tier == 5 else \
                "a tight, clearly felt" if tier == 4 else \
                "a moderate" if tier == 3 else "a loose, background-level"

    interpretation = (
        f"{planet_a} {aspect_name} {planet_b} ({tightness} {orb}° orb) {nature} — "
        f"connecting {sig_a} with {sig_b}."
    )
    return {
        "planet_a": planet_a, "planet_b": planet_b, "aspect": aspect_name, "orb": orb,
        "strength_tier": tier,
        "strength_label": STRENGTH_TIER_LANGUAGE[tier]["label"] if tier in STRENGTH_TIER_LANGUAGE else "Moderate",
        "interpretation": interpretation,
        "basis": [f"{aspect_name} within {orb}° orb"],
    }


def build_chart_evidence(chart: dict, top_n_aspects: int = 6) -> dict:
    """Runs interpret_placement() over every planet in a chart and
    interpret_aspect() over its tightest aspects, returning the full
    rule-based 'Evidence' layer ready to hand to the AI for synthesis —
    or to show directly in the UI without any AI call at all."""
    placements = []
    for name, data in chart["planets"].items():
        dignity = astro.vedic_dignity(name, data["sign"], data.get("degree_in_sign"))
        placements.append(interpret_placement(
            name, data["sign"], data["house"], dignity, data.get("retrograde", False),
            astro.HOUSE_CLASSIFICATION.get(data["house"], "Succedent"),
            planet_longitude=data.get("longitude"),
            sun_longitude=chart["planets"].get("Sun", {}).get("longitude"),
        ))
    placements.sort(key=lambda p: p["strength_tier"], reverse=True)

    longitudes = {name: d["longitude"] for name, d in chart["planets"].items()}
    aspects = astro.calculate_aspects(longitudes, include_minor=False)
    aspect_findings = [interpret_aspect(a) for a in aspects[:top_n_aspects]]

    return {"placements": placements, "aspects": aspect_findings}
