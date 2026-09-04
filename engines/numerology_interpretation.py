"""
ANUPT — Numerology rule-based interpretation engine.

Same role as engines/astrology_interpretation.py: the "Rule Engine" step
of Rule Engine -> Evidence -> AI -> Final Reading. Everything here is
deterministic — given the same numbers, always the same output — and
composed from documented rules and the trait-affinity data already built
for engines/numerology_scoring.py, not invented or AI-generated.

The relationship_strength() function below is deliberately built on top
of NUMBER_THEME_AFFINITY (already hand-authored and tested for scoring)
rather than a second, separately-invented "which numbers are compatible"
table — reusing one documented source of truth instead of maintaining two
tables that could quietly disagree with each other.
"""

import math

from engines import numerology as num
from engines.numerology_scoring import NUMBER_THEME_AFFINITY, THEMES

MASTER_REDUCTIONS = {11: 2, 22: 4, 33: 6}

RELATIONSHIP_TIER_LANGUAGE = {
    5: {"label": "Aligned", "verb": "point in the same direction and reinforce each other"},
    4: {"label": "Harmonic", "verb": "are different expressions of a shared underlying root"},
    3: {"label": "Complementary", "verb": "bring different strengths that can work well together"},
    2: {"label": "Distinct", "verb": "pull toward genuinely different territory, needing conscious balance"},
}


def _cosine_similarity(a: dict, b: dict) -> float:
    keys = THEMES
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(a.get(k, 0) ** 2 for k in keys))
    mag_b = math.sqrt(sum(b.get(k, 0) ** 2 for k in keys))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def relationship_strength(number_a: int, number_b: int, label_a: str, label_b: str) -> dict:
    """How two of a person's core numbers relate — genuinely graded, not a
    fixed 'here's what your numbers mean' answer regardless of which two
    numbers they actually are."""
    basis = []
    if number_a == number_b:
        tier = 5
        basis.append(f"{label_a} and {label_b} are the same number ({number_a})")
    elif MASTER_REDUCTIONS.get(number_a) == number_b or MASTER_REDUCTIONS.get(number_b) == number_a:
        tier = 4
        basis.append(f"one is the Master Number reduction of the other ({number_a} <-> {number_b})")
    else:
        similarity = _cosine_similarity(
            NUMBER_THEME_AFFINITY.get(number_a, {}), NUMBER_THEME_AFFINITY.get(number_b, {})
        )
        basis.append(f"trait-affinity similarity between {number_a} and {number_b}: {round(similarity, 2)}")
        if similarity >= 0.75:
            tier = 4
        elif similarity >= 0.45:
            tier = 3
        else:
            tier = 2

    tier_lang = RELATIONSHIP_TIER_LANGUAGE[tier]
    interpretation = (
        f"Your {label_a} ({number_a}) and {label_b} ({number_b}) {tier_lang['verb']}."
    )
    return {
        "number_a": number_a, "number_b": number_b, "label_a": label_a, "label_b": label_b,
        "strength_tier": tier, "strength_label": tier_lang["label"],
        "interpretation": interpretation, "basis": basis,
    }


def interpret_number(entry: dict, context_label: str) -> dict:
    """Rule-based interpretation for a single number entry from
    engines.numerology.full_profile() — folds in Master Number and Karmic
    Debt exception handling, not just the base meaning lookup."""
    value = entry["value"]
    basis = [f"{context_label} = {value}"]
    clauses = [f"Your {context_label} is {value} — {num.NUMBER_MEANINGS.get(value, '').rstrip('.').lower()}."]

    if entry.get("is_master"):
        clauses.append(
            f"As a Master Number, {value} carries amplified intensity here — real potential, but "
            "also real pressure to live up to it rather than an automatically easier path."
        )
        basis.append(f"{value} is a Master Number")

    if entry.get("karmic_debt"):
        clauses.append(entry.get("karmic_debt_meaning", ""))
        basis.append(f"Karmic Debt {entry['karmic_debt']} appeared in this number's reduction")

    return {
        "context_label": context_label, "value": value,
        "is_master": entry.get("is_master", False),
        "karmic_debt": entry.get("karmic_debt"),
        "interpretation": " ".join(c for c in clauses if c),
        "basis": basis,
    }


def build_profile_evidence(profile: dict) -> dict:
    """Runs interpret_number() over every core number plus
    relationship_strength() over the two most-referenced pairings
    (Life Path <-> Destiny, Life Path <-> Personal Year) — the full
    rule-based 'Evidence' layer, ready for the UI or for the AI to
    synthesize from, without any AI call needed to produce it."""
    numbers = [
        interpret_number(profile[key], label)
        for key, label in [
            ("life_path", "Life Path"), ("destiny", "Destiny"), ("soul_urge", "Soul Urge"),
            ("personality", "Personality"), ("birthday", "Birthday (Mulank)"),
            ("attitude", "Attitude"), ("maturity", "Maturity"),
        ]
    ]
    relationships = [
        relationship_strength(
            profile["life_path"]["value"], profile["destiny"]["value"], "Life Path", "Destiny"
        ),
        relationship_strength(
            profile["life_path"]["value"], profile["personal_year"]["value"], "Life Path", "Personal Year"
        ),
    ]
    return {"numbers": numbers, "relationships": relationships}
