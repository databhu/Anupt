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

# Translated versions of engines.numerology.NUMBER_MEANINGS and
# KARMIC_DEBT_MEANINGS, used only at this display/interpretation layer —
# engines/numerology.py's own full_profile() output stays English/
# canonical regardless of language, since it's tested extensively as a
# deterministic calculation contract, not display text. Falls back to the
# canonical English meaning if a value or language isn't covered here.
_NUMBER_MEANINGS_TRANSLATED = {
    1: {"hi": "नेतृत्व, स्वतंत्रता, मौलिकता, ऊर्जा।", "mr": "नेतृत्व, स्वातंत्र्य, मौलिकता, उत्साह."},
    2: {"hi": "साझेदारी, कूटनीति, संवेदनशीलता, सहयोग।", "mr": "भागीदारी, मुत्सद्देगिरी, संवेदनशीलता, सहकार्य."},
    3: {"hi": "रचनात्मकता, अभिव्यक्ति, आशावाद, संचार।", "mr": "सर्जनशीलता, अभिव्यक्ती, आशावाद, संवाद."},
    4: {"hi": "संरचना, अनुशासन, विश्वसनीयता, कड़ी मेहनत।", "mr": "रचना, शिस्त, विश्वासार्हता, कठोर परिश्रम."},
    5: {"hi": "स्वतंत्रता, परिवर्तन, अनुकूलनशीलता, साहसिकता।", "mr": "स्वातंत्र्य, बदल, अनुकूलनक्षमता, साहस."},
    6: {"hi": "जिम्मेदारी, पोषण, सामंजस्य, सेवा।", "mr": "जबाबदारी, संगोपन, सुसंवाद, सेवा."},
    7: {"hi": "आत्मनिरीक्षण, विश्लेषण, आध्यात्मिकता, ज्ञान।", "mr": "आत्मपरीक्षण, विश्लेषण, अध्यात्म, शहाणपण."},
    8: {"hi": "शक्ति, महत्वाकांक्षा, भौतिक निपुणता, अधिकार।", "mr": "सामर्थ्य, महत्त्वाकांक्षा, भौतिक प्रभुत्व, अधिकार."},
    9: {"hi": "करुणा, आदर्शवाद, पूर्णता, मानवतावाद।", "mr": "करुणा, आदर्शवाद, पूर्णता, मानवतावाद."},
    11: {"hi": "मास्टर नंबर — अंतर्ज्ञान, प्रकाश, आध्यात्मिक अंतर्दृष्टि।",
         "mr": "मास्टर नंबर — अंतर्ज्ञान, प्रकाश, आध्यात्मिक अंतर्दृष्टी."},
    22: {"hi": "मास्टर नंबर — महान निर्माता, बड़े पैमाने की दृष्टि को साकार करना।",
         "mr": "मास्टर नंबर — महान निर्माता, मोठ्या प्रमाणावरील दृष्टीकोन प्रत्यक्षात आणणे."},
    33: {"hi": "मास्टर नंबर — महान शिक्षक, निःस्वार्थ करुणा, उपचार।",
         "mr": "मास्टर नंबर — महान शिक्षक, निःस्वार्थ करुणा, उपचार."},
}
_KARMIC_DEBT_MEANINGS_TRANSLATED = {
    13: {"hi": "कर्म ऋण 13 — जहां कभी शॉर्टकट अपनाए गए थे, वहां अनुशासित, ईमानदार प्रयास की मांग करता है।",
         "mr": "कर्म ऋण 13 — जिथे पूर्वी शॉर्टकट घेतले गेले होते तिथे शिस्तबद्ध, प्रामाणिक प्रयत्नांची मागणी करतो."},
    14: {"hi": "कर्म ऋण 14 — जहां अति ने असंतुलन पैदा किया था, वहां संयम और अनुकूलनशीलता की मांग करता है।",
         "mr": "कर्म ऋण 14 — जिथे अतिरेकाने असंतुलन निर्माण केले होते तिथे संयम आणि अनुकूलनक्षमतेची मागणी करतो."},
    16: {"hi": "कर्म ऋण 16 — अहंकार-प्रेरित उथल-पुथल के बाद विनम्रता और स्वयं के पुनर्निर्माण की मांग करता है।",
         "mr": "कर्म ऋण 16 — अहंकारामुळे झालेल्या उलथापालथीनंतर नम्रता आणि स्वतःच्या पुनर्बांधणीची मागणी करतो."},
    19: {"hi": "कर्म ऋण 19 — दूसरों पर निर्भर हुए बिना आत्मनिर्भरता की मांग करता है।",
         "mr": "कर्म ऋण 19 — इतरांवर अवलंबून न राहता आत्मनिर्भरतेची मागणी करतो."},
}

# The two template sentences interpret_number() wraps around a core
# meaning — translated too, so the whole finding reads naturally in each
# language rather than mixing translated content with English scaffolding.
_MASTER_NUMBER_CLAUSE = {
    "en": "As a Master Number, {value} carries amplified intensity here — real potential, but "
          "also real pressure to live up to it rather than an automatically easier path.",
    "hi": "एक मास्टर नंबर के रूप में, {value} यहां और अधिक तीव्रता लेकर आता है — वास्तविक क्षमता, "
          "लेकिन साथ ही उस पर खरा उतरने का वास्तविक दबाव भी, न कि कोई अपने-आप आसान रास्ता।",
    "mr": "मास्टर नंबर म्हणून, {value} इथे अधिक तीव्रता घेऊन येतो — खरी क्षमता, पण त्यासोबतच ती "
          "सिद्ध करण्याचा खरा दबाव देखील, आपोआप सोपा मार्ग नाही.",
}
_YOUR_X_IS_Y_TEMPLATE = {
    "en": "Your {label} is {value} — {meaning}",
    "hi": "आपका {label} {value} है — {meaning}",
    "mr": "तुमचा {label} {value} आहे — {meaning}",
}


def translated_number_meaning(value: int, lang: str = "en") -> str:
    """The number's core meaning in `lang`, falling back to the canonical
    English meaning (engines.numerology.NUMBER_MEANINGS) if this value or
    language isn't covered."""
    if lang in ("hi", "mr") and value in _NUMBER_MEANINGS_TRANSLATED:
        translated = _NUMBER_MEANINGS_TRANSLATED[value].get(lang)
        if translated:
            return translated
    return num.NUMBER_MEANINGS.get(value, "")


def translated_karmic_debt_meaning(karmic_debt: int, lang: str = "en") -> str:
    """Same fallback pattern as translated_number_meaning(), for
    engines.numerology.KARMIC_DEBT_MEANINGS."""
    if lang in ("hi", "mr") and karmic_debt in _KARMIC_DEBT_MEANINGS_TRANSLATED:
        translated = _KARMIC_DEBT_MEANINGS_TRANSLATED[karmic_debt].get(lang)
        if translated:
            return translated
    return num.KARMIC_DEBT_MEANINGS.get(karmic_debt, "")

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


def interpret_number(entry: dict, context_label: str, lang: str = "en") -> dict:
    """Rule-based interpretation for a single number entry from
    engines.numerology.full_profile() — folds in Master Number and Karmic
    Debt exception handling, not just the base meaning lookup. `lang`
    ("en"/"hi"/"mr") controls the DISPLAY text only — entry["value"] and
    the underlying calculation are unaffected, since numbers aren't
    language-dependent."""
    value = entry["value"]
    basis = [f"{context_label} = {value}"]
    meaning = translated_number_meaning(value, lang)
    end_punctuation = "।" if lang == "hi" else "."
    meaning_display = meaning.rstrip("।.").strip()
    if lang == "en":
        meaning_display = meaning_display.lower()
    first_clause = _YOUR_X_IS_Y_TEMPLATE.get(lang, _YOUR_X_IS_Y_TEMPLATE["en"]).format(
        label=context_label, value=value, meaning=meaning_display + end_punctuation
    )
    clauses = [first_clause]

    if entry.get("is_master"):
        clauses.append(_MASTER_NUMBER_CLAUSE.get(lang, _MASTER_NUMBER_CLAUSE["en"]).format(value=value))
        basis.append(f"{value} is a Master Number")

    if entry.get("karmic_debt"):
        clauses.append(translated_karmic_debt_meaning(entry["karmic_debt"], lang))
        basis.append(f"Karmic Debt {entry['karmic_debt']} appeared in this number's reduction")

    return {
        "context_label": context_label, "value": value,
        "is_master": entry.get("is_master", False),
        "karmic_debt": entry.get("karmic_debt"),
        "interpretation": " ".join(c for c in clauses if c),
        "basis": basis,
    }


def build_profile_evidence(profile: dict, lang: str = "en") -> dict:
    """Runs interpret_number() over every core number plus
    relationship_strength() over the two most-referenced pairings
    (Life Path <-> Destiny, Life Path <-> Personal Year) — the full
    rule-based 'Evidence' layer, ready for the UI or for the AI to
    synthesize from, without any AI call needed to produce it."""
    numbers = [
        interpret_number(profile[key], label, lang)
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
