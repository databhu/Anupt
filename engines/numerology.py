"""
ANUPT — Numerology Engine
100% deterministic. No AI involved in calculation — only in later interpretation.
Pythagorean system.
"""

from datetime import date

LETTER_VALUES = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
    'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8,
}
VOWELS = set('AEIOU')
MASTER_NUMBERS = {11, 22, 33}

NUMBER_MEANINGS = {
    1: "Leadership, independence, originality, drive.",
    2: "Partnership, diplomacy, sensitivity, cooperation.",
    3: "Creativity, expression, optimism, communication.",
    4: "Structure, discipline, reliability, hard work.",
    5: "Freedom, change, adaptability, adventure.",
    6: "Responsibility, nurturing, harmony, service.",
    7: "Introspection, analysis, spirituality, wisdom.",
    8: "Power, ambition, material mastery, authority.",
    9: "Compassion, idealism, completion, humanitarianism.",
    11: "Master Number — intuition, illumination, spiritual insight.",
    22: "Master Number — the master builder, large-scale vision made real.",
    33: "Master Number — master teacher, selfless compassion, healing.",
}


def _reduce(n: int, keep_master: bool = True) -> int:
    """Digit-sum reduction, preserving master numbers 11/22/33 unless disabled."""
    while n > 9:
        if keep_master and n in MASTER_NUMBERS:
            return n
        n = sum(int(d) for d in str(n))
    return n


def _digit_sum_of_string(digits: str) -> int:
    n = sum(int(c) for c in digits if c.isdigit())
    return _reduce(n)


def life_path_number(dob: date) -> int:
    """Sum all digits of the birth date, reducing (master numbers preserved)."""
    digits = f"{dob.month}{dob.day}{dob.year}"
    return _digit_sum_of_string(digits)


def birthday_number(dob: date) -> int:
    return _reduce(dob.day)


def destiny_number(full_name: str) -> int:
    """Expression/Destiny number — sum of all letters in the full name."""
    total = sum(LETTER_VALUES.get(ch, 0) for ch in full_name.upper() if ch.isalpha())
    return _reduce(total)


def soul_urge_number(full_name: str) -> int:
    """Heart's Desire — sum of vowels only."""
    total = sum(LETTER_VALUES.get(ch, 0) for ch in full_name.upper() if ch in VOWELS)
    return _reduce(total)


def personality_number(full_name: str) -> int:
    """Sum of consonants only."""
    total = sum(
        LETTER_VALUES.get(ch, 0)
        for ch in full_name.upper()
        if ch.isalpha() and ch not in VOWELS
    )
    return _reduce(total)


def maturity_number(life_path: int, destiny: int) -> int:
    return _reduce(life_path + destiny)


def personal_year_number(dob: date, target_year: int) -> int:
    digits = f"{dob.month}{dob.day}{target_year}"
    return _digit_sum_of_string(digits)


def personal_month_number(personal_year: int, target_month: int) -> int:
    return _reduce(personal_year + target_month)


def personal_day_number(personal_month: int, target_day: int) -> int:
    return _reduce(personal_month + target_day)


def full_profile(full_name: str, dob: date, as_of: date | None = None) -> dict:
    """Returns the complete deterministic numerology profile as structured data."""
    as_of = as_of or date.today()
    lp = life_path_number(dob)
    des = destiny_number(full_name)
    su = soul_urge_number(full_name)
    per = personality_number(full_name)
    bd = birthday_number(dob)
    mat = maturity_number(lp, des)
    py = personal_year_number(dob, as_of.year)
    pm = personal_month_number(py, as_of.month)
    pd = personal_day_number(pm, as_of.day)

    return {
        "life_path": {"value": lp, "meaning": NUMBER_MEANINGS.get(lp, "")},
        "destiny": {"value": des, "meaning": NUMBER_MEANINGS.get(des, "")},
        "soul_urge": {"value": su, "meaning": NUMBER_MEANINGS.get(su, "")},
        "personality": {"value": per, "meaning": NUMBER_MEANINGS.get(per, "")},
        "birthday": {"value": bd, "meaning": NUMBER_MEANINGS.get(bd, "")},
        "maturity": {"value": mat, "meaning": NUMBER_MEANINGS.get(mat, "")},
        "personal_year": {"value": py, "meaning": NUMBER_MEANINGS.get(py, "")},
        "personal_month": {"value": pm, "meaning": NUMBER_MEANINGS.get(pm, "")},
        "personal_day": {"value": pd, "meaning": NUMBER_MEANINGS.get(pd, "")},
    }
