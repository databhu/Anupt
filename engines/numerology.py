"""
ANUPT — Numerology Engine (calculation layer only)

100% deterministic. No AI, no interpretation-writing, and no UI here — this
module's only job is arithmetic and the deterministic meaning lookups that
go with it. engines/numerology_scoring.py builds theme scores on top of
these numbers; ai/gemini_client.py turns them into prose; app.py renders
them. None of those layers recompute a number — they only ever read what
this module already calculated.

Belief-based, not scientific: numerology has no basis in evidence-based
science. Every number here is real arithmetic on a name/date, but what
those numbers are said to *mean* is a centuries-old symbolic tradition,
not a validated predictive method. Treat it as a structured way to reflect
on yourself — the app's UI carries this disclaimer explicitly (see
app.py's Numerology page header) and this module's numbers are the
"structured input" the AI is required to reason from, never numbers it's
allowed to invent.

Backward compatibility: full_profile(full_name, dob) is called elsewhere
in the app (engines/unified.py, app.py) expecting a flat dict where every
value is at least {"value": int, "meaning": str}, and dict.items() is
iterable as (label, {"value":.., ...}) pairs for a medallion grid. That
shape is preserved exactly — every field just gained extra keys
(calculation steps, karmic debt, master-number flag) that old callers
simply don't look at. Anything with a genuinely different shape (Pinnacles,
Challenges, name comparison) lives in its own new function instead of
being crammed into full_profile(), specifically so nothing that already
reads full_profile() can break.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Letter-value systems
# ---------------------------------------------------------------------------

PYTHAGOREAN_VALUES = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
    'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8,
}

# Chaldean is the older of the two systems in common use. Its letter-value
# table is fixed by tradition (not derived alphabetically like Pythagorean's)
# and deliberately never assigns 9 to a letter — 9 is treated as sacred/
# complete on its own. Name-based numbers (Expression, Soul Urge, Personality)
# differ between the two systems; date-based numbers (Life Path, Birthday,
# Attitude, Personal Year/Month/Day, Pinnacles, Challenges) do not, since the
# systems' actual difference is specifically about letter values, not dates.
CHALDEAN_VALUES = {
    'A': 1, 'I': 1, 'J': 1, 'Q': 1, 'Y': 1,
    'B': 2, 'K': 2, 'R': 2,
    'C': 3, 'G': 3, 'L': 3, 'S': 3,
    'D': 4, 'M': 4, 'T': 4,
    'E': 5, 'H': 5, 'N': 5, 'X': 5,
    'U': 6, 'V': 6, 'W': 6,
    'O': 7, 'Z': 7,
    'F': 8, 'P': 8,
}

LETTER_SYSTEMS = {"pythagorean": PYTHAGOREAN_VALUES, "chaldean": CHALDEAN_VALUES}
VOWELS = set('AEIOU')
MASTER_NUMBERS = {11, 22, 33}
KARMIC_DEBT_NUMBERS = {13, 14, 16, 19}

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

KARMIC_DEBT_MEANINGS = {
    13: "Karmic Debt 13 — asks for disciplined, honest effort where shortcuts were once taken.",
    14: "Karmic Debt 14 — asks for moderation and adaptability where excess once caused imbalance.",
    16: "Karmic Debt 16 — asks for humility and a rebuild of self after ego-driven upheaval.",
    19: "Karmic Debt 19 — asks for self-reliance without leaning on or over others.",
}

PINNACLE_MEANINGS = {
    1: "A period of new starts — independence, initiative, and stepping into leadership.",
    2: "A period of partnership — patience, diplomacy, and working well with others.",
    3: "A period of self-expression — creativity, connection, and optimism.",
    4: "A period of building — discipline, structure, and laying solid foundations.",
    5: "A period of change — freedom, movement, and adaptability.",
    6: "A period of responsibility — home, family, and service to others.",
    7: "A period of reflection — introspection, study, and inner development.",
    8: "A period of achievement — ambition, authority, and material reward.",
    9: "A period of completion — release, humanitarian focus, and closing a chapter.",
    11: "A heightened period of intuition — spiritual insight looks for a way to be shared.",
    22: "A heightened period of large-scale building — turning a big vision into something lasting.",
    33: "A heightened period of selfless service — teaching and healing on a wider scale.",
}

CHALLENGE_MEANINGS = {
    0: "A wide-open challenge — few fixed obstacles, but few easy answers either; the path is self-chosen.",
    1: "Learning independence — moving through self-doubt and reluctance to stand alone.",
    2: "Learning cooperation — moving through oversensitivity and difficulty trusting others.",
    3: "Learning expression — moving through scattered energy or holding feelings back.",
    4: "Learning structure — moving through resistance to discipline and steady work.",
    5: "Learning moderation — moving through restlessness and impulsive change.",
    6: "Learning balance — moving through over-responsibility or expecting too much of others.",
    7: "Learning trust — moving through isolation and overthinking.",
    8: "Learning healthy power — moving through struggles with money, control, or authority.",
}

CORE_NUMBER_CONTEXT = {
    "life_path": "Your overall journey — the path your life tends to move along.",
    "birthday": "Your Root Number (Mulank) — a natural talent tied to the day you were born.",
    "attitude": "How you meet the world — first impressions and instinctive reactions.",
    "destiny": "Your Destiny/Expression Number — the potential your full name points toward.",
    "soul_urge": "Your Heart's Desire — what you inwardly want, from the vowels in your name.",
    "personality": "How others perceive you — from the consonants in your name.",
    "maturity": "Where Life Path and Destiny converge — who you grow into later in life.",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def clean_letters(name: str) -> str:
    """Uppercased, letters only. Hyphens, apostrophes, spaces and any other
    punctuation are dropped, not treated as separators with their own value —
    so 'Mary-Jane', "O'Brien" and multi-word full names all just contribute
    their letters in sequence, which is the standard, widely-used convention."""
    return "".join(ch for ch in name.upper() if ch.isalpha())


def _reduce(n: int, keep_master: bool = True) -> int:
    """Digit-sum reduction, preserving master numbers 11/22/33 unless disabled.
    Kept as a simple value-only helper (no step tracking) since a few internal
    computations — Pinnacles/Challenges base digits — only need the number,
    not a rendered explanation."""
    while n > 9:
        if keep_master and n in MASTER_NUMBERS:
            return n
        n = sum(int(d) for d in str(n))
    return n


def _reduce_with_steps(n: int, keep_master: bool = True) -> tuple[int, int | None, list[str]]:
    """Same reduction as _reduce(), but also returns (a) the first Karmic Debt
    number (13/14/16/19) encountered anywhere along the way, if any, and
    (b) a human-readable step-by-step trail for the UI's calculation-
    transparency panel, e.g. ["8 + 8 + 1 + 9 + 9 + 5 = 40", "4 + 0 = 4"].
    Karmic debt is checked at every intermediate sum, not just the input —
    this is the standard method (a debt number can appear mid-reduction even
    if the very first raw total wasn't one)."""
    steps: list[str] = []
    karmic_debt = None
    while n > 9:
        if n in KARMIC_DEBT_NUMBERS and karmic_debt is None:
            karmic_debt = n
        if keep_master and n in MASTER_NUMBERS:
            break
        digits = str(n)
        digit_sum = sum(int(d) for d in digits)
        steps.append(f"{' + '.join(digits)} = {digit_sum}")
        n = digit_sum
    return n, karmic_debt, steps


def _number_result(value: int, karmic_debt: int | None, steps: list[str], input_breakdown: str,
                    context_key: str | None = None) -> dict:
    """The shared shape every core/cycle number is returned in. `value` and
    `meaning` are the two keys every existing caller reads — everything else
    is additive detail new callers can use for calculation transparency."""
    return {
        "value": value,
        "meaning": NUMBER_MEANINGS.get(value, ""),
        "context": CORE_NUMBER_CONTEXT.get(context_key, "") if context_key else "",
        "is_master": value in MASTER_NUMBERS,
        "karmic_debt": karmic_debt,
        "karmic_debt_meaning": KARMIC_DEBT_MEANINGS.get(karmic_debt) if karmic_debt else None,
        "calculation": {"input": input_breakdown, "steps": steps},
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_name(name: str) -> list[str]:
    """Returns a list of human-readable warnings for edge cases — never
    raises, since a warning should inform the reading, not block it. An
    empty list means nothing unusual was found."""
    warnings = []
    letters = clean_letters(name)
    if not letters:
        warnings.append("No letters found in this name — name-based numbers can't be calculated.")
        return warnings
    if len(letters) < 2:
        warnings.append("Very short name — name-based numbers may not be meaningful with a single letter.")
    if not any(ch in VOWELS for ch in letters):
        warnings.append("No vowels found — Soul Urge will be 0 (no vowel-based signal to calculate).")
    if all(ch in VOWELS for ch in letters):
        warnings.append("No consonants found — Personality will be 0 (no consonant-based signal to calculate).")
    return warnings


def validate_dob(dob: date, as_of: date | None = None) -> list[str]:
    """Sanity checks beyond what Python's own date type already guarantees
    (a `date` object can't represent Feb 30 etc. — this catches the cases
    that are structurally valid dates but not sensible birth dates)."""
    as_of = as_of or date.today()
    warnings = []
    if dob > as_of:
        warnings.append("Birth date is in the future.")
    age_years = (as_of - dob).days / 365.25
    if age_years > 130:
        warnings.append("Birth date implies an age over 130 years — please double-check it.")
    if dob.year < 1900:
        warnings.append("Birth year is before 1900 — results are still calculated, but double-check the year.")
    return warnings


# ---------------------------------------------------------------------------
# Core numbers
# ---------------------------------------------------------------------------

def life_path_number(dob: date) -> int:
    """Value-only convenience wrapper (kept for any external code that just
    wants the int). full_profile() below uses the detailed version internally."""
    return _life_path_detailed(dob)["value"]


def _life_path_detailed(dob: date) -> dict:
    digits = f"{dob.month}{dob.day}{dob.year}"
    raw = sum(int(c) for c in digits if c.isdigit())
    value, karmic, steps = _reduce_with_steps(raw)
    breakdown = f"{dob.month}/{dob.day}/{dob.year} digits: " + " + ".join(digits) + f" = {raw}"
    full_steps = [breakdown] + steps if steps else [breakdown + " (already a single digit or Master Number)"]
    return _number_result(value, karmic, full_steps, breakdown, "life_path")


def birthday_number(dob: date) -> int:
    return _birthday_detailed(dob)["value"]


def _birthday_detailed(dob: date) -> dict:
    value, karmic, steps = _reduce_with_steps(dob.day)
    breakdown = f"Birth day {dob.day}"
    full_steps = steps if steps else [f"{dob.day} is already a single digit or Master Number"]
    return _number_result(value, karmic, full_steps, breakdown, "birthday")


def _attitude_detailed(dob: date) -> dict:
    """Also called the 'Sun Number' in some traditions: month and day are
    each reduced on their own first, then added — deliberately different
    from Life Path's all-digits-at-once method, per the common convention
    for this number. Other schools compute it slightly differently; this
    module picks one well-documented method and shows its exact steps
    rather than quietly picking a number."""
    m_val, m_karmic, m_steps = _reduce_with_steps(dob.month)
    d_val, d_karmic, d_steps = _reduce_with_steps(dob.day)
    combined = m_val + d_val
    value, karmic, steps = _reduce_with_steps(combined)
    karmic = karmic or m_karmic or d_karmic
    breakdown = f"Month {dob.month} → {m_val}, Day {dob.day} → {d_val}; {m_val} + {d_val} = {combined}"
    full_steps = (m_steps or [f"Month {dob.month} is already single-digit"]) + \
                 (d_steps or [f"Day {dob.day} is already single-digit"]) + \
                 [f"{m_val} + {d_val} = {combined}"] + steps
    return _number_result(value, karmic, full_steps, breakdown, "attitude")


def destiny_number(full_name: str, system: str = "pythagorean") -> int:
    return _destiny_detailed(full_name, system)["value"]


def _name_letter_values(full_name: str, system: str) -> list[tuple[str, int]]:
    table = LETTER_SYSTEMS.get(system, PYTHAGOREAN_VALUES)
    return [(ch, table.get(ch, 0)) for ch in clean_letters(full_name)]


def _destiny_detailed(full_name: str, system: str = "pythagorean") -> dict:
    pairs = _name_letter_values(full_name, system)
    raw = sum(v for _, v in pairs)
    value, karmic, steps = _reduce_with_steps(raw)
    breakdown = "+".join(f"{ch}({v})" for ch, v in pairs) + f" = {raw}" if pairs else "No letters in name"
    full_steps = [breakdown] + steps if steps else [breakdown]
    return _number_result(value, karmic, full_steps, breakdown, "destiny")


def soul_urge_number(full_name: str, system: str = "pythagorean") -> int:
    return _soul_urge_detailed(full_name, system)["value"]


def _soul_urge_detailed(full_name: str, system: str = "pythagorean") -> dict:
    pairs = [(ch, v) for ch, v in _name_letter_values(full_name, system) if ch in VOWELS]
    raw = sum(v for _, v in pairs)
    value, karmic, steps = _reduce_with_steps(raw)
    breakdown = ("+".join(f"{ch}({v})" for ch, v in pairs) + f" = {raw}") if pairs else "No vowels in name"
    full_steps = [breakdown] + steps if steps else [breakdown]
    return _number_result(value, karmic, full_steps, breakdown, "soul_urge")


def personality_number(full_name: str, system: str = "pythagorean") -> int:
    return _personality_detailed(full_name, system)["value"]


def _personality_detailed(full_name: str, system: str = "pythagorean") -> dict:
    pairs = [(ch, v) for ch, v in _name_letter_values(full_name, system) if ch not in VOWELS]
    raw = sum(v for _, v in pairs)
    value, karmic, steps = _reduce_with_steps(raw)
    breakdown = ("+".join(f"{ch}({v})" for ch, v in pairs) + f" = {raw}") if pairs else "No consonants in name"
    full_steps = [breakdown] + steps if steps else [breakdown]
    return _number_result(value, karmic, full_steps, breakdown, "personality")


def maturity_number(life_path: int, destiny: int) -> int:
    return _reduce(life_path + destiny)


def _maturity_detailed(life_path_val: int, destiny_val: int) -> dict:
    combined = life_path_val + destiny_val
    value, karmic, steps = _reduce_with_steps(combined)
    breakdown = f"Life Path {life_path_val} + Destiny {destiny_val} = {combined}"
    full_steps = [breakdown] + steps if steps else [breakdown]
    return _number_result(value, karmic, full_steps, breakdown, "maturity")


# ---------------------------------------------------------------------------
# Life cycles: Personal Year / Month / Day
# ---------------------------------------------------------------------------

def personal_year_number(dob: date, target_year: int) -> int:
    return _personal_year_detailed(dob, target_year)["value"]


def _personal_year_detailed(dob: date, target_year: int) -> dict:
    digits = f"{dob.month}{dob.day}{target_year}"
    raw = sum(int(c) for c in digits if c.isdigit())
    value, karmic, steps = _reduce_with_steps(raw)
    breakdown = f"Birth month/day {dob.month}/{dob.day} + year {target_year}: " + " + ".join(digits) + f" = {raw}"
    full_steps = [breakdown] + steps if steps else [breakdown]
    return _number_result(value, karmic, full_steps, breakdown)


def personal_month_number(personal_year: int, target_month: int) -> int:
    return _reduce(personal_year + target_month)


def _personal_month_detailed(personal_year_val: int, target_month: int) -> dict:
    combined = personal_year_val + target_month
    value, karmic, steps = _reduce_with_steps(combined)
    breakdown = f"Personal Year {personal_year_val} + month {target_month} = {combined}"
    full_steps = [breakdown] + steps if steps else [breakdown]
    return _number_result(value, karmic, full_steps, breakdown)


def personal_day_number(personal_month: int, target_day: int) -> int:
    return _reduce(personal_month + target_day)


def _personal_day_detailed(personal_month_val: int, target_day: int) -> dict:
    combined = personal_month_val + target_day
    value, karmic, steps = _reduce_with_steps(combined)
    breakdown = f"Personal Month {personal_month_val} + day {target_day} = {combined}"
    full_steps = [breakdown] + steps if steps else [breakdown]
    return _number_result(value, karmic, full_steps, breakdown)


# ---------------------------------------------------------------------------
# Full profile — the main entry point, backward-compatible shape
# ---------------------------------------------------------------------------

def full_profile(full_name: str, dob: date, as_of: date | None = None,
                  system: str = "pythagorean") -> dict:
    """Returns the complete deterministic numerology profile. Every value is
    at minimum {"value": int, "meaning": str} — existing callers only ever
    read those two keys plus dict.items(), so this stays a drop-in match for
    every call site that predates this file's expansion. `system` is new and
    optional (defaults to "pythagorean", the original behaviour) — pass
    "chaldean" to use that letter-value table for the name-based numbers
    (date-based numbers are identical either way, see the module docstring)."""
    as_of = as_of or date.today()

    lp = _life_path_detailed(dob)
    bd = _birthday_detailed(dob)
    att = _attitude_detailed(dob)
    des = _destiny_detailed(full_name, system)
    su = _soul_urge_detailed(full_name, system)
    per = _personality_detailed(full_name, system)
    mat = _maturity_detailed(lp["value"], des["value"])

    py = _personal_year_detailed(dob, as_of.year)
    pm = _personal_month_detailed(py["value"], as_of.month)
    pd = _personal_day_detailed(pm["value"], as_of.day)

    return {
        "life_path": lp,
        "destiny": des,
        "soul_urge": su,
        "personality": per,
        "birthday": bd,
        "attitude": att,
        "maturity": mat,
        "personal_year": py,
        "personal_month": pm,
        "personal_day": pd,
    }


# ---------------------------------------------------------------------------
# Pinnacles & Challenges — separate function (different shape: lists of
# periods, not the flat {"value","meaning"} dict full_profile() returns) so
# nothing that iterates full_profile().items() for a medallion grid breaks.
# ---------------------------------------------------------------------------

def pinnacle_and_challenge_cycles(dob: date) -> dict:
    """The four Pinnacle periods (life themes) and four Challenge periods
    (what each period asks you to work through), with the age range each
    one covers. Age ranges are anchored to Life Path per the standard
    '36 minus Life Path' method; the fourth period has no fixed end."""
    m = _reduce(dob.month)
    d = _reduce(dob.day)
    y = _reduce(sum(int(c) for c in str(dob.year)))
    lp = _reduce(sum(int(c) for c in f"{dob.month}{dob.day}{dob.year}"))

    p1 = _reduce(m + d)
    p2 = _reduce(d + y)
    p3 = _reduce(p1 + p2)
    p4 = _reduce(m + y)

    # Challenges are conventionally reported as plain 0-8 numbers, not Master
    # Numbers, even though the m/d/y they're built from can be — see module
    # docstring on the "different schools calculate this differently" point.
    c1 = _reduce(abs(m - d), keep_master=False)
    c2 = _reduce(abs(d - y), keep_master=False)
    c3 = _reduce(abs(c1 - c2), keep_master=False)
    c4 = _reduce(abs(m - y), keep_master=False)

    end1 = 36 - lp
    end2 = end1 + 9
    end3 = end2 + 9

    ranges = [f"Birth–{end1}", f"{end1 + 1}–{end2}", f"{end2 + 1}–{end3}", f"{end3 + 1}+"]
    pinnacles = [
        {"period": i + 1, "number": val, "meaning": PINNACLE_MEANINGS.get(val, ""),
         "is_master": val in MASTER_NUMBERS, "age_range": ranges[i]}
        for i, val in enumerate([p1, p2, p3, p4])
    ]
    challenges = [
        {"period": i + 1, "number": val, "meaning": CHALLENGE_MEANINGS.get(val, ""), "age_range": ranges[i]}
        for i, val in enumerate([c1, c2, c3, c4])
    ]
    return {"pinnacles": pinnacles, "challenges": challenges, "life_path_used_for_timing": lp}


def current_cycle_index(dob: date, as_of: date | None = None) -> int:
    """Which of the 4 Pinnacle/Challenge periods (0-3) `as_of` falls in."""
    as_of = as_of or date.today()
    age = (as_of - dob).days / 365.25
    lp = _reduce(sum(int(c) for c in f"{dob.month}{dob.day}{dob.year}"))
    end1 = 36 - lp
    end2 = end1 + 9
    end3 = end2 + 9
    if age <= end1:
        return 0
    if age <= end2:
        return 1
    if age <= end3:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Birth name vs. current name comparison
# ---------------------------------------------------------------------------

def compare_names(birth_name: str, current_name: str, system: str = "pythagorean") -> dict:
    """Destiny/Soul Urge/Personality for both names side by side. Returns
    None-shaped 'current' fields cleanly if the two names are the same
    (nothing to compare) rather than presenting a misleading diff."""
    same = clean_letters(birth_name) == clean_letters(current_name)
    result = {
        "same_name": same,
        "birth_name": {
            "destiny": _destiny_detailed(birth_name, system),
            "soul_urge": _soul_urge_detailed(birth_name, system),
            "personality": _personality_detailed(birth_name, system),
        },
    }
    if same:
        result["current_name"] = None
    else:
        result["current_name"] = {
            "destiny": _destiny_detailed(current_name, system),
            "soul_urge": _soul_urge_detailed(current_name, system),
            "personality": _personality_detailed(current_name, system),
        }
    return result
