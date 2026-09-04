"""
ANUPT — Vedic Astrology Engine
Deterministic calculations via the Swiss Ephemeris (pyswisseph).
Sidereal zodiac, Lahiri ayanamsa (the standard Vedic ayanamsa).
No AI involved in calculation — only in later interpretation.
"""

from datetime import datetime, date, time, timedelta
import swisseph as swe

swe.set_sid_mode(swe.SIDM_LAHIRI)

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]

PLANETS = {
    "Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS, "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER, "Venus": swe.VENUS, "Saturn": swe.SATURN,
    "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE, "Pluto": swe.PLUTO,
}
# Chiron needs an extra asteroid ephemeris file (seas_18.se1) that isn't bundled
# with pyswisseph and won't exist on a fresh deploy. Rather than silently omit
# it or crash, compute_chart() tries it and marks it unavailable if the file
# isn't there — matching this app's existing "don't fabricate precision we
# don't have" stance (see engines/palmistry.py) rather than pretending.
CHIRON_CODE = swe.CHIRON

# Vimshottari Dasha sequence: (lord, years). Total = 120 years.
DASHA_SEQUENCE = [
    ("Ketu", 7), ("Venus", 20), ("Sun", 6), ("Moon", 10), ("Mars", 7),
    ("Rahu", 18), ("Jupiter", 16), ("Saturn", 19), ("Mercury", 17),
]
DASHA_TOTAL_YEARS = sum(y for _, y in DASHA_SEQUENCE)

HOUSE_THEMES = {
    1: "self, vitality, personality", 2: "money, family, speech",
    3: "courage, siblings, effort", 4: "home, mother, inner peace",
    5: "creativity, children, intelligence", 6: "health, obstacles, service",
    7: "partnership, marriage, business", 8: "transformation, longevity, the hidden",
    9: "fortune, dharma, higher learning", 10: "career, status, public life",
    11: "gains, aspirations, community", 12: "loss, release, spirituality",
}


HOUSE_CLASSIFICATION = {
    1: "Angular", 4: "Angular", 7: "Angular", 10: "Angular",
    2: "Succedent", 5: "Succedent", 8: "Succedent", 11: "Succedent",
    3: "Cadent", 6: "Cadent", 9: "Cadent", 12: "Cadent",
}

# ---------------------------------------------------------------------------
# Zodiac sign metadata — element, modality, polarity, and dignity tables.
# Rulership and Exaltation are the two tables actually hand-authored;
# Detriment and Fall are DERIVED from them (opposite sign, i.e. +6 signs),
# which is both the correct traditional definition and guarantees the four
# tables can never drift out of sync with each other the way four
# independently hand-typed tables could.
# ---------------------------------------------------------------------------

SIGN_ELEMENT = {
    "Aries": "Fire", "Leo": "Fire", "Sagittarius": "Fire",
    "Taurus": "Earth", "Virgo": "Earth", "Capricorn": "Earth",
    "Gemini": "Air", "Libra": "Air", "Aquarius": "Air",
    "Cancer": "Water", "Scorpio": "Water", "Pisces": "Water",
}
SIGN_MODALITY = {
    "Aries": "Cardinal", "Cancer": "Cardinal", "Libra": "Cardinal", "Capricorn": "Cardinal",
    "Taurus": "Fixed", "Leo": "Fixed", "Scorpio": "Fixed", "Aquarius": "Fixed",
    "Gemini": "Mutable", "Virgo": "Mutable", "Sagittarius": "Mutable", "Pisces": "Mutable",
}
SIGN_POLARITY = {s: ("Masculine" if i % 2 == 0 else "Feminine") for i, s in enumerate(SIGNS)}

# Traditional (classical 7-planet) rulership — the primary table dignity is
# derived from. Modern astrology also assigns Uranus/Neptune/Pluto as
# co-rulers of Aquarius/Pisces/Scorpio; those are listed separately since
# using them as the PRIMARY dignity reference is a modern convention some
# astrologers don't follow, not a settled replacement for the classical one.
SIGN_RULER_TRADITIONAL = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter",
}
SIGN_RULER_MODERN_CORULER = {"Scorpio": "Pluto", "Aquarius": "Uranus", "Pisces": "Neptune"}

# Exaltation sign per planet. The classical 7 are well-established and
# consistent across sources; outer-planet exaltations (Uranus/Neptune/Pluto)
# are modern proposals with real disagreement between astrologers — included
# for completeness but flagged as less standardized wherever they're shown.
PLANET_EXALTATION_SIGN = {
    "Sun": "Aries", "Moon": "Taurus", "Mercury": "Virgo", "Venus": "Pisces",
    "Mars": "Capricorn", "Jupiter": "Cancer", "Saturn": "Libra",
    "Uranus": "Scorpio", "Neptune": "Cancer", "Pluto": "Leo",
}
OUTER_PLANET_DIGNITY_CAVEAT = {"Uranus", "Neptune", "Pluto"}

# Vedic exaltation/debilitation degree (the single exact degree of maximum
# strength/weakness within the sign — a refinement classical Western
# astrology doesn't use but Vedic dignity analysis does).
VEDIC_EXALTATION_DEGREE = {
    "Sun": 10.0, "Moon": 3.0, "Mars": 28.0, "Mercury": 15.0,
    "Jupiter": 5.0, "Venus": 27.0, "Saturn": 20.0,
}
# Moolatrikona: a sign-and-degree-range each classical planet is "almost
# exalted" in — degree ranges as commonly cited; a few traditions draw the
# exact boundaries slightly differently.
MOOLATRIKONA = {
    "Sun": ("Leo", 0.0, 20.0), "Moon": ("Taurus", 4.0, 30.0), "Mars": ("Aries", 0.0, 12.0),
    "Mercury": ("Virgo", 16.0, 20.0), "Jupiter": ("Sagittarius", 0.0, 10.0),
    "Venus": ("Libra", 0.0, 15.0), "Saturn": ("Aquarius", 0.0, 20.0),
}


def _opposite_sign(sign: str) -> str:
    return SIGNS[(SIGNS.index(sign) + 6) % 12]


def sign_detriment_ruler(sign: str) -> str | None:
    """The planet in detriment when placed in `sign` — the ruler of the
    opposite sign, by definition. SIGN_RULER_TRADITIONAL is keyed by sign,
    so this is a direct lookup, not a search."""
    return SIGN_RULER_TRADITIONAL.get(_opposite_sign(sign))


def western_dignity(planet: str, sign: str) -> str:
    """Exalted / Rulership / Detriment / Fall / Detriment & Fall / Neutral —
    the classical Western dignity, derived consistently from the same two
    hand-authored tables (rulership, exaltation) rather than four separate
    ones that could disagree with each other.

    "Detriment & Fall" is a real, independently-documented compound case,
    not a bug: Mercury is the one planet whose exaltation sign (Virgo) is
    also one of its own rulership signs, so Pisces (opposite Virgo) is
    simultaneously opposite its domicile AND opposite its exaltation —
    standard dignity references list Mercury/Pisces as both at once rather
    than picking one, and this does the same instead of silently dropping
    whichever check runs second."""
    if PLANET_EXALTATION_SIGN.get(planet) == sign:
        return "Exalted"
    if SIGN_RULER_TRADITIONAL.get(sign) == planet:
        return "Rulership"

    exalted_sign = PLANET_EXALTATION_SIGN.get(planet)
    is_fall = bool(exalted_sign and _opposite_sign(exalted_sign) == sign)
    is_detriment = sign_detriment_ruler(sign) == planet
    if is_fall and is_detriment:
        return "Detriment & Fall"
    if is_fall:
        return "Fall"
    if is_detriment:
        return "Detriment"
    return "Neutral"


def vedic_dignity(planet: str, sign: str, degree_in_sign: float | None = None) -> str:
    """Own Sign / Exalted / Debilitated / Moolatrikona / Neutral — the Vedic
    dignity states, for the seven classical grahas (Rahu/Ketu and the outer
    planets don't carry traditional Vedic dignity and return 'Neutral')."""
    if planet not in VEDIC_EXALTATION_DEGREE:
        return "Neutral"
    if PLANET_EXALTATION_SIGN.get(planet) == sign:
        return "Exalted"
    exalted_sign = PLANET_EXALTATION_SIGN[planet]
    if _opposite_sign(exalted_sign) == sign:
        return "Debilitated"
    if degree_in_sign is not None and planet in MOOLATRIKONA:
        mt_sign, lo, hi = MOOLATRIKONA[planet]
        if sign == mt_sign and lo <= degree_in_sign <= hi:
            return "Moolatrikona"
    if SIGN_RULER_TRADITIONAL.get(sign) == planet:
        return "Own Sign"
    return "Neutral"


# ---------------------------------------------------------------------------
# Aspects — pure angular geometry between any set of longitudes, usable on
# either the Vedic sidereal chart or the Western tropical one, since an
# aspect is just the angular distance between two points regardless of
# which zodiac frame produced them.
# ---------------------------------------------------------------------------

MAJOR_ASPECTS = {
    "Conjunction": (0, 8), "Sextile": (60, 6), "Square": (90, 8),
    "Trine": (120, 8), "Opposition": (180, 8),
}
MINOR_ASPECTS = {
    "Semi-sextile": (30, 2), "Semi-square": (45, 2),
    "Quincunx": (150, 3), "Sesquiquadrate": (135, 2),
}


def angular_distance(lon_a: float, lon_b: float) -> float:
    """Shortest angular distance between two ecliptic longitudes, 0-180."""
    diff = abs(lon_a - lon_b) % 360
    return diff if diff <= 180 else 360 - diff


def calculate_aspects(longitudes: dict, orbs: dict | None = None, include_minor: bool = True) -> list:
    """`longitudes` is {point_name: longitude_degrees}. Returns every aspect
    found between each pair, closest-to-exact first. `orbs` lets a caller
    override the default orb for any aspect name — e.g. {"Conjunction": 10}
    for a wider conjunction orb — falling back to the documented defaults
    for anything not overridden."""
    aspect_defs = dict(MAJOR_ASPECTS)
    if include_minor:
        aspect_defs.update(MINOR_ASPECTS)
    if orbs:
        aspect_defs = {name: (angle, orbs.get(name, default_orb))
                       for name, (angle, default_orb) in aspect_defs.items()}

    names = list(longitudes.keys())
    found = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            dist = angular_distance(longitudes[a], longitudes[b])
            for aspect_name, (target_angle, orb) in aspect_defs.items():
                delta = abs(dist - target_angle)
                if delta <= orb:
                    found.append({
                        "point_a": a, "point_b": b, "aspect": aspect_name,
                        "exact_angle": target_angle, "actual_angle": round(dist, 2),
                        "orb": round(delta, 2),
                    })
                    break  # a pair gets at most one aspect — the first (closest) match
    found.sort(key=lambda x: x["orb"])
    return found


def _julian_day_ut(birth_dt_local: datetime, utc_offset_hours: float) -> float:
    dt_utc = birth_dt_local
    ut_hour = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600 - utc_offset_hours
    jd = swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, ut_hour)
    return jd


def _sign_of(longitude: float) -> tuple[str, float]:
    idx = int(longitude // 30) % 12
    deg_in_sign = longitude % 30
    return SIGNS[idx], deg_in_sign


def _nakshatra_of(moon_longitude: float) -> tuple[str, int]:
    span = 360 / 27  # 13°20'
    idx = int(moon_longitude // span) % 27
    pada = int((moon_longitude % span) // (span / 4)) + 1
    return NAKSHATRAS[idx], pada


def _whole_sign_house(planet_sign_idx: int, asc_sign_idx: int) -> int:
    return ((planet_sign_idx - asc_sign_idx) % 12) + 1


def _vimshottari_dasha(moon_longitude: float, birth_date: date, as_of: date) -> dict:
    """Compute current Mahadasha & Antardasha as of `as_of`, given Moon's sidereal longitude."""
    span = 360 / 27
    nak_idx = int(moon_longitude // span) % 27
    frac_elapsed = (moon_longitude % span) / span  # how far through the nakshatra Moon is

    start_lord_idx = nak_idx % 9  # nakshatra lord cycles every 9, matching dasha sequence order
    lord, total_years = DASHA_SEQUENCE[start_lord_idx]
    balance_years = total_years * (1 - frac_elapsed)

    # Build the Mahadasha timeline starting from birth
    periods = []
    cursor_year = birth_date.year + (birth_date.timetuple().tm_yday / 365.25)
    # first period is the balance of the birth nakshatra lord
    periods.append((lord, cursor_year, cursor_year + balance_years))
    cursor_year += balance_years
    i = (start_lord_idx + 1) % 9
    for _ in range(15):  # enough cycles to comfortably cover a lifetime
        l, y = DASHA_SEQUENCE[i]
        periods.append((l, cursor_year, cursor_year + y))
        cursor_year += y
        i = (i + 1) % 9

    as_of_year = as_of.year + (as_of.timetuple().tm_yday / 365.25)
    current_maha = next((p for p in periods if p[1] <= as_of_year < p[2]), periods[-1])

    # Antardasha: subdivide the Mahadasha span proportionally by the same 9-lord sequence,
    # starting from the Mahadasha's own lord.
    maha_lord, maha_start, maha_end = current_maha
    maha_span = maha_end - maha_start
    maha_lord_idx = [l for l, _ in DASHA_SEQUENCE].index(maha_lord)
    sub_cursor = maha_start
    antar = None
    for j in range(9):
        l, y = DASHA_SEQUENCE[(maha_lord_idx + j) % 9]
        sub_years = maha_span * (y / DASHA_TOTAL_YEARS)
        if sub_cursor <= as_of_year < sub_cursor + sub_years or j == 8:
            antar = l
            break
        sub_cursor += sub_years

    return {
        "mahadasha": maha_lord,
        "mahadasha_start_year": round(maha_start, 1),
        "mahadasha_end_year": round(maha_end, 1),
        "antardasha": antar,
    }


def compute_chart(full_name: str, birth_date: date, birth_time: time,
                   latitude: float, longitude: float, utc_offset_hours: float,
                   as_of: date | None = None) -> dict:
    """
    Returns a fully structured Vedic chart — JSON-serializable, no free text interpretation.
    """
    as_of = as_of or date.today()
    birth_dt = datetime.combine(birth_date, birth_time)
    jd_ut = _julian_day_ut(birth_dt, utc_offset_hours)

    flags = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

    planet_positions = {}
    for name, code in PLANETS.items():
        result = swe.calc_ut(jd_ut, code, flags)
        lon = result[0][0]
        speed = result[0][3]
        sign, deg = _sign_of(lon)
        planet_positions[name] = {
            "longitude": round(lon, 4), "sign": sign,
            "degree_in_sign": round(deg, 2), "retrograde": speed < 0,
        }

    # Rahu (mean lunar node) and Ketu (180° opposite)
    rahu_result = swe.calc_ut(jd_ut, swe.MEAN_NODE, flags)
    rahu_lon = rahu_result[0][0]
    ketu_lon = (rahu_lon + 180) % 360
    for name, lon in (("Rahu", rahu_lon), ("Ketu", ketu_lon)):
        sign, deg = _sign_of(lon)
        planet_positions[name] = {
            "longitude": round(lon, 4), "sign": sign,
            "degree_in_sign": round(deg, 2), "retrograde": True,
        }

    # Ascendant (Lagna) — sidereal
    cusps, ascmc = swe.houses_ex(jd_ut, latitude, longitude, b'W', flags)
    asc_lon = ascmc[0]
    asc_sign, asc_deg = _sign_of(asc_lon)
    asc_sign_idx = SIGNS.index(asc_sign)

    # Whole-sign houses for every planet
    for name, data in planet_positions.items():
        p_sign_idx = SIGNS.index(data["sign"])
        house = _whole_sign_house(p_sign_idx, asc_sign_idx)
        data["house"] = house
        data["house_theme"] = HOUSE_THEMES[house]

    nakshatra, pada = _nakshatra_of(planet_positions["Moon"]["longitude"])
    dasha = _vimshottari_dasha(planet_positions["Moon"]["longitude"], birth_date, as_of)

    return {
        "ascendant": {"sign": asc_sign, "degree_in_sign": round(asc_deg, 2)},
        "moon_sign": planet_positions["Moon"]["sign"],
        "sun_sign": planet_positions["Sun"]["sign"],
        "nakshatra": nakshatra,
        "nakshatra_pada": pada,
        "planets": planet_positions,
        "dasha": dasha,
        "career_indicator_house": 10,
        "career_house_sign": SIGNS[(asc_sign_idx + 9) % 12],
        "finance_indicator_house": 2,
        "finance_house_sign": SIGNS[(asc_sign_idx + 1) % 12],
        "relationship_indicator_house": 7,
        "relationship_house_sign": SIGNS[(asc_sign_idx + 6) % 12],
    }


# ---------------------------------------------------------------------------
# Western tropical chart — a genuinely separate calculation from
# compute_chart() above, not a variant of it: Western astrology conventionally
# uses the TROPICAL zodiac (seasons-anchored) with a selectable house system,
# where Vedic uses the SIDEREAL zodiac (star-anchored, Lahiri ayanamsa) with
# whole-sign houses. Mixing the two frames would produce a chart that's
# neither tradition's actual practice, so this keeps its own house-cusp math,
# its own dignity lookups (Western 5-state, not Vedic 4-state), and its own
# return shape entirely — nothing here overlaps with or changes what
# compute_chart() returns.
# ---------------------------------------------------------------------------

WESTERN_HOUSE_SYSTEMS = {"Placidus": b'P', "Equal": b'E', "Whole Sign": b'W', "Koch": b'K'}


def _house_for_longitude(lon: float, cusps: list) -> int:
    """Which of 12 house cusps `lon` falls after, handling the 360°->0°
    wraparound a house near the Aries point can create."""
    lon = lon % 360
    for i in range(12):
        start, end = cusps[i] % 360, cusps[(i + 1) % 12] % 360
        if start < end:
            if start <= lon < end:
                return i + 1
        else:  # this house's span crosses the 0°/360° boundary
            if lon >= start or lon < end:
                return i + 1
    return 12  # defensive fallback; shouldn't be reachable with valid cusps


def compute_western_chart(birth_date: date, birth_time: time, latitude: float, longitude: float,
                           utc_offset_hours: float, house_system: str = "Placidus",
                           orbs: dict | None = None, include_minor_aspects: bool = True) -> dict:
    """Tropical Western natal chart: planets, houses (selectable system),
    the four angles, aspects, and Western dignity for each planet."""
    birth_dt = datetime.combine(birth_date, birth_time)
    jd_ut = _julian_day_ut(birth_dt, utc_offset_hours)
    flags = swe.FLG_SWIEPH  # no FLG_SIDEREAL — tropical

    planet_positions = {}
    for name, code in PLANETS.items():
        result = swe.calc_ut(jd_ut, code, flags)
        lon, speed = result[0][0], result[0][3]
        sign, deg = _sign_of(lon)
        planet_positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2),
            "retrograde": speed < 0,
        }
    rahu_lon = swe.calc_ut(jd_ut, swe.MEAN_NODE, flags)[0][0]
    ketu_lon = (rahu_lon + 180) % 360
    for name, lon in (("Rahu", rahu_lon), ("Ketu", ketu_lon)):
        sign, deg = _sign_of(lon)
        planet_positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2), "retrograde": True,
        }

    hs_code = WESTERN_HOUSE_SYSTEMS.get(house_system, b'P')
    cusps, ascmc = swe.houses_ex(jd_ut, latitude, longitude, hs_code, flags)
    cusps = list(cusps)
    asc_lon, mc_lon = ascmc[0], ascmc[1]
    dsc_lon, ic_lon = (asc_lon + 180) % 360, (mc_lon + 180) % 360

    for name, data in planet_positions.items():
        house = _house_for_longitude(data["longitude"], cusps)
        data["house"] = house
        data["house_classification"] = HOUSE_CLASSIFICATION[house]
        data["dignity"] = western_dignity(name, data["sign"])

    def _point(lon):
        sign, deg = _sign_of(lon)
        return {"sign": sign, "degree_in_sign": round(deg, 2), "longitude": round(lon, 4)}

    longitudes_for_aspects = {name: d["longitude"] for name, d in planet_positions.items()}
    aspects = calculate_aspects(longitudes_for_aspects, orbs=orbs, include_minor=include_minor_aspects)

    return {
        "house_system": house_system,
        "ascendant": _point(asc_lon),
        "descendant": _point(dsc_lon),
        "midheaven": _point(mc_lon),
        "ic": _point(ic_lon),
        "house_cusps": [round(c, 4) for c in cusps],
        "planets": planet_positions,
        "aspects": aspects,
    }


# ---------------------------------------------------------------------------
# Vedic divisional charts (Vargas) — D9 Navamsha and D10 Dashamsha, computed
# from the sidereal (Vedic) longitude compute_chart() already produced.
# These operate on a longitude directly rather than needing a full chart
# object, so the same function works for any planet or the Ascendant alike.
# ---------------------------------------------------------------------------

# Navamsha (D9): each sign splits into 9 parts of 3°20'. Which sign the
# 9-part sequence STARTS from depends on the natal sign's element —
# fire signs start counting from Aries, earth from Capricorn, air from
# Libra, water from Cancer. This is the standard, widely-used method.
_NAVAMSHA_START_BY_ELEMENT = {"Fire": "Aries", "Earth": "Capricorn", "Air": "Libra", "Water": "Cancer"}


def navamsha_sign(longitude: float) -> str:
    """D9 sign for a sidereal longitude."""
    sign, deg_in_sign = _sign_of(longitude)
    element = SIGN_ELEMENT[sign]
    start_idx = SIGNS.index(_NAVAMSHA_START_BY_ELEMENT[element])
    segment = int(deg_in_sign // (30 / 9))
    return SIGNS[(start_idx + segment) % 12]


def dashamsha_sign(longitude: float) -> str:
    """D10 sign for a sidereal longitude. Odd signs count their 10 divisions
    starting from themselves; even signs start from the 9th sign counting
    inclusively (i.e. +8 in a 0-indexed sign list) — the standard method."""
    sign, deg_in_sign = _sign_of(longitude)
    sign_idx = SIGNS.index(sign)
    is_odd = (sign_idx % 2) == 0  # SIGNS[0]=Aries is the 1st sign, i.e. odd
    start_idx = sign_idx if is_odd else (sign_idx + 8) % 12
    segment = int(deg_in_sign // 3.0)
    return SIGNS[(start_idx + segment) % 12]


def divisional_chart(chart: dict, varga_fn) -> dict:
    """Applies a varga function (navamsha_sign or dashamsha_sign) to every
    planet plus the Ascendant in an existing compute_chart() result, without
    recomputing any ephemeris positions — vargas are a pure re-mapping of
    longitudes already calculated, not a new astronomical calculation."""
    asc_lon = None
    # compute_chart() doesn't currently store the Ascendant's raw longitude
    # (only sign + degree-in-sign), so reconstruct it from those two —
    # exact, since sign+degree fully determines a longitude.
    asc_sign_idx = SIGNS.index(chart["ascendant"]["sign"])
    asc_lon = asc_sign_idx * 30 + chart["ascendant"]["degree_in_sign"]

    result = {"ascendant": varga_fn(asc_lon), "planets": {}}
    for name, data in chart["planets"].items():
        result["planets"][name] = varga_fn(data["longitude"])
    return result


# ---------------------------------------------------------------------------
# Transits — where the sky is *right now* (or any given date), and which of
# those current positions form an aspect back to a natal placement. This is
# the standard "what's active for me at the moment" predictive technique,
# distinct from the Dasha timeline (which is about ruling periods, not
# moment-to-moment planetary aspects).
# ---------------------------------------------------------------------------

def compute_transits(natal_chart: dict, as_of: date | None = None) -> dict:
    """Current sidereal planetary positions (same frame as the natal chart,
    for a direct comparison) plus every aspect they form back to a natal
    placement. Uses noon UT on `as_of` — daily resolution, which is what a
    'what's active this week/month' reading needs; it isn't trying to be
    precise to the minute the way a natal chart's own Ascendant must be."""
    as_of = as_of or date.today()
    jd_ut = swe.julday(as_of.year, as_of.month, as_of.day, 12.0)
    flags = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

    transit_positions = {}
    for name, code in PLANETS.items():
        result = swe.calc_ut(jd_ut, code, flags)
        lon, speed = result[0][0], result[0][3]
        sign, deg = _sign_of(lon)
        transit_positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2), "retrograde": speed < 0,
        }
    rahu_lon = swe.calc_ut(jd_ut, swe.MEAN_NODE, flags)[0][0]
    ketu_lon = (rahu_lon + 180) % 360
    for name, lon in (("Rahu", rahu_lon), ("Ketu", ketu_lon)):
        sign, deg = _sign_of(lon)
        transit_positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2), "retrograde": True,
        }

    natal_longs = {f"natal_{name}": d["longitude"] for name, d in natal_chart["planets"].items()}
    transit_longs = {f"transit_{name}": d["longitude"] for name, d in transit_positions.items()}
    all_aspects = calculate_aspects({**natal_longs, **transit_longs}, include_minor=False)
    cross_aspects = [
        asp for asp in all_aspects
        if asp["point_a"].startswith("transit_") != asp["point_b"].startswith("transit_")
    ]
    for asp in cross_aspects:
        asp["natal_point"] = (asp["point_a"] if asp["point_a"].startswith("natal_") else asp["point_b"]).removeprefix("natal_")
        asp["transiting_point"] = (asp["point_a"] if asp["point_a"].startswith("transit_") else asp["point_b"]).removeprefix("transit_")

    return {"as_of": as_of.isoformat(), "transit_planets": transit_positions, "aspects_to_natal": cross_aspects}


# ---------------------------------------------------------------------------
# Synastry — inter-aspects between two people's natal planets. This is the
# aspect-grid technique, not a composite/Davison chart (a genuinely
# different, more involved calculation this doesn't attempt).
# ---------------------------------------------------------------------------

def compute_synastry(chart_a: dict, chart_b: dict, include_minor: bool = False) -> dict:
    longs_a = {f"A_{name}": d["longitude"] for name, d in chart_a["planets"].items()}
    longs_b = {f"B_{name}": d["longitude"] for name, d in chart_b["planets"].items()}
    all_aspects = calculate_aspects({**longs_a, **longs_b}, include_minor=include_minor)
    cross_aspects = [
        asp for asp in all_aspects
        if asp["point_a"].startswith("A_") != asp["point_b"].startswith("A_")
    ]
    for asp in cross_aspects:
        asp["person_a_point"] = (asp["point_a"] if asp["point_a"].startswith("A_") else asp["point_b"]).removeprefix("A_")
        asp["person_b_point"] = (asp["point_a"] if asp["point_a"].startswith("B_") else asp["point_b"]).removeprefix("B_")
    return {"aspects": cross_aspects}


# ---------------------------------------------------------------------------
# House lords & planetary strength — the building blocks Yoga detection
# needs, and useful on their own for a chart summary.
# ---------------------------------------------------------------------------

def house_lords(chart: dict) -> dict:
    """Which planet rules each whole-sign house's occupying sign, for this
    specific chart. {1: "Mars", 2: "Venus", ...} — house number to lord."""
    asc_sign_idx = SIGNS.index(chart["ascendant"]["sign"])
    lords = {}
    for house in range(1, 13):
        sign = SIGNS[(asc_sign_idx + house - 1) % 12]
        lords[house] = SIGN_RULER_TRADITIONAL[sign]
    return lords


STRENGTH_BY_DIGNITY = {
    "Exalted": "Strong", "Own Sign": "Strong", "Moolatrikona": "Strong",
    "Debilitated": "Weak", "Neutral": "Moderate",
}


def planetary_strength_summary(chart: dict) -> dict:
    """A simplified, dignity-based strength read for each classical planet —
    NOT the full classical Shadbala (six-fold strength) system, which
    combines positional, directional, temporal, motional and several other
    factors into a numeric score. That's a much larger calculation this
    doesn't attempt; this gives an honest, immediately-useful 'strong /
    moderate / weak' read from dignity alone, clearly labeled as such."""
    result = {}
    for name, data in chart["planets"].items():
        dignity = vedic_dignity(name, data["sign"], data["degree_in_sign"])
        result[name] = {"dignity": dignity, "strength": STRENGTH_BY_DIGNITY.get(dignity, "Moderate")}
    return result


# ---------------------------------------------------------------------------
# Yogas — a deliberately small set of well-documented, clearly-defined
# combinations, not an exhaustive classical catalogue (there are hundreds
# of named yogas across different texts, with real variation in exact
# criteria between traditions). Each one implemented here uses a specific,
# commonly-cited simplified definition, stated plainly in its docstring
# rather than left ambiguous.
# ---------------------------------------------------------------------------

def detect_yogas(chart: dict) -> list:
    """Returns every yoga (from the small set below) actually present in
    this chart, each as {"name", "description", "planets_involved"}."""
    found = []
    planets = chart["planets"]
    lords = house_lords(chart)

    # Gaja Kesari Yoga: Jupiter sits in a kendra (angular house — 1st/4th/
    # 7th/10th) counted FROM the Moon's own house.
    if "Moon" in planets and "Jupiter" in planets:
        house_diff = (planets["Jupiter"]["house"] - planets["Moon"]["house"]) % 12
        if house_diff in (0, 3, 6, 9):
            found.append({
                "name": "Gaja Kesari Yoga",
                "description": "Jupiter stands in a kendra from the Moon — a classic combination "
                               "for steady reputation, wisdom, and resilience.",
                "planets_involved": ["Moon", "Jupiter"],
            })

    # Budhaditya Yoga: Sun and Mercury conjunct in the same sign.
    if "Sun" in planets and "Mercury" in planets and planets["Sun"]["sign"] == planets["Mercury"]["sign"]:
        found.append({
            "name": "Budhaditya Yoga",
            "description": "Sun and Mercury share a sign — associated with sharp intellect "
                           "and communication ability.",
            "planets_involved": ["Sun", "Mercury"],
        })

    # Chandra-Mangal Yoga: Moon and Mars conjunct in the same sign.
    if "Moon" in planets and "Mars" in planets and planets["Moon"]["sign"] == planets["Mars"]["sign"]:
        found.append({
            "name": "Chandra-Mangal Yoga",
            "description": "Moon and Mars share a sign — traditionally linked to drive, "
                           "resourcefulness, and wealth-building energy.",
            "planets_involved": ["Moon", "Mars"],
        })

    # Kendra-Trikona Raja Yoga (simplified): the lord of a kendra house
    # (1/4/7/10) and the lord of a trikona house (1/5/9) are either
    # conjunct (same sign) or in mutual sign-exchange (Parivartana) with
    # each other. House 1 counts as both kendra and trikona; that overlap
    # is intentional (it's the classical exception).
    kendra_houses, trikona_houses = (1, 4, 7, 10), (1, 5, 9)
    for kh in kendra_houses:
        for th in trikona_houses:
            if kh == th:
                continue
            kl, tl = lords[kh], lords[th]
            if kl == tl:
                continue
            if kl not in planets or tl not in planets:
                continue
            same_sign = planets[kl]["sign"] == planets[tl]["sign"]
            kl_in_tl_sign = SIGN_RULER_TRADITIONAL.get(planets[kl]["sign"]) == tl
            tl_in_kl_sign = SIGN_RULER_TRADITIONAL.get(planets[tl]["sign"]) == kl
            if same_sign or (kl_in_tl_sign and tl_in_kl_sign):
                label = "conjunct" if same_sign else "in mutual sign-exchange"
                entry_key = tuple(sorted([kl, tl]))
                if not any(y.get("_key") == entry_key for y in found):
                    found.append({
                        "name": "Raja Yoga",
                        "description": f"The {kh}th house lord ({kl}) and {th}th house lord ({tl}) "
                                       f"are {label} — a classical combination for status and success.",
                        "planets_involved": [kl, tl],
                        "_key": entry_key,
                    })

    # Dhana Yoga (simplified): 2nd house lord (wealth) and 11th house lord
    # (gains) conjunct or in mutual sign-exchange.
    l2, l11 = lords[2], lords[11]
    if l2 != l11 and l2 in planets and l11 in planets:
        same_sign = planets[l2]["sign"] == planets[l11]["sign"]
        l2_in_l11_sign = SIGN_RULER_TRADITIONAL.get(planets[l2]["sign"]) == l11
        l11_in_l2_sign = SIGN_RULER_TRADITIONAL.get(planets[l11]["sign"]) == l2
        if same_sign or (l2_in_l11_sign and l11_in_l2_sign):
            label = "conjunct" if same_sign else "in mutual sign-exchange"
            found.append({
                "name": "Dhana Yoga",
                "description": f"The 2nd house lord ({l2}) and 11th house lord ({l11}) are {label} "
                               "— a classical combination for financial gain.",
                "planets_involved": [l2, l11],
            })

    for y in found:
        y.pop("_key", None)
    return found


# ---------------------------------------------------------------------------
# Predictive techniques beyond Dasha/Transits: Secondary Progressions and
# Solar Return — both operate in the same sidereal frame as compute_chart()
# for direct comparison against natal positions.
# ---------------------------------------------------------------------------

def secondary_progressions(birth_date: date, birth_time: time, latitude: float, longitude: float,
                            utc_offset_hours: float, as_of: date | None = None) -> dict:
    """The 'a day for a year' technique: someone who is N years old has their
    progressed positions calculated for the Nth day after their birth date,
    at the same birth time/location — only the date advances, which is the
    standard method (not a variant with progressed time or relocated
    houses, which some practitioners use instead)."""
    as_of = as_of or date.today()
    age_years = (as_of - birth_date).days / 365.25
    progressed_date = birth_date + timedelta(days=age_years)

    birth_dt = datetime.combine(progressed_date, birth_time)
    jd_ut = _julian_day_ut(birth_dt, utc_offset_hours)
    flags = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

    positions = {}
    for name, code in PLANETS.items():
        result = swe.calc_ut(jd_ut, code, flags)
        lon, speed = result[0][0], result[0][3]
        sign, deg = _sign_of(lon)
        positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2), "retrograde": speed < 0,
        }
    rahu_lon = swe.calc_ut(jd_ut, swe.MEAN_NODE, flags)[0][0]
    ketu_lon = (rahu_lon + 180) % 360
    for name, lon in (("Rahu", rahu_lon), ("Ketu", ketu_lon)):
        sign, deg = _sign_of(lon)
        positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2), "retrograde": True,
        }

    return {"progressed_date": progressed_date.isoformat(), "age_years": round(age_years, 2), "planets": positions}


def solar_return_chart(natal_chart: dict, birth_date: date, target_year: int,
                        latitude: float, longitude: float) -> dict:
    """The moment in `target_year` the transiting Sun returns to the exact
    sidereal degree it held at birth, computed for the given location
    (conventionally where the person currently lives, which may differ
    from their birthplace — both are passed in, not assumed to match).
    Uses a short numerical search (the Sun moves at a very predictable
    ~0.9856°/day, so this converges in a handful of iterations) rather than
    a closed-form solution, then builds a full chart — planets, Ascendant,
    whole-sign houses — for that exact moment."""
    natal_sun_lon = natal_chart["planets"]["Sun"]["longitude"]
    flags = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

    try:
        jd = swe.julday(target_year, birth_date.month, birth_date.day, 12.0)
    except Exception:
        jd = swe.julday(target_year, birth_date.month, 28, 12.0)  # Feb 29 safety

    for _ in range(20):
        current_lon = swe.calc_ut(jd, swe.SUN, flags)[0][0]
        diff = ((natal_sun_lon - current_lon + 180) % 360) - 180  # shortest signed difference
        if abs(diff) < 0.0001:
            break
        jd += diff / 0.9856  # Sun's average daily motion; converges fast since it's near-linear

    y, m, d, ut_hour = swe.revjul(jd)
    hour = int(ut_hour)
    minute = round((ut_hour - hour) * 60)

    positions = {}
    for name, code in PLANETS.items():
        result = swe.calc_ut(jd, code, flags)
        lon, speed = result[0][0], result[0][3]
        sign, deg = _sign_of(lon)
        positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2), "retrograde": speed < 0,
        }
    rahu_lon = swe.calc_ut(jd, swe.MEAN_NODE, flags)[0][0]
    ketu_lon = (rahu_lon + 180) % 360
    for name, lon in (("Rahu", rahu_lon), ("Ketu", ketu_lon)):
        sign, deg = _sign_of(lon)
        positions[name] = {
            "longitude": round(lon, 4), "sign": sign, "degree_in_sign": round(deg, 2), "retrograde": True,
        }

    _, ascmc = swe.houses_ex(jd, latitude, longitude, b'W', flags)
    asc_sign, asc_deg = _sign_of(ascmc[0])
    asc_sign_idx = SIGNS.index(asc_sign)
    for name, data in positions.items():
        planet_sign_idx = SIGNS.index(data["sign"])
        data["house"] = ((planet_sign_idx - asc_sign_idx) % 12) + 1

    return {
        "exact_moment_utc": f"{y:04d}-{m:02d}-{d:02d} {hour:02d}:{minute:02d}",
        "ascendant": {"sign": asc_sign, "degree_in_sign": round(asc_deg, 2)},
        "planets": positions,
    }


def solar_arc_directions(natal_chart: dict, birth_date: date, birth_time: time, latitude: float,
                          longitude: float, utc_offset_hours: float, as_of: date | None = None) -> dict:
    """Solar Arc Directions: every natal planet advanced by the same arc the
    progressed Sun has moved from its own natal position. This is the
    standard modern 'Directions' technique and a well-defined single
    calculation once the progressed Sun's position is known — unlike
    Primary Directions (an older technique built on spherical-trigonometry
    oblique-ascension math that different historical sources genuinely
    compute differently), which this does not attempt, rather than
    presenting one arbitrarily-chosen version of a disputed method as
    settled fact."""
    prog = secondary_progressions(birth_date, birth_time, latitude, longitude, utc_offset_hours, as_of)
    natal_sun_lon = natal_chart["planets"]["Sun"]["longitude"]
    progressed_sun_lon = prog["planets"]["Sun"]["longitude"]
    arc = (progressed_sun_lon - natal_sun_lon) % 360

    directed = {}
    for name, data in natal_chart["planets"].items():
        directed_lon = (data["longitude"] + arc) % 360
        sign, deg = _sign_of(directed_lon)
        directed[name] = {"longitude": round(directed_lon, 4), "sign": sign, "degree_in_sign": round(deg, 2)}

    return {
        "arc_degrees": round(arc, 4),
        "as_of": (as_of or date.today()).isoformat(),
        "planets": directed,
    }
