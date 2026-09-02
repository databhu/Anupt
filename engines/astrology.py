"""
ANUPT — Vedic Astrology Engine
Deterministic calculations via the Swiss Ephemeris (pyswisseph).
Sidereal zodiac, Lahiri ayanamsa (the standard Vedic ayanamsa).
No AI involved in calculation — only in later interpretation.
"""

from datetime import datetime, date, time
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
}

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
