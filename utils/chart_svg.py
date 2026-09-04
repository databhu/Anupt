"""
ANUPT — Natal Wheel renderer.

Builds the circular birth-chart wheel as raw SVG from the already-computed,
deterministic chart data (engines/astrology.py). This is informative, not
decorative: sign per house, planet placement and retrograde status are all
read directly off the chart, not invented for effect.

Convention: Ascendant at 9 o'clock, houses run counter-clockwise (so the
Midheaven/10th lands at 12 o'clock, Descendant/7th at 3 o'clock, IC/4th at
6 o'clock) — the standard Western wheel layout, applied here to whole-sign
houses.
"""

import math

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
# U+FE0E (variation selector-15) after each glyph forces the browser's *text*
# presentation instead of a colorful emoji glyph, so the wheel stays in the
# curated brass/parchment palette rather than falling back to a system emoji font.
_VS = "\uFE0E"
SIGN_GLYPH = {s: g + _VS for s, g in zip(SIGNS, "♈♉♊♋♌♍♎♏♐♑♒♓")}
PLANET_GLYPH = {
    "Sun": "☉" + _VS, "Moon": "☽" + _VS, "Mars": "♂" + _VS, "Mercury": "☿" + _VS,
    "Jupiter": "♃" + _VS, "Venus": "♀" + _VS, "Saturn": "♄" + _VS,
    "Uranus": "♅" + _VS, "Neptune": "♆" + _VS, "Pluto": "♇" + _VS,
    "Rahu": "☊" + _VS, "Ketu": "☋" + _VS, "Chiron": "⚷" + _VS,
}


def _pt(cx, cy, r, angle_deg):
    a = math.radians(angle_deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def natal_wheel_svg(chart: dict, size: int = 440) -> str:
    cx = cy = size / 2
    r_outer, r_signring, r_housenum, r_inner = size * 0.485, size * 0.40, size * 0.34, size * 0.135

    asc_sign_idx = SIGNS.index(chart["ascendant"]["sign"])

    def sign_for_house(h):
        return SIGNS[(asc_sign_idx + h - 1) % 12]

    def house_center_angle(h):
        return (180 + (h - 1) * 30) % 360

    # planets grouped by house
    by_house = {}
    for name, data in chart["planets"].items():
        by_house.setdefault(data["house"], []).append((name, data["retrograde"]))

    parts = [
        f'<svg viewBox="0 0 {size} {size}" width="100%" role="img" '
        f'aria-label="Natal wheel: Ascendant {chart["ascendant"]["sign"]}, Moon in {chart["moon_sign"]}, Sun in {chart["sun_sign"]}">',
        '<g class="anupt-wheel-spin">',
        f'<circle cx="{cx}" cy="{cy}" r="{r_outer}" class="wheel-ring-outer"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{r_signring}" class="wheel-ring-inner"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{r_inner}" class="wheel-ring-core"/>',
    ]

    # 12 sector dividers, from core to outer edge
    for h in range(1, 13):
        boundary = (house_center_angle(h) + 15) % 360
        x1, y1 = _pt(cx, cy, r_inner, boundary)
        x2, y2 = _pt(cx, cy, r_outer, boundary)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="wheel-divider"/>')

    # sign glyphs (outer ring) + house numbers (just inside sign ring)
    for h in range(1, 13):
        ang = house_center_angle(h)
        sx, sy = _pt(cx, cy, (r_outer + r_signring) / 2, ang)
        sign = sign_for_house(h)
        parts.append(f'<text x="{sx:.1f}" y="{sy:.1f}" class="wheel-sign-glyph" '
                      f'text-anchor="middle" dominant-baseline="central">{SIGN_GLYPH[sign]}</text>')
        nx, ny = _pt(cx, cy, (r_signring + r_housenum) / 2 + 6, ang)
        parts.append(f'<text x="{nx:.1f}" y="{ny:.1f}" class="wheel-house-num" '
                      f'text-anchor="middle" dominant-baseline="central">{h}</text>')

    # planets, spread within their house sector if more than one
    for h, planets in by_house.items():
        center = house_center_angle(h)
        n = len(planets)
        spread = min(9, 22 / max(n, 1))
        for i, (name, retro) in enumerate(planets):
            offset = (i - (n - 1) / 2) * spread
            ang = center + offset
            px, py = _pt(cx, cy, r_housenum * 0.62, ang)
            cls = "wheel-planet-retro" if retro else "wheel-planet"
            parts.append(f'<text x="{px:.1f}" y="{py:.1f}" class="{cls}" '
                          f'text-anchor="middle" dominant-baseline="central">{PLANET_GLYPH.get(name, "•")}'
                          f'{"℞" if retro else ""}</text>')

    # Ascendant marker
    ax1, ay1 = _pt(cx, cy, r_inner, 180)
    ax2, ay2 = _pt(cx, cy, r_outer, 180)
    parts.append(f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}" class="wheel-asc-line"/>')
    lx, ly = _pt(cx, cy, r_outer + 16, 180)
    parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="wheel-asc-label" '
                 f'text-anchor="middle" dominant-baseline="central">ASC</text>')

    # center medallion text
    parts.append(f'<text x="{cx}" y="{cy-6}" class="wheel-core-label" text-anchor="middle">{chart["nakshatra"]}</text>')
    parts.append(f'<text x="{cx}" y="{cy+14}" class="wheel-core-sub" text-anchor="middle">pada {chart["nakshatra_pada"]}</text>')

    parts.append('</g></svg>')
    return "".join(parts)
