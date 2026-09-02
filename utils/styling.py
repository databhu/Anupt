"""
ANUPT — visual identity, v2: built around the real uploaded logo.

The logo fixed the brand for real: a magenta → indigo → blue gradient
crest, gold sparkle accents, "Insights for a better you" as the tagline,
and per-discipline color coding (numerology's circle is magenta,
palmistry's is blue). This file now derives every token from that asset
instead of an invented placeholder palette.

Color (named, sampled from the logo — see assets/logo_full.png):
  ink          #120C22  — night background (kept dark; logo has a
                          transparent-safe crest so it now sits on it
                          directly, no more text-only wordmark)
  ink-panel    #1C1533  — raised surface
  ink-line     #342A54  — hairline borders on ink
  magenta      #C2157E  — gradient start (vivid, decorative use)
  indigo       #4C1F93  — gradient mid (vivid, decorative use)
  blue         #1857C4  — gradient end (vivid, decorative use)
  gold         #C99A5E  — sparkle accent from the logo (text-safe, 7.5:1)
  magenta-tint #E263A8  — lightened magenta for legible text/markers (6:1)
  parchment    #F1ECFA  — reading text on dark surfaces
  mist         #ADA1C9  — secondary/muted text

The three vivid brand hues (magenta/indigo/blue) read at 1.7–3.4:1
against the ink background — fine for large fills, borders and the
gradient logo itself, but not for text. Gold and the lightened magenta
tint carry every place text needs AA contrast; that split is deliberate,
not an oversight (see the contrast check this was built against).

Old variable names (--brass, --brass-soft, --oxblood) are kept so the
handful of inline `var(--brass-soft)` references already in app.py don't
need touching — only their values changed, to the mapping above.

Layout / restraint: the gradient itself is the signature element (it's
the actual logo now, plus the primary button and the wordmark text) —
everything else stays in flat gold/magenta-tint/indigo so the crest
stays the one bold thing per the "spend your boldness in one place" rule.
"""

import base64
from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _b64(filename: str) -> str:
    return base64.b64encode((_ASSETS_DIR / filename).read_bytes()).decode("utf-8")


_LOGO_MARK = _b64("logo_mark.png")

SUIT_COLORS = {
    "Wands": "#C99A5E",     # fire — gold, matches the logo's sparkle accent
    "Cups": "#1857C4",      # water — brand blue
    "Swords": "#9AA0AE",    # air — neutral steel
    "Pentacles": "#5C7A54", # earth — green (kept distinct from brand hues on purpose)
    None: "#4C1F93",        # Major Arcana — brand indigo
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --ink: #120C22;
    --ink-panel: #1C1533;
    --ink-line: #342A54;
    --brass: #C99A5E;        /* gold accent, from the logo's sparkle color */
    --brass-soft: #DDBB86;   /* lighter gold, for emphasis text */
    --oxblood: #E263A8;      /* legible magenta tint — markers, retrograde */
    --parchment: #F1ECFA;
    --mist: #ADA1C9;
    --magenta: #C2157E;      /* vivid brand magenta — decorative/gradient only */
    --indigo: #4C1F93;       /* vivid brand indigo — decorative/gradient only */
    --blue: #1857C4;         /* vivid brand blue — decorative/gradient only */
    --brand-gradient: linear-gradient(90deg, var(--magenta) 0%, var(--indigo) 55%, var(--blue) 100%);
}

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 18% -8%, rgba(194,21,126,0.07), transparent 60%),
        radial-gradient(ellipse 800px 500px at 85% 0%, rgba(24,87,196,0.06), transparent 55%),
        var(--ink);
    color: var(--parchment);
}

section[data-testid="stSidebar"] {
    background: #0D0919;
    border-right: 1px solid var(--ink-line);
}
section[data-testid="stSidebar"] * { color: var(--mist); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 { color: var(--brass-soft) !important; }

h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; font-weight: 600; color: var(--parchment) !important; }
h1 { font-weight: 700; }
p, li, span, label, div { line-height: 1.55; }

::selection { background: rgba(194,21,126,0.35); }
*:focus-visible { outline: 2px solid var(--magenta); outline-offset: 2px; }

/* ---------- Hero (main page top) ---------- */
.anupt-hero { text-align: center; padding: 0.3rem 1rem 1.3rem 1rem; border-bottom: 1px solid var(--ink-line); margin-bottom: 1.6rem; }
.anupt-hero .hero-mark { width: 104px; height: auto; filter: drop-shadow(0 0 22px rgba(194,21,126,0.18)); margin-bottom: 0.3rem; }
.anupt-hero .brand-word {
    font-family: 'Fraunces', serif; font-optical-sizing: auto; font-weight: 700; font-size: 2.3rem;
    background: var(--brand-gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
    letter-spacing: 0.03em; line-height: 1.1;
}
.anupt-hero .tagline { color: var(--mist); font-size: 1rem; font-style: italic; font-family: 'Fraunces', serif; margin-top: 0.2rem; }
.anupt-hero .systems-line {
    font-family: 'Manrope', sans-serif; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--brass); margin-top: 0.55rem;
}

/* ---------- Sidebar mark (compact) ---------- */
.anupt-sidebar-mark { text-align: center; padding: 0.3rem 0 1.1rem 0; }
.anupt-sidebar-mark img { width: 58px; height: auto; filter: drop-shadow(0 0 14px rgba(194,21,126,0.2)); }
.anupt-sidebar-mark .sidebar-word {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.25rem; margin-top: 0.3rem; letter-spacing: 0.03em;
    background: var(--brand-gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* ---------- Scroll panel (AI-written readings) ---------- */
.anupt-scroll {
    background: var(--ink-panel);
    border-top: 2px solid var(--oxblood);
    border-radius: 3px;
    padding: 1.5rem 1.7rem;
    margin: 0.7rem 0 1.1rem 0;
    color: var(--parchment);
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    line-height: 1.75;
    max-width: 74ch;
}
.anupt-scroll::first-letter {
    font-family: 'Fraunces', serif; font-size: 3.4rem; font-weight: 700; color: var(--magenta);
    float: left; line-height: 0.8; padding-right: 0.5rem; padding-top: 0.4rem;
}
.anupt-scroll .scroll-label {
    font-family: 'Manrope', sans-serif; font-size: 0.72rem; color: var(--mist);
    letter-spacing: 0.05em; margin-bottom: 0.6rem; display:block;
}

/* ---------- Plain hairline tables ---------- */
table.anupt-table { width: 100%; border-collapse: collapse; margin: 0.4rem 0 1.2rem 0; font-family: 'Manrope', sans-serif; font-size: 0.92rem; }
table.anupt-table th {
    text-align: left; color: var(--mist); font-weight: 600; font-size: 0.78rem;
    padding: 0.4rem 0.7rem; border-bottom: 1px solid var(--brass);
}
table.anupt-table td { padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--ink-line); color: var(--parchment); }
table.anupt-table tr:last-child td { border-bottom: none; }
table.anupt-table .glyph { color: var(--brass-soft); font-size: 1.1rem; margin-right: 0.4rem; }
table.anupt-table .retro { color: var(--oxblood); font-weight: 600; }

/* ---------- Theme-strength meter ---------- */
.anupt-meter-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0; }
.anupt-meter-label { font-family: 'Fraunces', serif; font-size: 0.98rem; color: var(--parchment); width: 9.5rem; flex-shrink: 0; }
.anupt-meter-dots { display: flex; gap: 4px; }
.anupt-meter-dots span { width: 9px; height: 9px; border-radius: 50%; background: var(--ink-line); display:inline-block; }
.anupt-meter-dots span.filled { background: var(--magenta); }
.anupt-meter-systems { color: var(--mist); font-size: 0.8rem; }

/* ---------- Natal wheel ---------- */
.anupt-wheel-wrap { text-align: center; margin: 0.6rem 0 1.2rem 0; }
.anupt-wheel-spin { transform-origin: center; animation: wheel-cast 1.1s cubic-bezier(.2,.7,.3,1) both; }
@keyframes wheel-cast { from { opacity: 0; transform: rotate(-10deg) scale(0.94); } to { opacity: 1; transform: rotate(0) scale(1); } }
@media (prefers-reduced-motion: reduce) { .anupt-wheel-spin { animation: none; } }
.wheel-ring-outer, .wheel-ring-inner, .wheel-ring-core { fill: none; stroke: var(--ink-line); stroke-width: 1.2; }
.wheel-ring-outer { stroke: var(--indigo); stroke-width: 1.6; }
.wheel-divider { stroke: var(--ink-line); stroke-width: 1; }
.wheel-sign-glyph { font-size: 17px; fill: var(--blue); font-family: 'Noto Sans Symbols2','Segoe UI Symbol','DejaVu Sans',sans-serif; }
.wheel-house-num { font-size: 10.5px; fill: var(--mist); font-family: 'Manrope', sans-serif; }
.wheel-planet { font-size: 16px; fill: var(--parchment); font-family: 'Noto Sans Symbols2','Segoe UI Symbol','DejaVu Sans',sans-serif; }
.wheel-planet-retro { font-size: 16px; fill: var(--oxblood); font-family: 'Noto Sans Symbols2','Segoe UI Symbol','DejaVu Sans',sans-serif; }
.wheel-asc-line { stroke: var(--magenta); stroke-width: 1.8; }
.wheel-asc-label { font-size: 10px; fill: var(--oxblood); font-weight: 700; letter-spacing: 0.05em; }
.wheel-core-label { font-size: 12.5px; fill: var(--brass-soft); font-family: 'Fraunces', serif; }
.wheel-core-sub { font-size: 9.5px; fill: var(--mist); }

/* ---------- Numerology medallions (magenta — matches the logo's numerology circle) ---------- */
.anupt-medallion-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 0.8rem; margin: 0.6rem 0 1.2rem 0; }
.anupt-medallion { border: 1px solid var(--ink-line); border-radius: 4px; padding: 0.9rem 0.7rem; text-align: center; background: var(--ink-panel); }
.anupt-medallion .med-num { font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 700; color: var(--magenta); line-height: 1; }
.anupt-medallion .med-name { font-size: 0.74rem; color: var(--mist); margin-top: 0.35rem; letter-spacing: 0.02em; }

/* ---------- Tarot cards ---------- */
.anupt-tarot-row { display: flex; gap: 0.9rem; flex-wrap: wrap; justify-content: center; margin: 0.6rem 0; }
.anupt-tarot-card {
    width: 148px; aspect-ratio: 2 / 3.3;
    background: linear-gradient(165deg, #1B1430 0%, #100B22 100%);
    border-radius: 8px; padding: 0.85rem 0.6rem; text-align: center;
    display: flex; flex-direction: column; justify-content: space-between;
    position: relative;
}
.anupt-tarot-card::before {
    content: ""; position: absolute; inset: 6px; border: 1px solid var(--card-suit-color, var(--indigo));
    border-radius: 5px; pointer-events: none; opacity: 0.6;
}
.anupt-tarot-card .tarot-pos { color: var(--mist); font-size: 0.68rem; letter-spacing: 0.03em; }
.anupt-tarot-card .tarot-name { font-family: 'Fraunces', serif; color: var(--parchment); font-size: 1rem; margin: 0.3rem 0; }
.anupt-tarot-card .tarot-orient { color: var(--card-suit-color, var(--magenta)); font-size: 0.76rem; font-weight: 600; }
.anupt-tarot-card .tarot-kw { color: var(--mist); font-size: 0.76rem; margin-top: 0.4rem; line-height: 1.4; }

/* ---------- Buttons & inputs ---------- */
.stButton > button {
    background: var(--brand-gradient); color: #FFFFFF; border: none; border-radius: 4px;
    font-weight: 700; font-family: 'Manrope', sans-serif; padding: 0.55rem 1.3rem; letter-spacing: 0.01em;
}
.stButton > button:hover { filter: brightness(1.1); }
.stButton > button p { color: #FFFFFF !important; font-weight: 700 !important; }

div[data-testid="stMetric"] { background: var(--ink-panel); border: 1px solid var(--ink-line); border-radius: 4px; padding: 0.6rem 0.8rem; }
div[data-testid="stMetric"] label { color: var(--mist) !important; }
div[data-testid="stMetric"] div { color: var(--brass-soft) !important; }

div[data-testid="stChatMessage"] { background: var(--ink-panel); border: 1px solid var(--ink-line); border-radius: 6px; }

hr, .anupt-divider { border: none; border-top: 1px solid var(--ink-line); margin: 1.2rem 0; }

.anupt-disclaimer { font-size: 0.76rem; color: var(--mist); border-top: 1px solid var(--ink-line); padding-top: 0.6rem; margin-top: 1.5rem; }
.anupt-caption { color: var(--mist); font-size: 0.85rem; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero():
    """Main page-top brand mark: real logo crest + gradient wordmark + the logo's own tagline copy."""
    st.markdown(
        f"""<div class="anupt-hero">
        <img class="hero-mark" src="data:image/png;base64,{_LOGO_MARK}" alt="ANUPT emblem" />
        <div class="brand-word">ANUPT</div>
        <div class="tagline">Insights for a better you</div>
        <div class="systems-line">Astrology &nbsp;·&nbsp; Numerology &nbsp;·&nbsp; Palmistry &nbsp;·&nbsp; Tarot</div>
        </div>""",
        unsafe_allow_html=True,
    )


def sidebar_mark():
    """Compact sidebar version — just the crest and wordmark, no tagline (space is tight)."""
    st.markdown(
        f"""<div class="anupt-sidebar-mark">
        <img src="data:image/png;base64,{_LOGO_MARK}" alt="ANUPT" />
        <div class="sidebar-word">ANUPT</div>
        </div>""",
        unsafe_allow_html=True,
    )


def scroll_panel(label: str, html_body: str):
    """The AI-written reading surface — illuminated drop-cap opening, parchment-on-ink."""
    st.markdown(
        f"""<div class="anupt-scroll"><span class="scroll-label">{label}</span>{html_body}</div>""",
        unsafe_allow_html=True,
    )


def strength_meter(theme_label: str, strength: str, systems: list) -> str:
    """Three-dot meter."""
    filled = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}.get(strength, 0)
    dots = "".join(f'<span class="{"filled" if i < filled else ""}"></span>' for i in range(3))
    sys_txt = ", ".join(systems) if systems else "no signal"
    return (f'<div class="anupt-meter-row"><span class="anupt-meter-label">{theme_label}</span>'
            f'<span class="anupt-meter-dots">{dots}</span>'
            f'<span class="anupt-meter-systems">{sys_txt}</span></div>')


def table(headers: list, rows: list) -> str:
    """Plain hairline HTML table — used instead of st.dataframe for full style control."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body += f"<tr>{cells}</tr>"
    return f'<table class="anupt-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def medallion_grid(items: list) -> str:
    """items: list of (number, label) tuples — numerology display, magenta to match the logo's numerology circle."""
    cells = "".join(
        f'<div class="anupt-medallion"><div class="med-num">{num}</div>'
        f'<div class="med-name">{label}</div></div>'
        for num, label in items
    )
    return f'<div class="anupt-medallion-grid">{cells}</div>'


def tarot_card_html(card: dict) -> str:
    color = SUIT_COLORS.get(card.get("suit"), "#4C1F93")
    return (
        f'<div class="anupt-tarot-card" style="--card-suit-color:{color}">'
        f'<div class="tarot-pos">{card["position"]}</div>'
        f'<div><div class="tarot-name">{card["name"]}</div>'
        f'<div class="tarot-orient">{card["orientation"]}</div></div>'
        f'<div class="tarot-kw">{card["keywords"]}</div>'
        f'</div>'
    )
