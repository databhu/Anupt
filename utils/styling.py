"""
ANUPT — visual identity, v3: modern premium light theme + bottom navigation.

Same brand hues as before (they were derived from the real uploaded logo,
so they don't change), just re-balanced for a light surface instead of a
night-sky one, and restructured around a mobile-app navigation pattern
(slim sticky top bar + fixed bottom nav) instead of a desktop sidebar.

Color (named):
  bg           #FAF9FD  — page background (soft lavender-white, not sterile #FFF)
  surface      #FFFFFF  — cards, panels
  surface-alt  #F5F2FA  — subtle secondary surface (hover states, alt rows)
  border       #E8E4EF  — hairlines
  ink          #1D1830  — primary text (dark violet-black, not pure black)
  mist         #6B6478  — secondary/muted text
  magenta      #C2157E  — brand accent (5.4:1 on bg — safe for text too)
  indigo       #4C1F93  — brand accent (10.4:1 on bg)
  blue         #1857C4  — brand accent (6.3:1 on bg)
  gold         #C99A5E  — decorative accent only (2.4:1 — not for text)
  gold-text    #8C6530  — darkened gold for the few places gold needs to read as text (5.0:1)

Layout: a slim sticky top bar (just the mark + wordmark) replaces the old
full-hero-on-every-page pattern — the full hero now only appears once, on
Home. Navigation moves from the old sidebar radio to a fixed bottom bar
(st.container(key=...) pinned via CSS, real st.button widgets inside so
clicks still work as ordinary Streamlit interactions — no custom JS
component involved). Surface language stays varied by content type
(scroll panel / medallions / tarot cards / hairline tables / the wheel)
rather than flattening everything into one repeated card style.
"""

import base64
import html
import re
from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _b64(filename: str) -> str:
    return base64.b64encode((_ASSETS_DIR / filename).read_bytes()).decode("utf-8")


def _prose_to_html(text: str) -> str:
    """Defensive formatting for AI-generated text embedded as raw HTML (not markdown-
    rendered by Streamlit in this context): escape it, then convert the handful of
    markdown patterns models use out of habit despite being told not to (bold,
    blank-line paragraph breaks) into real HTML, so a stray '**word**' never shows
    up as literal asterisks on screen."""
    escaped = html.escape(text.strip())
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", escaped) if p.strip()]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs) or f"<p>{escaped}</p>"


_LOGO_MARK = _b64("logo_mark.png")

SUIT_COLORS = {
    "Wands": "#B8863F",     # fire — darkened gold, text-safe on a white card
    "Cups": "#1857C4",      # water — brand blue
    "Swords": "#6B7280",    # air — neutral steel
    "Pentacles": "#3F6B3A", # earth — darkened green, text-safe
    None: "#4C1F93",        # Major Arcana — brand indigo
}

# Nav items shared between styling (icons) and app.py (routing) so the two
# never drift apart. Tuple is (routing_key, material_icon, short_display_label) —
# the routing key stays the full word (used throughout app.py's page routing and
# reading-history labels); only the on-screen button text is shortened, since 7
# full-length labels don't fit a phone-width bottom bar without truncating.
NAV_ITEMS = [
    ("Home", "home", "Home"),
    ("Astrology", "travel_explore", "Astro"),
    ("Numerology", "tag", "Nums"),
    ("Palmistry", "back_hand", "Palm"),
    ("Tarot", "style", "Tarot"),
    ("ANUPT", "hub", "ANUPT"),
    ("Profile", "person", "You"),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg: #FAF9FD;
    --surface: #FFFFFF;
    --surface-alt: #F5F2FA;
    --border: #E8E4EF;
    --ink: #1D1830;
    --mist: #6B6478;
    --magenta: #C2157E;
    --indigo: #4C1F93;
    --blue: #1857C4;
    --gold: #C99A5E;
    --gold-text: #8C6530;
    --brand-gradient: linear-gradient(90deg, var(--magenta) 0%, var(--indigo) 55%, var(--blue) 100%);
    /* legacy aliases so nothing else in the codebase needs to change names */
    --brass: var(--gold-text);
    --brass-soft: var(--indigo);
    --oxblood: var(--magenta);
    --parchment: var(--ink);
    /* Responsive content width: a comfortable single-column reading width on
       phones, growing on wider screens so desktop/tablet actually uses the
       space instead of sitting in a narrow centered strip with huge empty
       margins either side (that was the whole "web view too limited" bug).
       Top bar, main content and bottom nav all reference this one variable
       so they stay visually aligned as they grow together. */
    --content-max: 760px;
}
@media (min-width: 900px)  { :root { --content-max: 900px; } }
@media (min-width: 1200px) { :root { --content-max: 1040px; } }
@media (min-width: 1600px) { :root { --content-max: 1160px; } }

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

.stApp { background: var(--bg); color: var(--ink); }

/* Streamlit's own sidebar is unused now (nav lives in the bottom bar) — hide it and its toggle */
section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; font-weight: 600; color: var(--ink) !important; }
h1 { font-weight: 700; }
p, li, span, label, div { line-height: 1.55; }

::selection { background: rgba(194,21,126,0.18); }
*:focus-visible { outline: 2px solid var(--magenta); outline-offset: 2px; }

/* Leave room at the bottom so the fixed nav never covers content */
.block-container { padding-bottom: 6rem !important; padding-top: 1rem !important; max-width: var(--content-max) !important; margin-left: auto !important; margin-right: auto !important; }

/* ---------- Slim sticky top bar (every page) ---------- */
.st-key-topbar {
    position: sticky; top: 0; z-index: 998;
    background: rgba(250,249,253,0.92); backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    margin: -1rem -1rem 1rem -1rem; padding: 0.55rem 1rem;
}
.anupt-topbar-inner { display: flex; align-items: center; gap: 0.5rem; max-width: var(--content-max); margin: 0 auto; }
.anupt-topbar-inner img { width: 26px; height: 26px; }
.anupt-topbar-inner .word {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.05rem;
    background: var(--brand-gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.anupt-topbar-inner .who { margin-left: auto; color: var(--mist); font-size: 0.82rem; }

/* ---------- Hero (Home page only) ---------- */
.anupt-hero { text-align: center; padding: 0.4rem 1rem 1.2rem 1rem; }
.anupt-hero .hero-mark { width: 84px; height: auto; margin-bottom: 0.2rem; }
.anupt-hero .brand-word {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.9rem;
    background: var(--brand-gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
    letter-spacing: 0.02em;
}
.anupt-hero .tagline { color: var(--mist); font-size: 0.95rem; font-style: italic; font-family: 'Fraunces', serif; margin-top: 0.1rem; }
.anupt-hero .systems-line {
    font-family: 'Manrope', sans-serif; font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--gold-text); margin-top: 0.5rem;
}

/* ---------- Fixed bottom navigation ---------- */
.st-key-bottomnav {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 999;
    background: rgba(255,255,255,0.94); backdrop-filter: blur(14px);
    border-top: 1px solid var(--border);
    box-shadow: 0 -6px 24px rgba(29,24,48,0.06);
    padding: 0.3rem 0.3rem calc(0.3rem + env(safe-area-inset-bottom, 0px)) 0.3rem;
}
.st-key-bottomnav > div { max-width: var(--content-max); margin: 0 auto; }
.st-key-bottomnav [data-testid="stHorizontalBlock"] { gap: 0.1rem; align-items: stretch; flex-wrap: nowrap !important; }
.st-key-bottomnav [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
.st-key-bottomnav .stButton { width: 100%; }
.st-key-bottomnav .stButton button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    width: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 1px; color: var(--mist) !important; font-size: 0.6rem; font-weight: 600;
    padding: 0.3rem 0.05rem !important; min-height: 3.1rem; border-radius: 12px !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.st-key-bottomnav .stButton button:hover { background: var(--surface-alt) !important; }
.st-key-bottomnav .stButton button p {
    font-size: 0.6rem !important; margin: 0 !important; overflow: hidden; text-overflow: ellipsis; max-width: 100%;
}
.st-key-bottomnav .stButton button span[role="img"] { font-size: 1.15rem !important; }
.st-key-bottomnav .stButton button[kind="primary"] { color: var(--magenta) !important; background: rgba(194,21,126,0.08) !important; }

/* ---------- Scroll panel (AI-written readings) ---------- */
.anupt-scroll {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--magenta);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 0.7rem 0 1.1rem 0;
    color: var(--ink);
    font-family: 'Fraunces', serif;
    font-size: 1.03rem;
    line-height: 1.75;
    box-shadow: 0 2px 14px rgba(29,24,48,0.04);
}
.anupt-scroll::first-letter {
    font-family: 'Fraunces', serif; font-size: 3.1rem; font-weight: 700; color: var(--magenta);
    float: left; line-height: 0.8; padding-right: 0.5rem; padding-top: 0.3rem;
}
.anupt-scroll .scroll-label {
    font-family: 'Manrope', sans-serif; font-size: 0.72rem; color: var(--mist);
    letter-spacing: 0.05em; margin-bottom: 0.6rem; display:block;
}

/* ---------- Plain hairline tables ---------- */
table.anupt-table { width: 100%; border-collapse: collapse; margin: 0.4rem 0 1.2rem 0; font-family: 'Manrope', sans-serif; font-size: 0.92rem;
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
table.anupt-table th {
    text-align: left; color: var(--mist); font-weight: 600; font-size: 0.78rem;
    padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); background: var(--surface-alt);
}
table.anupt-table td { padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); color: var(--ink); }
table.anupt-table tr:last-child td { border-bottom: none; }
table.anupt-table .retro { color: var(--magenta); font-weight: 600; }

/* ---------- Summary card (the concise, question-first answer) ---------- */
.anupt-summary-card {
    background: linear-gradient(165deg, rgba(194,21,126,0.06) 0%, rgba(24,87,196,0.05) 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--magenta);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin: 0.7rem 0 1rem 0;
    color: var(--ink);
    animation: fade-up 0.35s ease-out both;
}
.anupt-summary-card .summary-label {
    font-family: 'Manrope', sans-serif; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--magenta); font-weight: 700; margin-bottom: 0.4rem; display: block;
}
.anupt-summary-card .summary-text {
    font-family: 'Fraunces', serif; font-size: 1.12rem; line-height: 1.6; font-weight: 500;
}
@keyframes fade-up { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@media (prefers-reduced-motion: reduce) { .anupt-summary-card { animation: none; } }

/* ---------- Guidance / tip box (palm-photo instructions etc.) ---------- */
.anupt-tips { background: var(--surface-alt); border: 1px solid var(--border); border-radius: 12px;
    padding: 0.85rem 1rem; margin: 0.5rem 0 1rem 0; font-size: 0.85rem; color: var(--ink); }
.anupt-tips ul { margin: 0.3rem 0 0 0; padding-left: 1.1rem; }
.anupt-tips li { margin-bottom: 0.2rem; }

/* ---------- ANUPT insight cards (combined page) ---------- */
.anupt-insight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.6rem; margin: 0.7rem 0 1rem 0; }
.anupt-insight-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 0.85rem 0.9rem;
    box-shadow: 0 2px 10px rgba(29,24,48,0.03); animation: fade-up 0.35s ease-out both;
}
.anupt-insight-card .ic-theme { font-family: 'Fraunces', serif; font-size: 0.98rem; color: var(--ink); font-weight: 600; }
.anupt-insight-card .ic-dots { margin: 0.35rem 0; }
.anupt-insight-card .ic-systems { font-size: 0.72rem; color: var(--mist); }
.anupt-meter-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0; }
.anupt-meter-label { font-family: 'Fraunces', serif; font-size: 0.96rem; color: var(--ink); width: 9.5rem; flex-shrink: 0; }
.anupt-meter-dots { display: flex; gap: 4px; }
.anupt-meter-dots span { width: 9px; height: 9px; border-radius: 50%; background: var(--border); display:inline-block; }
.anupt-meter-dots span.filled { background: var(--magenta); }
.anupt-meter-systems { color: var(--mist); font-size: 0.8rem; }

/* ---------- Natal wheel ---------- */
.anupt-wheel-wrap { text-align: center; margin: 0.6rem 0 1.2rem 0; }
.anupt-wheel-spin { transform-origin: center; animation: wheel-cast 1.1s cubic-bezier(.2,.7,.3,1) both; }
@keyframes wheel-cast { from { opacity: 0; transform: rotate(-10deg) scale(0.94); } to { opacity: 1; transform: rotate(0) scale(1); } }
@media (prefers-reduced-motion: reduce) { .anupt-wheel-spin { animation: none; } }
.wheel-ring-outer, .wheel-ring-inner, .wheel-ring-core { fill: none; stroke: var(--border); stroke-width: 1.2; }
.wheel-ring-outer { stroke: var(--indigo); stroke-width: 1.6; }
.wheel-divider { stroke: var(--border); stroke-width: 1; }
.wheel-sign-glyph { font-size: 17px; fill: var(--blue); font-family: 'Noto Sans Symbols2','Segoe UI Symbol','DejaVu Sans',sans-serif; }
.wheel-house-num { font-size: 10.5px; fill: var(--mist); font-family: 'Manrope', sans-serif; }
.wheel-planet { font-size: 16px; fill: var(--ink); font-family: 'Noto Sans Symbols2','Segoe UI Symbol','DejaVu Sans',sans-serif; }
.wheel-planet-retro { font-size: 16px; fill: var(--magenta); font-family: 'Noto Sans Symbols2','Segoe UI Symbol','DejaVu Sans',sans-serif; }
.wheel-asc-line { stroke: var(--magenta); stroke-width: 1.8; }
.wheel-asc-label { font-size: 10px; fill: var(--magenta); font-weight: 700; letter-spacing: 0.05em; }
.wheel-core-label { font-size: 12.5px; fill: var(--indigo); font-family: 'Fraunces', serif; }
.wheel-core-sub { font-size: 9.5px; fill: var(--mist); }

/* ---------- Numerology medallions ---------- */
.anupt-medallion-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 0.7rem; margin: 0.6rem 0 1.2rem 0; }
.anupt-medallion { border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem 0.7rem; text-align: center;
    background: var(--surface); box-shadow: 0 2px 10px rgba(29,24,48,0.03); }
.anupt-medallion .med-num { font-family: 'Fraunces', serif; font-size: 2.05rem; font-weight: 700; color: var(--magenta); line-height: 1; }
.anupt-medallion .med-name { font-size: 0.73rem; color: var(--mist); margin-top: 0.35rem; letter-spacing: 0.02em; }

/* ---------- Tarot cards (light card face, suit-colored edge) ---------- */
.anupt-tarot-row { display: flex; gap: 0.8rem; flex-wrap: wrap; justify-content: center; margin: 0.6rem 0; }
.anupt-tarot-card {
    width: 140px; aspect-ratio: 2 / 3.3;
    background: var(--surface);
    border-radius: 10px; padding: 0.8rem 0.55rem; text-align: center;
    display: flex; flex-direction: column; justify-content: space-between;
    position: relative; box-shadow: 0 3px 14px rgba(29,24,48,0.06);
}
.anupt-tarot-card::before {
    content: ""; position: absolute; inset: 6px; border: 1.5px solid var(--card-suit-color, var(--indigo));
    border-radius: 6px; pointer-events: none; opacity: 0.55;
}
.anupt-tarot-card .tarot-pos { color: var(--mist); font-size: 0.66rem; letter-spacing: 0.03em; }
.anupt-tarot-card .tarot-name { font-family: 'Fraunces', serif; color: var(--ink); font-size: 0.98rem; margin: 0.3rem 0; }
.anupt-tarot-card .tarot-orient { color: var(--card-suit-color, var(--magenta)); font-size: 0.74rem; font-weight: 700; }
.anupt-tarot-card .tarot-kw { color: var(--mist); font-size: 0.74rem; margin-top: 0.4rem; line-height: 1.4; }

/* ---------- Cards (city suggestions, generic content cards) ---------- */
.anupt-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 1rem 1.1rem; margin-bottom: 0.6rem; box-shadow: 0 2px 10px rgba(29,24,48,0.03); }

/* ---------- Buttons & inputs ---------- */
.stButton button {
    background: var(--brand-gradient); color: #FFFFFF; border: none; border-radius: 10px;
    font-weight: 700; font-family: 'Manrope', sans-serif; padding: 0.55rem 1.3rem; letter-spacing: 0.01em;
}
.stButton button:hover { filter: brightness(1.08); }
.stButton button p { color: #FFFFFF !important; font-weight: 700 !important; }
.stButton button[kind="secondary"] {
    background: var(--surface) !important; color: var(--ink) !important; border: 1px solid var(--border) !important;
}
.stButton button[kind="secondary"] p { color: var(--ink) !important; }

div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input, div[data-testid="stTimeInput"] input {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px; color: var(--ink);
}

div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.6rem 0.8rem; }
div[data-testid="stMetric"] label { color: var(--mist) !important; }
div[data-testid="stMetric"] div { color: var(--indigo) !important; }

div[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }

.stTabs [data-baseweb="tab"] { color: var(--mist); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--magenta) !important; }

hr, .anupt-divider { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }

.anupt-disclaimer { font-size: 0.76rem; color: var(--mist); border-top: 1px solid var(--border); padding-top: 0.6rem; margin-top: 1.5rem; }
.anupt-caption { color: var(--mist); font-size: 0.85rem; }
/* ---------- Compact inline utility buttons (e.g. the Home page's quick Edit link) ---------- */
.st-key-home_edit_profile .stButton button {
    padding: 0.35rem 0.7rem !important; font-size: 0.8rem !important; min-height: 38px;
}
.st-key-home_edit_profile .stButton button p { font-size: 0.8rem !important; }

.anupt-badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
    background: var(--surface-alt); color: var(--indigo); border: 1px solid var(--border);
}

/* ---------- Touch & interaction polish ---------- */
.stButton button, .st-key-bottomnav .stButton button { min-height: 44px; touch-action: manipulation; transition: filter 0.12s ease, background 0.12s ease; }
.stButton button:active { filter: brightness(0.94); }
.st-key-bottomnav .stButton button:active { background: var(--surface-alt) !important; }
[data-testid="stExpander"] summary { min-height: 44px; display: flex; align-items: center; touch-action: manipulation; }
div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label { min-height: 30px; touch-action: manipulation; }

/* ---------- Mobile-width refinements (phones, ~480px and under) ---------- */
@media (max-width: 480px) {
    .block-container { padding-left: 0.9rem !important; padding-right: 0.9rem !important; }
    .anupt-hero .brand-word { font-size: 1.6rem; }
    .anupt-hero .hero-mark { width: 70px; }
    h1 { font-size: 1.5rem !important; }
    h2, .stApp h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.1rem !important; }
    .anupt-scroll, .anupt-summary-card { padding: 1rem 1.05rem; }
    .anupt-summary-card .summary-text { font-size: 1.02rem; }
    .anupt-medallion-grid { grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 0.5rem; }
    .anupt-medallion .med-num { font-size: 1.7rem; }
    .anupt-tarot-card { width: 118px; }
    table.anupt-table { font-size: 0.82rem; }
    table.anupt-table th, table.anupt-table td { padding: 0.45rem 0.55rem; }
    .anupt-meter-label { width: 7rem; font-size: 0.86rem; }
}

/* ---------- Small-phone refinements (≤400px): icon-only nav ---------- */
@media (max-width: 400px) {
    .anupt-tarot-row { gap: 0.5rem; }
    .anupt-tarot-card { width: 100px; padding: 0.6rem 0.4rem; }
    /* 7 full icon+label buttons don't reliably fit this narrow without wrapping/
       truncating (verified: breaks at 320px even with a smaller font) — icon-only
       is the robust fix rather than chasing ever-smaller font sizes. The full name
       is still available as a tooltip/accessible label via the button's `help=`.
       Note: Streamlit nests the icon <span> INSIDE the label <p>, so `display:none`
       on the <p> would hide the icon too (it did — caught in visual QA). Collapsing
       the <p> to font-size:0 hides only the text while the icon span keeps its own
       explicit size. */
    .st-key-bottomnav .stButton button p { font-size: 0 !important; line-height: 1 !important; }
    .st-key-bottomnav .stButton button { min-height: 48px; padding: 0.4rem 0.1rem !important; }
    .st-key-bottomnav .stButton button span[role="img"] { font-size: 1.4rem !important; }
}

/* ---------- Wider bottom nav breathing room on tablet/desktop ---------- */
@media (min-width: 700px) {
    .st-key-bottomnav .stButton button { font-size: 0.72rem; min-height: 3.4rem; }
    .st-key-bottomnav .stButton button p { font-size: 0.72rem !important; }
    .st-key-bottomnav .stButton button span[role="img"] { font-size: 1.35rem !important; }
}

/* Never allow horizontal scroll from an oversized child */
.stApp, .block-container { overflow-x: hidden; }
img, svg { max-width: 100%; height: auto; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def top_bar(who: str | None = None):
    """Slim sticky header shown on every page — replaces the old sidebar brand mark."""
    who_html = f'<span class="who">{who}</span>' if who else ""
    with st.container(key="topbar"):
        st.markdown(
            f"""<div class="anupt-topbar-inner">
            <img src="data:image/png;base64,{_LOGO_MARK}" alt="ANUPT" />
            <span class="word">ANUPT</span>{who_html}
            </div>""",
            unsafe_allow_html=True,
        )


def hero():
    """Full brand moment — Home page only now, not repeated on every page."""
    st.markdown(
        f"""<div class="anupt-hero">
        <img class="hero-mark" src="data:image/png;base64,{_LOGO_MARK}" alt="ANUPT emblem" />
        <div class="brand-word">ANUPT</div>
        <div class="tagline">Insights for a better you</div>
        <div class="systems-line">Astrology &nbsp;·&nbsp; Numerology &nbsp;·&nbsp; Palmistry &nbsp;·&nbsp; Tarot</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _set_nav(key: str):
    """on_click callback: runs BEFORE the automatic rerun Streamlit already performs
    after any button click, so the very next script pass already reflects the new
    page — no manual st.rerun() needed, and no risk of a click needing to land twice
    for the page to actually change (the old 'detect the return value, then rerun'
    pattern this replaced was correct in theory but this is the pattern Streamlit
    itself recommends for state-driven navigation, and removes an extra rerun)."""
    st.session_state.nav = key


def bottom_nav(active: str):
    """Renders the fixed bottom navigation bar. Each button updates
    st.session_state.nav directly via on_click — callers don't need to do
    anything with a return value or call st.rerun() themselves.
    Every button carries its full name as a `help` tooltip: on very narrow
    phones the CSS hides the on-screen label to stop it wrapping/truncating
    (icon-only there), so the tooltip/accessible-name is what keeps the
    button's purpose available rather than silently dropping the label."""
    with st.container(key="bottomnav"):
        cols = st.columns(len(NAV_ITEMS))
        for col, (key, icon, short_label) in zip(cols, NAV_ITEMS):
            is_active = key == active
            with col:
                st.button(
                    f":material/{icon}: {short_label}", key=f"nav_{key}",
                    type="primary" if is_active else "secondary",
                    on_click=_set_nav, args=(key,), help=key,
                )


def scroll_panel(label: str, text: str):
    """The AI-written reading surface — illuminated drop-cap opening. `text` is
    the AI's plain-prose output, converted defensively (see _prose_to_html)."""
    st.markdown(
        f"""<div class="anupt-scroll"><span class="scroll-label">{label}</span>{_prose_to_html(text)}</div>""",
        unsafe_allow_html=True,
    )


def summary_card(label: str, text: str):
    """The concise, question-first answer — shown before the detailed/expandable content."""
    st.markdown(
        f"""<div class="anupt-summary-card"><span class="summary-label">{label}</span>
        <div class="summary-text">{_prose_to_html(text)}</div></div>""",
        unsafe_allow_html=True,
    )


def tips_box(title: str, tips: list):
    """Short guidance list, e.g. how to photograph a palm well."""
    items = "".join(f"<li>{t}</li>" for t in tips)
    st.markdown(
        f"""<div class="anupt-tips"><b>{title}</b><ul>{items}</ul></div>""",
        unsafe_allow_html=True,
    )


def insight_grid(cards: list):
    """cards: list of (theme_label, strength, systems_list) — the ANUPT combined page's
    visual theme-strength grid, a more engaging alternative to a plain list of meters."""
    html = ""
    for theme_label, strength, systems in cards:
        filled = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}.get(strength, 0)
        dots = "".join(f'<span class="{"filled" if i < filled else ""}"></span>' for i in range(3))
        sys_txt = ", ".join(systems) if systems else "no signal yet"
        html += (
            f'<div class="anupt-insight-card"><div class="ic-theme">{theme_label}</div>'
            f'<div class="ic-dots anupt-meter-dots">{dots}</div>'
            f'<div class="ic-systems">{sys_txt}</div></div>'
        )
    st.markdown(f'<div class="anupt-insight-grid">{html}</div>', unsafe_allow_html=True)


def strength_meter(theme_label: str, strength: str, systems: list) -> str:
    filled = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}.get(strength, 0)
    dots = "".join(f'<span class="{"filled" if i < filled else ""}"></span>' for i in range(3))
    sys_txt = ", ".join(systems) if systems else "no signal"
    return (f'<div class="anupt-meter-row"><span class="anupt-meter-label">{theme_label}</span>'
            f'<span class="anupt-meter-dots">{dots}</span>'
            f'<span class="anupt-meter-systems">{sys_txt}</span></div>')


def table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body += f"<tr>{cells}</tr>"
    return f'<table class="anupt-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def medallion_grid(items: list) -> str:
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
