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
    /* Score-gauge semantic colors — intuitive red→green favorability convention
       matters more here than strict brand-palette purity, kept muted/pastel so
       it still reads as part of the same soft aesthetic. score-high is the one
       genuinely new hue; mid/low reuse existing brand colors. */
    --score-high: #3F8F5F;
    --score-mid: var(--gold-text);
    --score-low: var(--magenta);
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

/* ---------- Hide Streamlit's own chrome (default header, Deploy button,
   hamburger menu, footer) so ANUPT's own UI owns the full screen. This is
   also what was silently overlapping our own logo before: Streamlit's
   header renders position:absolute at an extremely high z-index
   (999990 — far above anything of ours), directly across the same
   top-of-viewport strip our topbar's logo sits in, visually covering it
   even though nothing in OUR code was actually clipping the image. */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}

/* Leave room at the bottom so the fixed nav never covers content, and
   enough room at the TOP for our own fixed topbar (see below) now that
   Streamlit's header no longer occupies that space. */
.block-container { padding-bottom: 6rem !important; padding-top: 4rem !important; max-width: var(--content-max) !important; margin-left: auto !important; margin-right: auto !important; }

/* ---------- Fixed top bar (every page) ---------- */
/* position:fixed rather than sticky: Streamlit's own .block-container is
   an internally-scrolling region (overflow-y:auto), not the page body —
   confirmed by direct measurement that "sticky" simply scrolled away
   with the content instead of pinning, since its scroll-containing
   ancestor isn't the viewport. Fixed positioning is anchored to the
   viewport directly, so it stays visible regardless of that inner
   scroll container's own behavior. */
.st-key-topbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 998;
    background: rgba(250,249,253,0.92); backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    padding: 0.65rem 1rem;
}
.anupt-topbar-inner { display: flex; align-items: center; gap: 0.5rem; max-width: var(--content-max); margin: 0 auto; }
.anupt-topbar-inner img { width: 26px; height: 26px; flex-shrink: 0; }
.anupt-topbar-inner .word {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.05rem;
    background: var(--brand-gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.anupt-topbar-inner .who { margin-left: auto; color: var(--mist); font-size: 0.82rem; }

/* ---------- Hero (Home page only) ---------- */
.anupt-hero { text-align: center; padding: 0.4rem 1rem 1.2rem 1rem; }
.anupt-hero .hero-mark { width: 84px; height: auto; margin-bottom: 0.2rem; display: inline-block; }
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

/* ---------- Score gauge (favorability rating, 1-10) ---------- */
.anupt-score-row { display: flex; align-items: center; gap: 1rem; margin: 0.6rem 0 1rem 0; }
.anupt-score-gauge { flex-shrink: 0; width: 84px; height: 84px; }
.anupt-score-gauge svg { width: 100%; height: 100%; }
.anupt-score-copy .score-band-label {
    font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 700; color: var(--ink); margin-bottom: 0.15rem;
}
.anupt-score-copy .score-sub { font-size: 0.8rem; color: var(--mist); }

/* ---------- Vedic numerology hero cards (Mulank / Bhagyank) ---------- */
.anupt-vedic-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.8rem; margin: 0.6rem 0 1.1rem 0; }
.anupt-vedic-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.1rem 1.2rem; box-shadow: 0 3px 14px rgba(76,31,147,0.06);
    border-top: 3px solid transparent; border-image: var(--brand-gradient) 1;
}
.anupt-vedic-card .vc-label {
    font-family: 'Manrope', sans-serif; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--magenta); font-weight: 700;
}
.anupt-vedic-card .vc-num { font-family: 'Fraunces', serif; font-size: 2.6rem; font-weight: 700; color: var(--indigo); line-height: 1.15; }
.anupt-vedic-card .vc-tagline { font-family: 'Fraunces', serif; font-style: italic; font-size: 0.92rem; color: var(--ink); margin-top: -0.1rem; }
.anupt-vedic-card .vc-desc { font-size: 0.8rem; color: var(--mist); margin-top: 0.35rem; line-height: 1.4; }

/* ---------- Palm capture guide (camera-viewfinder style reference) ---------- */
.anupt-palm-guide { text-align: center; margin: 0.6rem 0 0.9rem 0; }
.anupt-palm-guide svg { width: 100%; max-width: 220px; height: auto; }
.anupt-palm-guide .pg-caption { font-size: 0.78rem; color: var(--mist); margin-top: 0.3rem; }
/* Overlay variant: sits on top of the live camera preview, low-opacity, non-interactive */
.anupt-palm-overlay { position: relative; }
.anupt-palm-overlay .pg-overlay-svg {
    position: absolute; inset: 0; margin: auto; width: 55%; max-width: 200px; height: auto;
    opacity: 0.4; pointer-events: none; z-index: 5;
}

/* ---------- Life-cycle timeline (Pinnacles & Challenges) ---------- */
.anupt-timeline { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.8rem; margin: 0.7rem 0 1.1rem 0; }
.anupt-timeline-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(29,24,48,0.03); }
.anupt-timeline-card.current { border: 2px solid var(--magenta); box-shadow: 0 4px 18px rgba(194,21,126,0.14); }
.anupt-timeline-card .tc-period { font-family: 'Manrope', sans-serif; font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--mist); }
.anupt-timeline-card .tc-age { font-family: 'Fraunces', serif; font-size: 1.05rem; color: var(--ink);
    font-weight: 600; margin: 0.15rem 0 0.6rem 0; }
.anupt-timeline-card .tc-row { display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.5rem; }
.anupt-timeline-card .tc-label { font-size: 0.75rem; color: var(--mist); }
.anupt-timeline-card .tc-num { font-family: 'Fraunces', serif; font-weight: 700; color: var(--indigo); font-size: 1.3rem; }
.anupt-timeline-card .tc-num.challenge { color: var(--magenta); }
.anupt-timeline-card .tc-meaning { font-size: 0.78rem; color: var(--mist); margin-top: 0.15rem; line-height: 1.4; }
.anupt-timeline-current-badge { display: inline-block; background: var(--magenta); color: white; font-size: 0.62rem;
    font-weight: 700; letter-spacing: 0.04em; padding: 0.12rem 0.55rem; border-radius: 999px; margin-left: 0.4rem; }

/* ---------- Life-area score bars (career/finance/relationships/leadership/creativity) ---------- */
.anupt-score-bar-row { margin-bottom: 1rem; }
.anupt-score-bar-row .sb-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem; }
.anupt-score-bar-row .sb-label { font-family: 'Fraunces', serif; font-weight: 600; color: var(--ink); font-size: 1rem; }
.anupt-score-bar-row .sb-value { font-family: 'Fraunces', serif; font-weight: 700; font-size: 0.92rem; }
.anupt-score-bar-track { background: var(--surface-alt); border-radius: 999px; height: 10px; overflow: hidden; }
.anupt-score-bar-fill { height: 100%; border-radius: 999px; }
.anupt-score-bar-reason { font-size: 0.78rem; color: var(--mist); margin-top: 0.35rem; line-height: 1.4; }

.anupt-meter-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0; }
.anupt-meter-label { font-family: 'Fraunces', serif; font-size: 0.96rem; color: var(--ink); width: 9.5rem; flex-shrink: 0; }

/* ---------- Dignity badges (Astrology) ---------- */
.anupt-dignity-badge {
    display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; font-size: 0.68rem;
    font-weight: 700; letter-spacing: 0.02em;
}
.anupt-dignity-strong { background: rgba(63,143,95,0.12); color: var(--score-high); }
.anupt-dignity-weak { background: rgba(194,21,126,0.1); color: var(--score-low); }
.anupt-dignity-neutral { background: var(--surface-alt); color: var(--mist); }

/* ---------- Rule-based Key Findings (shared across Astrology/Numerology) ---------- */
.anupt-rule-badge {
    display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.1rem 0.55rem; border-radius: 999px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.02em;
    background: rgba(76,31,147,0.08); color: var(--indigo); border: 1px solid rgba(76,31,147,0.25);
}
.anupt-finding-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 0.9rem 1.05rem; margin-bottom: 0.65rem; box-shadow: 0 2px 10px rgba(29,24,48,0.03);
}
.anupt-finding-card .fc-head { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
.anupt-finding-card .fc-title { font-family: 'Fraunces', serif; font-weight: 700; color: var(--ink); font-size: 1rem; }
.anupt-finding-card .fc-tier { font-size: 0.72rem; font-weight: 700; padding: 0.1rem 0.5rem; border-radius: 999px; white-space: nowrap; }
.anupt-finding-card .fc-tier-5, .anupt-finding-card .fc-tier-4 { background: rgba(63,143,95,0.12); color: var(--score-high); }
.anupt-finding-card .fc-tier-3 { background: rgba(140,101,48,0.14); color: var(--score-mid); }
.anupt-finding-card .fc-tier-2, .anupt-finding-card .fc-tier-1 { background: rgba(194,21,126,0.1); color: var(--score-low); }
.anupt-finding-card .fc-text { font-size: 0.87rem; color: var(--ink); line-height: 1.5; }
.anupt-finding-card .fc-exception { font-size: 0.78rem; color: var(--score-low); margin-top: 0.4rem; }
.anupt-finding-card .fc-basis { font-size: 0.74rem; color: var(--mist); margin-top: 0.4rem; }

/* ---------- Aspect list (Astrology) ---------- */
.anupt-aspect-row {
    display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0.2rem;
    border-bottom: 1px solid var(--border); font-size: 0.86rem;
}
.anupt-aspect-row:last-child { border-bottom: none; }
.anupt-aspect-points { font-weight: 600; color: var(--ink); flex: 1; }
.anupt-aspect-name { color: var(--indigo); font-weight: 600; min-width: 5.5rem; }
.anupt-aspect-orb { color: var(--mist); font-size: 0.76rem; }

/* ---------- Yoga cards (Vedic) ---------- */
.anupt-yoga-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.7rem; margin: 0.6rem 0 1rem 0; }
.anupt-yoga-card {
    background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--gold);
    border-radius: 12px; padding: 0.85rem 1rem; box-shadow: 0 2px 10px rgba(29,24,48,0.03);
}
.anupt-yoga-card .yc-name { font-family: 'Fraunces', serif; font-weight: 700; color: var(--ink); font-size: 1rem; }
.anupt-yoga-card .yc-desc { font-size: 0.8rem; color: var(--mist); margin-top: 0.25rem; line-height: 1.4; }

/* ---------- Ask ANUPT navigation chatbot ---------- */
.st-key-anupt_chip_row .stButton > button {
    border-radius: 999px !important; font-size: 0.82rem !important; padding: 0.4rem 0.3rem !important;
    background: var(--surface-alt) !important; border: 1px solid var(--border) !important;
    color: var(--ink) !important; min-height: 2.4rem;
}
.st-key-anupt_chip_row .stButton > button:hover { border-color: var(--magenta) !important; color: var(--magenta) !important; }
.anupt-nav-chat-destination { font-weight: 700; color: var(--magenta); }
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


_SCORE_BANDS = [
    (9, "score-high", "Highly Favorable"),
    (7, "score-high", "Favorable"),
    (5, "score-mid", "Balanced"),
    (3, "score-mid", "Needs Attention"),
    (0, "score-low", "Challenging"),
]


def _score_band(score: int) -> tuple[str, str]:
    for threshold, color_var, label in _SCORE_BANDS:
        if score >= threshold:
            return color_var, label
    return "score-low", "Challenging"


def score_gauge(score: int | None, context_label: str = "This reading") -> str:
    """A ring gauge (0-10) with a plain-language band label next to it — the
    'brief, easy to understand' half of the scoring feature; the AI's written
    summary is the other half. Returns '' (renders nothing) when the model
    didn't produce a usable score, rather than showing a fabricated number."""
    if score is None:
        return ""
    color_var, band_label = _score_band(score)
    r = 34
    circumference = 2 * 3.14159265 * r
    offset = circumference * (1 - score / 10)
    svg = (
        f'<svg viewBox="0 0 84 84">'
        f'<circle cx="42" cy="42" r="{r}" fill="none" stroke="var(--border)" stroke-width="9"/>'
        f'<circle cx="42" cy="42" r="{r}" fill="none" stroke="var(--{color_var})" stroke-width="9" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.1f}" '
        f'stroke-dashoffset="{offset:.1f}" transform="rotate(-90 42 42)"/>'
        f'<text x="42" y="48" text-anchor="middle" font-family="Fraunces,serif" '
        f'font-weight="700" font-size="26" fill="var(--{color_var})">{score}</text>'
        f'</svg>'
    )
    return (
        f'<div class="anupt-score-row">'
        f'<div class="anupt-score-gauge">{svg}</div>'
        f'<div class="anupt-score-copy">'
        f'<div class="score-band-label" style="color:var(--{color_var})">{band_label}</div>'
        f'<div class="score-sub">{context_label} scores {score}/10 based on the evidence above.</div>'
        f'</div></div>'
    )


def vedic_number_cards(mulank: int, mulank_meaning: str, bhagyank: int, bhagyank_meaning: str) -> str:
    """Dedicated hero cards for the two most-asked-about Vedic numerology numbers —
    distinct from the plain medallion grid so they read as headline insights, not
    just two more entries in a list. Mulank = digit-sum of the birth day (the
    engine's 'birthday' number); Bhagyank = digit-sum of the full birth date (the
    engine's 'life_path' number) — same deterministic math, the names Indian
    numerology traditionally uses for them."""
    return (
        '<div class="anupt-vedic-grid">'
        f'<div class="anupt-vedic-card"><div class="vc-label">Mulank</div>'
        f'<div class="vc-num">{mulank}</div><div class="vc-tagline">Your Root Number</div>'
        f'<div class="vc-desc">From your birth day — {mulank_meaning}</div></div>'
        f'<div class="anupt-vedic-card"><div class="vc-label">Bhagyank</div>'
        f'<div class="vc-num">{bhagyank}</div><div class="vc-tagline">Your Destiny Number</div>'
        f'<div class="vc-desc">From your full birth date — {bhagyank_meaning}</div></div>'
        '</div>'
    )


def palm_reference_diagram_svg() -> str:
    """A labeled schematic hand showing where the major lines and two key
    mounts traditionally sit — illustrative only, never the user's own
    hand. Reuses the same hand-outline geometry as the capture guide for
    visual consistency across the Palmistry page."""
    return (
        '<svg viewBox="0 0 280 260" xmlns="http://www.w3.org/2000/svg" '
        'font-family="Manrope, sans-serif" font-size="11">'
        # hand outline: palm + four fingers + thumb (shifted +40 from the capture-guide version)
        '<rect x="95" y="128" width="90" height="102" rx="34" fill="none" stroke="var(--ink)" stroke-width="2.5"/>'
        '<rect x="102" y="54" width="17" height="82" rx="8.5" fill="none" stroke="var(--ink)" stroke-width="2.5"/>'
        '<rect x="123" y="38" width="17" height="98" rx="8.5" fill="none" stroke="var(--ink)" stroke-width="2.5"/>'
        '<rect x="144" y="48" width="17" height="88" rx="8.5" fill="none" stroke="var(--ink)" stroke-width="2.5"/>'
        '<rect x="165" y="68" width="15" height="68" rx="7.5" fill="none" stroke="var(--ink)" stroke-width="2.5"/>'
        '<rect x="58" y="160" width="52" height="23" rx="11.5" fill="none" stroke="var(--ink)" stroke-width="2.5" '
        'transform="rotate(-38 84 171)"/>'
        # the four major lines
        '<path d="M108,136 Q93,152 89,178 Q87,202 99,226" fill="none" stroke="var(--magenta)" stroke-width="2.5"/>'
        '<path d="M100,141 Q140,137 179,143" fill="none" stroke="var(--indigo)" stroke-width="2.5"/>'
        '<path d="M103,159 Q142,164 177,169" fill="none" stroke="var(--blue)" stroke-width="2.5"/>'
        '<path d="M141,226 L141,149" fill="none" stroke="var(--gold-text)" stroke-width="2.5"/>'
        # two key mounts, as dots
        '<circle cx="107" cy="196" r="4" fill="var(--magenta)"/>'
        '<circle cx="172" cy="206" r="4" fill="var(--indigo)"/>'
        # leader lines + labels, kept clear of the hand itself
        '<line x1="60" y1="141" x2="100" y2="141" stroke="var(--indigo)" stroke-width="1" stroke-dasharray="2 2"/>'
        '<text x="4" y="139" fill="var(--indigo)">Heart Line</text>'
        '<line x1="60" y1="159" x2="103" y2="159" stroke="var(--blue)" stroke-width="1" stroke-dasharray="2 2"/>'
        '<text x="4" y="157" fill="var(--blue)">Head Line</text>'
        '<line x1="60" y1="180" x2="89" y2="180" stroke="var(--magenta)" stroke-width="1" stroke-dasharray="2 2"/>'
        '<text x="4" y="183" fill="var(--magenta)">Life Line</text>'
        '<line x1="60" y1="196" x2="103" y2="196" stroke="var(--magenta)" stroke-width="1" stroke-dasharray="2 2"/>'
        '<text x="4" y="199" fill="var(--magenta)">Mount of Venus</text>'
        '<line x1="185" y1="149" x2="220" y2="149" stroke="var(--gold-text)" stroke-width="1" stroke-dasharray="2 2"/>'
        '<text x="222" y="152" fill="var(--gold-text)">Fate Line</text>'
        '<line x1="185" y1="206" x2="210" y2="206" stroke="var(--indigo)" stroke-width="1" stroke-dasharray="2 2"/>'
        '<text x="212" y="209" fill="var(--indigo)">Mount of Luna</text>'
        '</svg>'
    )


def palm_reference_card():
    """The illustrative reference diagram in its own clearly-labeled card —
    never to be confused with the user's own annotated photo."""
    st.markdown(
        f'<div class="anupt-palm-guide">{palm_reference_diagram_svg()}'
        f'<div class="pg-caption">Illustrative example — general locations only, not your hand. '
        f'Every real hand differs.</div></div>',
        unsafe_allow_html=True,
    )


def _palm_guide_inner_svg() -> str:
    """The hand-outline + viewfinder-corner reference shape, shared by both the
    standalone guide card and the on-camera overlay variant."""
    return (
        '<svg viewBox="0 0 200 260" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="8" y="8" width="184" height="244" rx="18" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-dasharray="7 6" opacity="0.55"/>'
        # corner brackets, camera-viewfinder style
        '<path d="M8 34 V8 H34" fill="none" stroke="currentColor" stroke-width="3.5"/>'
        '<path d="M166 8 H192 V34" fill="none" stroke="currentColor" stroke-width="3.5"/>'
        '<path d="M192 226 V252 H166" fill="none" stroke="currentColor" stroke-width="3.5"/>'
        '<path d="M34 252 H8 V226" fill="none" stroke="currentColor" stroke-width="3.5"/>'
        # hand: palm + four fingers + thumb
        '<rect x="55" y="128" width="90" height="102" rx="34" fill="none" stroke="currentColor" stroke-width="3"/>'
        '<rect x="62" y="54" width="17" height="82" rx="8.5" fill="none" stroke="currentColor" stroke-width="3"/>'
        '<rect x="83" y="38" width="17" height="98" rx="8.5" fill="none" stroke="currentColor" stroke-width="3"/>'
        '<rect x="104" y="48" width="17" height="88" rx="8.5" fill="none" stroke="currentColor" stroke-width="3"/>'
        '<rect x="125" y="68" width="15" height="68" rx="7.5" fill="none" stroke="currentColor" stroke-width="3"/>'
        '<rect x="18" y="160" width="52" height="23" rx="11.5" fill="none" stroke="currentColor" stroke-width="3" '
        'transform="rotate(-38 44 171)"/>'
        '</svg>'
    )


def palm_guide_card(caption: str = "Frame your palm like this — fingers spread, palm filling the frame"):
    """Standalone reference card shown above the capture widgets. Always rendered
    (guaranteed to work everywhere), independent of the best-effort live overlay."""
    st.markdown(
        f'<div class="anupt-palm-guide" style="color:var(--indigo)">{_palm_guide_inner_svg()}'
        f'<div class="pg-caption">{caption}</div></div>',
        unsafe_allow_html=True,
    )


def timeline_cards(pinnacles: list, challenges: list, current_index: int | None = None) -> str:
    """The 4 Pinnacle/Challenge life periods side by side (stacking on mobile
    via CSS grid auto-fit), with whichever period `current_index` points to
    visually highlighted as "You are here"."""
    cards = ""
    for i, (p, c) in enumerate(zip(pinnacles, challenges)):
        is_current = current_index is not None and i == current_index
        badge = '<span class="anupt-timeline-current-badge">You are here</span>' if is_current else ""
        cards += (
            f'<div class="anupt-timeline-card{" current" if is_current else ""}">'
            f'<div class="tc-period">Period {p["period"]}{badge}</div>'
            f'<div class="tc-age">Ages {p["age_range"]}</div>'
            f'<div class="tc-row"><span class="tc-label">Pinnacle</span>'
            f'<span class="tc-num">{p["number"]}</span></div>'
            f'<div class="tc-meaning">{p["meaning"]}</div>'
            f'<div class="tc-row"><span class="tc-label">Challenge</span>'
            f'<span class="tc-num challenge">{c["number"]}</span></div>'
            f'<div class="tc-meaning">{c["meaning"]}</div>'
            f'</div>'
        )
    return f'<div class="anupt-timeline">{cards}</div>'


_NUMEROLOGY_SCORE_BAND_COLORS = {
    "Strong": "score-high", "Good": "score-high", "Developing": "score-mid", "Growth Area": "score-low",
}


def score_bar(label: str, score: int, band: str, reason: str) -> str:
    """A 0-100 horizontal score bar with its plain-language band and the
    specific-numbers reason underneath — the visual half of the deterministic
    life-area scoring (engines/numerology_scoring.py provides the numbers)."""
    color_var = _NUMEROLOGY_SCORE_BAND_COLORS.get(band, "score-mid")
    return (
        f'<div class="anupt-score-bar-row">'
        f'<div class="sb-head"><span class="sb-label">{label}</span>'
        f'<span class="sb-value" style="color:var(--{color_var})">{score} · {band}</span></div>'
        f'<div class="anupt-score-bar-track">'
        f'<div class="anupt-score-bar-fill" style="width:{score}%;background:var(--{color_var})"></div></div>'
        f'<div class="anupt-score-bar-reason">{reason}</div>'
        f'</div>'
    )


def calculation_steps(label: str, entry: dict):
    """Renders one number's step-by-step arithmetic inside an expander —
    the calculation-transparency requirement. `entry` is one of
    full_profile()'s per-number dicts (has "calculation", "karmic_debt", etc)."""
    calc = entry.get("calculation") or {}
    steps = calc.get("steps") or []
    with st.expander(f"How {label} ({entry['value']}) was calculated"):
        for s in steps:
            st.markdown(f"- {s}")
        if entry.get("is_master"):
            st.caption(f"✦ {entry['value']} is a Master Number and isn't reduced further.")
        if entry.get("karmic_debt"):
            st.caption(f"⚠ Karmic Debt {entry['karmic_debt']} appeared during this reduction — "
                       f"{entry.get('karmic_debt_meaning', '')}")


_DIGNITY_CSS_CLASS = {
    "Exalted": "strong", "Rulership": "strong", "Own Sign": "strong", "Moolatrikona": "strong",
    "Detriment": "weak", "Fall": "weak", "Debilitated": "weak", "Detriment & Fall": "weak",
    "Neutral": "neutral",
}


def dignity_badge(dignity: str) -> str:
    css_class = _DIGNITY_CSS_CLASS.get(dignity, "neutral")
    return f'<span class="anupt-dignity-badge anupt-dignity-{css_class}">{dignity}</span>'


def rule_based_badge() -> str:
    """A small visual tag marking content as coming from the deterministic
    rule engine, not the AI — the concrete, always-visible answer to
    'why should I trust this' for anything wearing this badge."""
    return '<span class="anupt-rule-badge">⚙ Rule-Based — no AI involved</span>'


def key_finding_card(title: str, strength_label: str, strength_tier: int, interpretation: str,
                      basis: list, exceptions_applied: list | None = None):
    """One shared card layout for a rule-engine finding, used identically
    across Astrology and Numerology (and anywhere else a Finding -> Rule ->
    Evidence -> Interpretation -> Confidence chain needs to render) —
    'same card/layout structure across all engines' by construction, since
    every caller goes through this one function rather than each engine
    building its own bespoke markup."""
    basis_text = "; ".join(basis) if basis else ""
    exception_html = ""
    if exceptions_applied:
        exception_html = f'<div class="fc-exception">⚠ {" ".join(exceptions_applied)}</div>'
    st.markdown(
        f'<div class="anupt-finding-card">'
        f'<div class="fc-head"><span class="fc-title">{title}</span>'
        f'<span class="fc-tier fc-tier-{strength_tier}">{strength_label}</span></div>'
        f'<div class="fc-text">{interpretation}</div>'
        f'{exception_html}'
        f'<div class="fc-basis">Basis: {basis_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_ASPECT_SYMBOL = {
    "Conjunction": "☌", "Sextile": "⚹", "Square": "□", "Trine": "△", "Opposition": "☍",
    "Semi-sextile": "⚺", "Semi-square": "∠", "Quincunx": "⚻", "Sesquiquadrate": "⚼",
}


def aspect_list(aspects: list, point_a_key: str = "point_a", point_b_key: str = "point_b") -> str:
    """Renders a list of aspect dicts (from engines.astrology.calculate_aspects,
    or its transit/synastry variants with different point-name keys) as clean
    rows, closest-to-exact orb first — the caller has typically already
    sorted them, this just displays them."""
    if not aspects:
        return '<p class="anupt-caption">No aspects found within the current orbs.</p>'
    rows = ""
    for asp in aspects:
        symbol = _ASPECT_SYMBOL.get(asp["aspect"], "•")
        rows += (
            f'<div class="anupt-aspect-row">'
            f'<span class="anupt-aspect-points">{asp[point_a_key]} {symbol} {asp[point_b_key]}</span>'
            f'<span class="anupt-aspect-name">{asp["aspect"]}</span>'
            f'<span class="anupt-aspect-orb">orb {asp["orb"]}°</span>'
            f'</div>'
        )
    return f'<div>{rows}</div>'


def yoga_cards(yogas: list) -> str:
    if not yogas:
        return '<p class="anupt-caption">None of the yogas this app checks for are present in this chart — that\'s common and not a negative sign, these are just a handful of many possible classical combinations.</p>'
    cards = "".join(
        f'<div class="anupt-yoga-card"><div class="yc-name">{y["name"]}</div>'
        f'<div class="yc-desc">{y["description"]}</div></div>'
        for y in yogas
    )
    return f'<div class="anupt-yoga-grid">{cards}</div>'


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
