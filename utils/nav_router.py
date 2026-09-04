"""
ANUPT — Navigation intent router for the "Ask ANUPT" chatbot.

Rule-based first: this module's job is to map a natural-language request
("show my palm reading", "life path number") to a page in the app using
plain keyword matching — no AI, no randomness, and a closed vocabulary of
valid destinations so a match can never point somewhere that doesn't
exist. ai/gemini_client.py's classify_navigation_intent() is only called
when THIS module finds no match at all — the "AI only when needed" half
of the flow.

Flow this implements: User Request -> Intent Detection (this module,
rule-based) -> Router (app.py, sets st.session_state.nav) -> Correct
Feature -> Reading.
"""

DESTINATIONS = ["Home", "Astrology", "Numerology", "Palmistry", "Tarot", "ANUPT", "Profile"]

DESTINATION_DESCRIPTIONS = {
    "Home": "Your dashboard — quick chart glance and today's reading",
    "Astrology": "Birth chart, planets, houses, dashas, transits",
    "Numerology": "Life Path, Destiny, Mulank/Bhagyank, and other core numbers",
    "Palmistry": "AI-analyzed palm photo readings",
    "Tarot": "Card draws and spreads",
    "ANUPT": "Combined reading across every system",
    "Profile": "Your birth details, palm photos, and settings",
}

# Unambiguous keywords: seeing any of these is enough to route directly,
# no disambiguation needed. Longer/more specific phrases are listed so a
# precise match (e.g. "life path number") isn't accidentally shadowed by
# a shorter, vaguer one.
_UNAMBIGUOUS_KEYWORDS = {
    "Palmistry": [
        "palm reading", "palmistry", "my palm", "read my palm", "read my hand",
        "hand reading", "palm map", "life line", "heart line", "head line",
        "fate line", "mount of venus", "mount of jupiter", "palm photo",
    ],
    "Numerology": [
        "numerology", "life path number", "life path", "destiny number",
        "mulank", "bhagyank", "soul urge", "personal year", "personal month",
        "karmic debt", "master number", "birthday number", "expression number",
        "pinnacle", "challenge number", "name number",
    ],
    "Astrology": [
        "astrology", "horoscope", "birth chart", "natal chart", "my chart",
        "planets", "planetary position", "houses", "ascendant", "rising sign",
        "moon sign", "sun sign", "zodiac sign", "dasha", "mahadasha", "transit",
        "nakshatra", "western chart", "vedic chart", "retrograde", "solar return",
        "aspect", "conjunction", "yoga in my chart",
    ],
    "Tarot": [
        "tarot", "tarot card", "draw a card", "card reading", "pull a card",
        "tarot spread", "card of the day",
    ],
    "ANUPT": [
        "combined reading", "overall reading", "full reading", "everything",
        "all systems", "complete picture", "anupt insight", "synastry",
        "compatibility check",
    ],
    "Profile": [
        "edit profile", "my profile", "my details", "birth details",
        "change my birth", "update profile", "account settings", "add palm photo",
    ],
    "Home": ["home", "dashboard", "go home", "main page"],
}

# Ambiguous life-area terms: several engines can meaningfully answer these,
# so the router surfaces a choice rather than guessing which one the user
# meant. Each maps to the destinations actually relevant to it. Split into
# domain-specific terms (career, love, ...) and generic terms (prediction,
# future, ...) so a query matching both — "career prediction" — prefers the
# more specific, more meaningful candidate list rather than whichever
# string happens to be longer.
_AMBIGUOUS_KEYWORDS_SPECIFIC = {
    "career": ["Astrology", "Numerology", "ANUPT"],
    "job": ["Astrology", "Numerology", "ANUPT"],
    "work": ["Astrology", "Numerology", "ANUPT"],
    "finance": ["Astrology", "Numerology", "ANUPT"],
    "money": ["Astrology", "Numerology", "ANUPT"],
    "wealth": ["Astrology", "Numerology", "ANUPT"],
    "relationship": ["Astrology", "Numerology", "Tarot", "ANUPT"],
    "love": ["Astrology", "Numerology", "Tarot", "ANUPT"],
    "marriage": ["Astrology", "Numerology", "ANUPT"],
    "compatibility": ["Astrology", "ANUPT"],
    "personal growth": ["Astrology", "Numerology", "ANUPT"],
    "life timeline": ["Astrology", "Numerology"],
    "timeline": ["Astrology", "Numerology"],
}
_AMBIGUOUS_KEYWORDS_GENERIC = {
    "future": ["Astrology", "Numerology", "Tarot", "ANUPT"],
    "prediction": ["Astrology", "Numerology", "Tarot", "ANUPT"],
}
_AMBIGUOUS_KEYWORDS = {**_AMBIGUOUS_KEYWORDS_SPECIFIC, **_AMBIGUOUS_KEYWORDS_GENERIC}

# Follow-up phrases: resolved using the PREVIOUS turn's destination rather
# than matched against keywords — "tell me more" means nothing on its own.
_FOLLOWUP_PHRASES = [
    "detailed reading", "tell me more", "go deeper", "more detail", "more details",
    "explore further", "show more", "full details", "advanced analysis",
]


def detect_intent(query: str, last_destination: str | None = None) -> dict:
    """Rule-based intent detection. Returns one of three shapes:
    - {"matched": True, "destination": <page>, "reason": <which keyword hit>}
    - {"matched": "ambiguous", "candidates": [<page>, ...], "term": <matched term>}
    - {"matched": False} — no rule matched; caller should fall back to AI
      (ai.gemini_client.classify_navigation_intent) or show all options."""
    q = query.lower().strip()
    if not q:
        return {"matched": False}

    if any(phrase in q for phrase in _FOLLOWUP_PHRASES) and last_destination:
        return {"matched": True, "destination": last_destination, "reason": "follow-up on previous topic"}

    # Unambiguous keywords first, longest phrase first within each destination
    # so a specific match (e.g. "life path number") wins over a shorter one
    # that might otherwise be checked first by dict ordering alone.
    all_matches = []
    for destination, phrases in _UNAMBIGUOUS_KEYWORDS.items():
        for phrase in phrases:
            if phrase in q:
                all_matches.append((len(phrase), destination, phrase))
    if all_matches:
        all_matches.sort(reverse=True)
        _, destination, phrase = all_matches[0]
        return {"matched": True, "destination": destination, "reason": f"matched '{phrase}'"}

    # Ambiguous life-area terms — surface a choice instead of guessing.
    # Domain-specific terms (career, love, ...) take priority over generic
    # ones (prediction, future, ...) so "career prediction" surfaces
    # career's candidate list, not prediction's broader/less precise one.
    specific_matches = [(len(term), term) for term in _AMBIGUOUS_KEYWORDS_SPECIFIC if term in q]
    if specific_matches:
        specific_matches.sort(reverse=True)
        _, term = specific_matches[0]
        return {"matched": "ambiguous", "candidates": _AMBIGUOUS_KEYWORDS[term], "term": term}

    generic_matches = [(len(term), term) for term in _AMBIGUOUS_KEYWORDS_GENERIC if term in q]
    if generic_matches:
        generic_matches.sort(reverse=True)
        _, term = generic_matches[0]
        return {"matched": "ambiguous", "candidates": _AMBIGUOUS_KEYWORDS[term], "term": term}

    return {"matched": False}
