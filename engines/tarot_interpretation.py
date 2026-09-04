"""
ANUPT — Tarot rule-based interpretation engine.

Same role as the astrology/numerology interpretation engines: the "Rule
Engine" step of Rule Engine -> Evidence -> AI -> Final Reading. Everything
here is deterministic, built from the already-deterministic card data in
engines/tarot.py (keywords, career/finance/love meanings — fixed in
data/tarot_deck.json, never AI-generated) plus documented tarot
conventions: Major Arcana cards are traditionally read as carrying more
weight than Minor Arcana ("big life themes" vs "everyday, more within
your control"), and a spread with genuine PATTERNS across its cards
(repeated suits, multiple Major Arcana together) is traditionally read
differently than the same cards considered one at a time.
"""

from collections import Counter

# What each spread position contributes to a card's meaning — genuinely
# different framing, not just a label.
POSITION_CONTEXT = {
    "Guidance": "offers direct guidance for right now",
    "Past": "shapes what's already in motion, whether or not it's still visible",
    "Present": "reflects your current situation directly",
    "Future": "points toward where things are heading if the current path continues",
    "Situation": "frames the core circumstance at hand",
    "Challenge": "names the central obstacle to work through",
    "Past Influence": "shapes what's already in motion, whether or not it's still visible",
    "Advice": "offers direct guidance for how to move forward",
    "Likely Outcome": "points toward where things are heading if the current path continues",
}

SUIT_THEMES = {
    "Wands": "action, passion, and creative drive",
    "Cups": "emotion, relationships, and inner life",
    "Swords": "thought, conflict, and communication",
    "Pentacles": "material matters, work, and practical resources",
}

STRENGTH_TIER_LANGUAGE = {
    5: "Major Arcana, Upright", 4: "Major Arcana, Reversed",
    3: "Minor Arcana, Upright", 2: "Minor Arcana, Reversed",
}


def interpret_card(card: dict) -> dict:
    """Rule-based interpretation for one drawn card (from
    engines.tarot.draw_spread()'s output). Strength tier reflects the
    Major/Minor Arcana convention — Major cards are traditionally read as
    the bigger, less everyday-controllable signals."""
    is_major = card["arcana"] == "Major"
    is_reversed = card["orientation"] == "Reversed"
    tier = (5 if not is_reversed else 4) if is_major else (3 if not is_reversed else 2)

    position_clause = POSITION_CONTEXT.get(card["position"], "speaks to this part of the spread")
    orientation_clause = (
        "Upright, this energy tends to express directly and outwardly."
        if not is_reversed else
        "Reversed, this energy tends to turn inward, feel blocked, or show up in a "
        "delayed or internalized way rather than plainly."
    )
    weight_clause = (
        "As a Major Arcana card, this tends to signal a bigger, less everyday-controllable theme."
        if is_major else
        "As a Minor Arcana card, this tends to reflect a more everyday, situational influence."
    )

    interpretation = (
        f"{card['name']} ({card['orientation']}) in the {card['position']} position "
        f"{position_clause}. Key themes: {card['keywords']} {orientation_clause} {weight_clause}"
    )
    basis = [f"{card['name']} drawn {card['orientation']} in position '{card['position']}'",
             f"{'Major' if is_major else 'Minor'} Arcana"]

    return {
        "name": card["name"], "position": card["position"], "orientation": card["orientation"],
        "arcana": card["arcana"], "suit": card.get("suit"),
        "strength_tier": tier, "strength_label": STRENGTH_TIER_LANGUAGE[tier],
        "interpretation": interpretation, "basis": basis,
    }


def detect_spread_patterns(cards: list) -> list:
    """Genuine combination rules ACROSS cards in a spread, not just
    per-card meaning: repeated suits and multiple Major Arcana are both
    traditionally read as significant patterns in their own right."""
    patterns = []

    suits = [c["suit"] for c in cards if c.get("suit")]
    suit_counts = Counter(suits)
    for suit, count in suit_counts.items():
        if count >= 2:
            patterns.append({
                "pattern": f"{count}x {suit}",
                "interpretation": f"{count} {suit} cards appear in this spread, reinforcing "
                                  f"themes of {SUIT_THEMES.get(suit, suit)} — likely a genuine "
                                  "throughline, not a coincidence worth ignoring.",
                "basis": [f"{count} cards share the {suit} suit"],
            })

    major_count = sum(1 for c in cards if c["arcana"] == "Major")
    if major_count >= 2:
        patterns.append({
            "pattern": f"{major_count}x Major Arcana",
            "interpretation": f"{major_count} Major Arcana cards appear together in this spread, "
                              "traditionally suggesting a period of more significant, less "
                              "everyday-controllable change than a spread of mostly Minor Arcana cards.",
            "basis": [f"{major_count} of {len(cards)} cards are Major Arcana"],
        })

    reversed_count = sum(1 for c in cards if c["orientation"] == "Reversed")
    if len(cards) >= 3 and reversed_count == len(cards):
        patterns.append({
            "pattern": "All cards reversed",
            "interpretation": "Every card in this spread is reversed — traditionally read as a "
                              "period where energy generally feels blocked, delayed, or turned "
                              "inward across the board, not just in one area.",
            "basis": ["all drawn cards are reversed"],
        })

    return patterns


def build_spread_evidence(cards: list) -> dict:
    """Runs interpret_card() over every card plus detect_spread_patterns()
    across the whole spread — the full rule-based 'Evidence' layer, ready
    for the UI or for the AI to synthesize from."""
    card_findings = [interpret_card(c) for c in cards]
    card_findings.sort(key=lambda c: c["strength_tier"], reverse=True)
    patterns = detect_spread_patterns(cards)
    return {"cards": card_findings, "patterns": patterns}
