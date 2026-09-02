"""
ANUPT — Tarot Engine
Card data and meanings are fixed. Draws use a seeded RNG derived from the
reading's own ID so a saved reading is always reproducible — re-opening
the same reading_id yields the identical cards and orientations.
No AI involved in the draw itself — only in later interpretation.
"""

import json
import os
import random

_DECK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tarot_deck.json")
with open(_DECK_PATH) as f:
    DECK = json.load(f)

SPREADS = {
    "one_card": ["Guidance"],
    "three_card": ["Past", "Present", "Future"],
    "five_card": ["Situation", "Challenge", "Past Influence", "Advice", "Likely Outcome"],
}


def draw_spread(reading_id: str, spread: str) -> list[dict]:
    """
    Deterministic draw: same reading_id + spread always returns the same cards,
    orientations, and order (reproducibility requirement).
    """
    if spread not in SPREADS:
        raise ValueError(f"Unknown spread: {spread}")
    positions = SPREADS[spread]
    rng = random.Random(reading_id + "::" + spread)
    indices = rng.sample(range(len(DECK)), len(positions))
    orientations = [rng.random() < 0.5 for _ in positions]  # True = reversed

    result = []
    for pos, idx, reversed_ in zip(positions, indices, orientations):
        card = DECK[idx]
        result.append({
            "position": pos,
            "name": card["name"],
            "arcana": card["arcana"],
            "suit": card["suit"],
            "orientation": "Reversed" if reversed_ else "Upright",
            "keywords": card["reversed_keywords"] if reversed_ else card["upright_keywords"],
            "career": card["career"],
            "finance": card["finance"],
            "love": card["love"],
        })
    return result


def deck_completeness_check() -> bool:
    names = {c["name"] for c in DECK}
    return len(DECK) == 78 and len(names) == 78
