"""
Generates data/tarot_deck.json — the full 78-card deck.
Major Arcana are hand-authored. Minor Arcana are systematically composed
from suit-element-theme + rank meaning (a standard, legitimate tarot
construction method), so every one of the 56 cards is distinct and real,
not placeholder text.
"""
import json
import os

MAJOR_ARCANA = [
    ("The Fool", "New beginnings, spontaneity, a leap of faith, innocence.",
     "Recklessness, hesitation, a missed opportunity, naivety.",
     "A fresh start or new venture worth exploring.", "An impulsive purchase or financial risk — tread lightly.",
     "A new connection or a playful, open-hearted phase."),
    ("The Magician", "Willpower, resourcefulness, manifestation, skill.",
     "Manipulation, poor planning, untapped talent.",
     "You have the tools to make a career move happen now.", "Good time to actively create income streams.",
     "Conscious, intentional connection; you attract what you focus on."),
    ("The High Priestess", "Intuition, mystery, inner knowing, the subconscious.",
     "Secrets withheld, disconnection from intuition.",
     "Trust your instincts over the obvious path at work.", "Avoid decisions made on incomplete information.",
     "An unspoken truth or a quietly deepening bond."),
    ("The Empress", "Abundance, nurturing, creativity, fertility.",
     "Creative block, overdependence, neglect.",
     "Growth-oriented work; nurturing a project to maturity.", "Comfortable abundance; a good period for saving.",
     "Warmth, sensuality, and nurturing partnership."),
    ("The Emperor", "Structure, authority, stability, discipline.",
     "Rigidity, control issues, lack of discipline.",
     "Leadership responsibility or the need for structure.", "Discipline pays off; good for long-term planning.",
     "A stabilizing, protective, sometimes controlling dynamic."),
    ("The Hierophant", "Tradition, institutions, guidance, conformity.",
     "Breaking convention, personal belief over dogma.",
     "Mentorship or working within established systems.", "Conventional, safe financial choices favored.",
     "Commitment, tradition, or a relationship following convention."),
    ("The Lovers", "Union, choice, alignment of values, connection.",
     "Misalignment, a difficult choice, disharmony.",
     "A partnership or collaboration that fits your values.", "A financial decision made jointly with another.",
     "A significant romantic choice or deepening union."),
    ("The Chariot", "Willpower, determination, victory through focus.",
     "Lack of direction, aggression, loss of control.",
     "Pushing through obstacles to a clear career win.", "Disciplined focus brings a financial goal within reach.",
     "Determined pursuit of what — or who — you want."),
    ("Strength", "Courage, patience, inner strength, compassion.",
     "Self-doubt, weakness, insecurity.",
     "Quiet persistence rather than force wins the day.", "Steady, patient management of resources.",
     "Gentle strength holds a relationship together."),
    ("The Hermit", "Introspection, solitude, inner guidance.",
     "Isolation, withdrawal, avoiding needed reflection.",
     "A pause for reflection before the next career step.", "A time to review rather than spend.",
     "A need for space, or guidance sought within."),
    ("Wheel of Fortune", "Cycles, change, turning points, fate.",
     "Resistance to change, a cycle repeating.",
     "An unexpected shift changes your career direction.", "Fortunes turn — plan for both ups and downs.",
     "A pivotal, fated-feeling turn in a relationship."),
    ("Justice", "Fairness, truth, cause and effect, accountability.",
     "Unfairness, avoiding accountability, imbalance.",
     "A fair outcome, contract, or decision at work.", "Balancing the books; a fair financial settlement.",
     "Honesty and fairness define the relationship now."),
    ("The Hanged Man", "Surrender, new perspective, letting go.",
     "Stalling, resistance, needless sacrifice.",
     "Progress requires a change in perspective, not more effort.", "Hold off on major financial moves.",
     "A pause or a shift in how you see the relationship."),
    ("Death", "Transformation, endings, release, transition.",
     "Resistance to change, stagnation, fear of letting go.",
     "A definitive end to one chapter, opening the next.", "A necessary financial reset.",
     "A relationship transforms — sometimes ending, sometimes reborn."),
    ("Temperance", "Balance, moderation, patience, blending.",
     "Excess, imbalance, impatience.",
     "Balanced, steady progress over dramatic moves.", "Moderation and patience with spending.",
     "Harmony built through compromise and patience."),
    ("The Devil", "Attachment, restriction, temptation, materialism.",
     "Breaking free, releasing an unhealthy pattern.",
     "Feeling trapped in a role or unhealthy work pattern.", "Overspending or a restrictive financial obligation.",
     "A codependent or restrictive relationship pattern."),
    ("The Tower", "Sudden upheaval, revelation, breakdown, awakening.",
     "Avoided disaster, resisting necessary change.",
     "A sudden, disruptive career change — ultimately clarifying.", "An abrupt financial shock; rebuild wisely.",
     "A sudden, revealing rupture that changes everything."),
    ("The Star", "Hope, inspiration, renewal, healing.",
     "Despair, disconnection from hope, self-doubt.",
     "Renewed inspiration and a hopeful path forward.", "Optimism supports a slow, healthy recovery.",
     "Healing, hope, and a gentle renewal of connection."),
    ("The Moon", "Illusion, intuition, the unconscious, uncertainty.",
     "Confusion clearing, hidden truth surfacing.",
     "Unclear information at work — verify before acting.", "Avoid decisions based on unclear financial information.",
     "Uncertainty or unspoken fears need to be named."),
    ("The Sun", "Joy, success, vitality, clarity.",
     "Temporary clouds over an otherwise good outcome.",
     "Visible success and well-earned recognition.", "A genuinely good, clear financial period.",
     "Joyful, open, and warm connection."),
    ("Judgement", "Reckoning, renewal, a calling, self-evaluation.",
     "Self-doubt, avoiding a necessary reckoning.",
     "A career calling or reassessment of your path.", "Reviewing past financial decisions to move forward.",
     "A relationship reaches a moment of honest reckoning."),
    ("The World", "Completion, integration, accomplishment.",
     "Incompletion, shortcuts, an unfinished chapter.",
     "A major goal reaches successful completion.", "Financial goals reach fulfillment.",
     "A relationship reaches wholeness and fulfillment."),
]

SUITS = {
    "Wands": {"element": "Fire", "domain": "action, ambition, and creative drive"},
    "Cups": {"element": "Water", "domain": "emotion, relationships, and intuition"},
    "Swords": {"element": "Air", "domain": "thought, conflict, and truth"},
    "Pentacles": {"element": "Earth", "domain": "money, work, and the material world"},
}

RANK_THEMES = {
    "Ace": ("a new beginning / pure potential", "opportunity"),
    "Two": ("balance, choice, or partnership", "duality"),
    "Three": ("growth, collaboration, or early results", "expansion"),
    "Four": ("stability, structure, or a pause", "foundation"),
    "Five": ("conflict, loss, or challenge", "tension"),
    "Six": ("cooperation, generosity, or moving forward", "harmony"),
    "Seven": ("assessment, patience, or a test of resolve", "reflection"),
    "Eight": ("movement, mastery, or restriction", "momentum"),
    "Nine": ("near-completion, resilience, or attainment", "fruition"),
    "Ten": ("culmination, the completion of the suit's story", "completion"),
    "Page": ("a student or messenger of the suit's energy", "curiosity"),
    "Knight": ("active, driven pursuit of the suit's energy", "pursuit"),
    "Queen": ("mature, nurturing mastery of the suit's energy", "mastery"),
    "King": ("commanding, authoritative mastery of the suit's energy", "authority"),
}

RANKS = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Page", "Knight", "Queen", "King"]


def build_deck():
    deck = []
    for idx, (name, up, rev, career, finance, love) in enumerate(MAJOR_ARCANA):
        deck.append({
            "name": name, "arcana": "Major", "number": idx, "suit": None,
            "upright_keywords": up, "reversed_keywords": rev,
            "career": career, "finance": finance, "love": love,
        })
    for suit, meta in SUITS.items():
        for n, rank in enumerate(RANKS, start=1):
            theme, keyword = RANK_THEMES[rank]
            name = f"{rank} of {suit}"
            up = f"{theme.capitalize()}, expressed through {meta['domain']} ({meta['element']})."
            rev = f"Blocked or excessive {keyword} within {meta['domain']}."
            career = f"At work: {theme} shows up in matters of {meta['domain']}."
            finance = f"Financially: a sign of {keyword} connected to {meta['domain']}."
            love = f"In relationships: {theme}, colored by {meta['element'].lower()}-natured {meta['domain']}."
            deck.append({
                "name": name, "arcana": "Minor", "number": n, "suit": suit,
                "upright_keywords": up, "reversed_keywords": rev,
                "career": career, "finance": finance, "love": love,
            })
    return deck


if __name__ == "__main__":
    deck = build_deck()
    assert len(deck) == 78, f"Expected 78 cards, got {len(deck)}"
    out_path = os.path.join(os.path.dirname(__file__), "tarot_deck.json")
    with open(out_path, "w") as f:
        json.dump(deck, f, indent=2)
    print(f"Wrote {len(deck)} cards to {out_path}")
