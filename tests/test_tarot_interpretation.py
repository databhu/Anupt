"""
Tests for engines/tarot_interpretation.py — the deterministic rule engine
for tarot readings. Same properties under test as the astrology/numerology
interpretation suites: determinism, and genuinely different output for
genuinely different inputs (Major vs Minor, upright vs reversed, position),
plus correct detection (and correct non-detection) of spread-level patterns.
"""

from engines import tarot
from engines import tarot_interpretation as interp


def _card(name="Test Card", arcana="Major", orientation="Upright", position="Present",
          suit=None, keywords="x"):
    return {"name": name, "arcana": arcana, "orientation": orientation, "position": position,
            "suit": suit, "keywords": keywords}


class TestInterpretCardDeterminism:
    def test_same_input_produces_identical_output(self):
        c = _card()
        assert interp.interpret_card(c) == interp.interpret_card(c)


class TestStrengthTiers:
    def test_major_upright_scores_highest(self):
        assert interp.interpret_card(_card(arcana="Major", orientation="Upright"))["strength_tier"] == 5

    def test_major_reversed_scores_second(self):
        assert interp.interpret_card(_card(arcana="Major", orientation="Reversed"))["strength_tier"] == 4

    def test_minor_upright_scores_third(self):
        assert interp.interpret_card(_card(arcana="Minor", orientation="Upright"))["strength_tier"] == 3

    def test_minor_reversed_scores_lowest(self):
        assert interp.interpret_card(_card(arcana="Minor", orientation="Reversed"))["strength_tier"] == 2

    def test_full_ordering_major_beats_minor_upright_beats_reversed(self):
        tiers = [
            interp.interpret_card(_card(arcana=a, orientation=o))["strength_tier"]
            for a, o in [("Major", "Upright"), ("Major", "Reversed"), ("Minor", "Upright"), ("Minor", "Reversed")]
        ]
        assert tiers == sorted(tiers, reverse=True)
        assert len(set(tiers)) == 4  # all four combinations genuinely distinct


class TestPositionAndOrientationDifferentiation:
    def test_same_card_different_position_differs(self):
        past = interp.interpret_card(_card(name="The Tower", position="Past"))
        future = interp.interpret_card(_card(name="The Tower", position="Future"))
        assert past["interpretation"] != future["interpretation"]

    def test_same_card_different_orientation_differs(self):
        up = interp.interpret_card(_card(name="The Sun", orientation="Upright"))
        rev = interp.interpret_card(_card(name="The Sun", orientation="Reversed"))
        assert up["interpretation"] != rev["interpretation"]

    def test_unknown_position_does_not_crash(self):
        result = interp.interpret_card(_card(position="Some Unlisted Position"))
        assert result["interpretation"].strip() != ""


class TestSpreadPatterns:
    def test_repeated_suit_detected(self):
        cards = [_card(suit="Cups"), _card(suit="Cups"), _card(arcana="Minor", suit="Wands")]
        patterns = interp.detect_spread_patterns(cards)
        assert any("Cups" in p["pattern"] for p in patterns)

    def test_no_pattern_when_suits_all_different(self):
        cards = [_card(arcana="Minor", suit="Cups"), _card(arcana="Minor", suit="Wands"),
                 _card(arcana="Minor", suit="Swords")]
        patterns = interp.detect_spread_patterns(cards)
        assert patterns == []

    def test_multiple_major_arcana_detected(self):
        cards = [_card(arcana="Major"), _card(arcana="Major"), _card(arcana="Minor", suit="Wands")]
        patterns = interp.detect_spread_patterns(cards)
        assert any("Major Arcana" in p["pattern"] for p in patterns)

    def test_single_major_arcana_not_flagged_as_pattern(self):
        # Only ONE major card shouldn't trigger the "multiple majors" pattern.
        cards = [_card(arcana="Major"), _card(arcana="Minor", suit="Wands"),
                 _card(arcana="Minor", suit="Cups")]
        patterns = interp.detect_spread_patterns(cards)
        assert not any("Major Arcana" in p["pattern"] for p in patterns)

    def test_all_reversed_detected_for_three_or_more_cards(self):
        cards = [_card(orientation="Reversed") for _ in range(3)]
        patterns = interp.detect_spread_patterns(cards)
        assert any("reversed" in p["pattern"].lower() for p in patterns)

    def test_not_all_reversed_does_not_trigger_pattern(self):
        cards = [_card(orientation="Reversed"), _card(orientation="Reversed"), _card(orientation="Upright")]
        patterns = interp.detect_spread_patterns(cards)
        assert not any("reversed" in p["pattern"].lower() for p in patterns)

    def test_single_card_spread_never_triggers_all_reversed_pattern(self):
        # A 1-card spread being "all reversed" isn't a meaningful pattern.
        cards = [_card(orientation="Reversed")]
        patterns = interp.detect_spread_patterns(cards)
        assert not any("reversed" in p["pattern"].lower() for p in patterns)

    def test_every_pattern_has_basis(self):
        cards = [_card(suit="Cups"), _card(suit="Cups")]
        for p in interp.detect_spread_patterns(cards):
            assert len(p["basis"]) >= 1


class TestBuildSpreadEvidence:
    def test_works_with_real_deterministic_draw(self):
        cards = tarot.draw_spread("test-reading-id-fixed", "three_card")
        evidence = interp.build_spread_evidence(cards)
        assert len(evidence["cards"]) == 3
        for c in evidence["cards"]:
            assert c["interpretation"].strip() != ""

    def test_cards_sorted_strongest_first(self):
        cards = tarot.draw_spread("another-fixed-id", "five_card")
        evidence = interp.build_spread_evidence(cards)
        tiers = [c["strength_tier"] for c in evidence["cards"]]
        assert tiers == sorted(tiers, reverse=True)

    def test_reproducible_across_calls(self):
        # Same reading_id -> same draw -> same evidence, every time.
        cards_a = tarot.draw_spread("reproducibility-check", "three_card")
        cards_b = tarot.draw_spread("reproducibility-check", "three_card")
        assert interp.build_spread_evidence(cards_a) == interp.build_spread_evidence(cards_b)

    def test_no_crash_across_many_spreads(self):
        for spread in ("one_card", "three_card", "five_card"):
            for seed in range(5):
                cards = tarot.draw_spread(f"seed-{seed}", spread)
                evidence = interp.build_spread_evidence(cards)
                assert len(evidence["cards"]) == len(cards)
