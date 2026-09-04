"""
Tests for utils/nav_router.py — the rule-based navigation intent detector
behind the "Ask ANUPT" chatbot. The exact phrasings from the feature
request are tested explicitly (see TestRequestExamples), plus broader
coverage of unambiguous routing, ambiguous disambiguation, follow-up
context resolution, and no-match fallback.
"""

from utils import nav_router as nr


class TestRequestExamples:
    """The four example phrasings named directly in the feature request."""

    def test_show_my_palm_reading(self):
        result = nr.detect_intent("Show my palm reading")
        assert result["matched"] is True
        assert result["destination"] == "Palmistry"

    def test_career_prediction_is_ambiguous(self):
        result = nr.detect_intent("Career prediction")
        assert result["matched"] == "ambiguous"
        assert "Astrology" in result["candidates"]
        assert "Numerology" in result["candidates"]

    def test_life_path_number(self):
        result = nr.detect_intent("Life path number")
        assert result["matched"] is True
        assert result["destination"] == "Numerology"

    def test_relationship_reading_is_ambiguous(self):
        result = nr.detect_intent("Relationship reading")
        assert result["matched"] == "ambiguous"
        assert "Astrology" in result["candidates"]
        assert "Numerology" in result["candidates"]


class TestUnambiguousRouting:
    def test_astrology_terms(self):
        for q in ["my birth chart", "what's my ascendant", "moon sign", "dasha timeline"]:
            result = nr.detect_intent(q)
            assert result["matched"] is True, q
            assert result["destination"] == "Astrology", q

    def test_numerology_terms(self):
        for q in ["my mulank", "bhagyank number", "master number", "karmic debt check"]:
            result = nr.detect_intent(q)
            assert result["matched"] is True, q
            assert result["destination"] == "Numerology", q

    def test_palmistry_terms(self):
        for q in ["read my hand", "palm reading please", "my life line"]:
            result = nr.detect_intent(q)
            assert result["matched"] is True, q
            assert result["destination"] == "Palmistry", q

    def test_tarot_terms(self):
        for q in ["draw a tarot card", "pull a card for me", "card of the day"]:
            result = nr.detect_intent(q)
            assert result["matched"] is True, q
            assert result["destination"] == "Tarot", q

    def test_anupt_terms(self):
        for q in ["give me the combined reading", "overall reading please", "everything at once"]:
            result = nr.detect_intent(q)
            assert result["matched"] is True, q
            assert result["destination"] == "ANUPT", q

    def test_profile_terms(self):
        for q in ["edit my profile", "update my birth details", "account settings"]:
            result = nr.detect_intent(q)
            assert result["matched"] is True, q
            assert result["destination"] == "Profile", q

    def test_case_insensitive(self):
        result = nr.detect_intent("SHOW MY PALM READING")
        assert result["matched"] is True
        assert result["destination"] == "Palmistry"

    def test_specific_phrase_beats_shorter_generic_one(self):
        # "life path number" (Numerology-specific) should win over any
        # shorter/coincidental overlap.
        result = nr.detect_intent("what is my life path number")
        assert result["destination"] == "Numerology"


class TestAmbiguousDisambiguation:
    def test_finance_is_ambiguous(self):
        result = nr.detect_intent("how's my money situation")
        assert result["matched"] == "ambiguous"
        assert set(result["candidates"]) <= set(nr.DESTINATIONS)

    def test_specific_domain_term_beats_generic_term(self):
        # "career" (specific) should be picked over "prediction" (generic)
        # when both appear, since it gives a more precise candidate list.
        result = nr.detect_intent("give me a career prediction")
        assert result["term"] == "career"
        assert "Tarot" not in result["candidates"]

    def test_generic_term_still_matches_alone(self):
        result = nr.detect_intent("what does the future hold")
        assert result["matched"] == "ambiguous"
        assert result["term"] == "future"

    def test_ambiguous_candidates_are_all_valid_destinations(self):
        for term, candidates in nr._AMBIGUOUS_KEYWORDS.items():
            for c in candidates:
                assert c in nr.DESTINATIONS, f"{term} -> {c} is not a valid destination"


class TestFollowUpContext:
    def test_followup_uses_previous_destination(self):
        result = nr.detect_intent("tell me more", last_destination="Astrology")
        assert result["matched"] is True
        assert result["destination"] == "Astrology"

    def test_followup_without_context_does_not_match(self):
        result = nr.detect_intent("tell me more", last_destination=None)
        assert result["matched"] is False

    def test_detailed_reading_followup(self):
        result = nr.detect_intent("show detailed reading", last_destination="Numerology")
        assert result["destination"] == "Numerology"


class TestNoMatch:
    def test_unrelated_query_does_not_match(self):
        result = nr.detect_intent("what's the weather like today")
        assert result["matched"] is False

    def test_empty_query_does_not_match(self):
        assert nr.detect_intent("")["matched"] is False
        assert nr.detect_intent("   ")["matched"] is False

    def test_no_match_has_no_stray_keys(self):
        result = nr.detect_intent("asdkjfh qwerty nonsense")
        assert result == {"matched": False}


class TestDestinationIntegrity:
    def test_every_unambiguous_destination_is_valid(self):
        for destination in nr._UNAMBIGUOUS_KEYWORDS:
            assert destination in nr.DESTINATIONS

    def test_every_destination_has_a_description(self):
        for destination in nr.DESTINATIONS:
            assert destination in nr.DESTINATION_DESCRIPTIONS
            assert nr.DESTINATION_DESCRIPTIONS[destination].strip() != ""

    def test_no_keyword_is_empty_string(self):
        for phrases in nr._UNAMBIGUOUS_KEYWORDS.values():
            for phrase in phrases:
                assert phrase.strip() != ""
