"""
Tests for ai/gemini_client.py's generate_engine_reading() — redesigned to
carry the same astrologer-persona/explain-why/actionable-tip/no-generic-
padding instructions as the ANUPT combined reading (test_unified_reading_
prompt.py), since the feedback driving this applies to every reading
page, not just the combined one.
"""

import logging
import os
from unittest.mock import MagicMock, patch

logging.disable(logging.CRITICAL)
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")

from ai import gemini_client as gc


def _gemini_response(text: str):
    r = MagicMock()
    r.status_code = 200
    r.json = lambda: {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return r


_SAMPLE_DATA = {"context": {"nakshatra": "Mula"}, "rule_based_findings": {"placements": []}}


def _sent_prompt(mock_post) -> str:
    return mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]


class TestAstrologerPersonaAndTips:
    def test_astrologer_persona_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Astrology", _SAMPLE_DATA, "Life")
        sent = _sent_prompt(mock_post)
        assert "experienced astrologer" in sent.lower()

    def test_explain_why_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Numerology", _SAMPLE_DATA, "Life")
        sent = _sent_prompt(mock_post)
        assert "Explain WHY" in sent

    def test_actionable_tip_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Tarot", _SAMPLE_DATA, "Life")
        sent = _sent_prompt(mock_post)
        assert "concrete, doable tip" in sent

    def test_anti_generic_padding_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Astrology", _SAMPLE_DATA, "Life")
        sent = _sent_prompt(mock_post)
        assert "generic statements that could apply to anyone" in sent

    def test_single_engine_scope_instruction_still_present(self):
        # A pre-existing, important instruction this redesign must not
        # have accidentally dropped: single-engine pages must not
        # reference OTHER systems.
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Astrology", _SAMPLE_DATA, "Life")
        sent = _sent_prompt(mock_post)
        assert "do not reference" in sent.lower()

    def test_rule_based_findings_synthesis_instruction_still_present(self):
        # Another pre-existing instruction that must survive this change.
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Astrology", _SAMPLE_DATA, "Life")
        sent = _sent_prompt(mock_post)
        assert "SYNTHESIZE and PERSONALIZE" in sent


class TestFocusAndQuestionHandling:
    def test_focus_area_still_included(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Numerology", _SAMPLE_DATA, "Life", focus="career")
        sent = _sent_prompt(mock_post)
        assert "career" in sent

    def test_question_still_included(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Astrology", _SAMPLE_DATA, "Life", question="Will I move abroad?")
        sent = _sent_prompt(mock_post)
        assert "Will I move abroad?" in sent

    def test_marathi_instruction_still_included(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_engine_reading("Astrology", _SAMPLE_DATA, "Life", language="mr")
        sent = _sent_prompt(mock_post)
        assert "Marathi" in sent


class TestParsing:
    def test_score_summary_details_still_parsed_correctly(self):
        with patch("requests.post", return_value=_gemini_response(
                "6\n---SPLIT---\nA brief answer.\n---SPLIT---\nFuller explanation with a tip.")):
            score, summary, details = gc.generate_engine_reading("Astrology", _SAMPLE_DATA, "Life")
        assert score == 6
        assert summary == "A brief answer."
        assert "tip" in details
