"""
Tests for ai/gemini_client.py's generate_unified_reading() — redesigned
per explicit feedback that the ANUPT combined reading should (a) always
work through each contributing engine's own answer before a final
synthesis, (b) read like an astrologer explaining WHY something is
indicated and what to do about it in plain language, and (c) never pad
out generic statements that could apply to anyone.
"""

import json
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


_SAMPLE_EVIDENCE = {
    "ranked_themes": ["career"],
    "theme_scores": {"career": {"strength": "Strong", "supporting_systems": ["Astrology", "Numerology"]}},
}
_SAMPLE_EVIDENCE_WITH_PALM = dict(_SAMPLE_EVIDENCE, palmistry={"findings": []})


def _sent_prompt(mock_post) -> str:
    return mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]


class TestPerEngineStructure:
    def test_names_astrology_numerology_tarot_when_no_palm(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "Astrology, Numerology, Tarot" in sent
        assert "Palmistry" not in sent.split("one short paragraph EACH for")[1].split(".")[0]

    def test_includes_palmistry_when_present_in_evidence(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE_WITH_PALM)
        sent = _sent_prompt(mock_post)
        assert "Astrology, Numerology, Tarot, Palmistry" in sent

    def test_requests_per_engine_paragraphs_then_final_synthesis(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "one short paragraph EACH for" in sent
        assert "final short paragraph" in sent
        assert "genuinely synthesizes" in sent


class TestAstrologerPersonaAndTips:
    def test_astrologer_persona_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "experienced astrologer" in sent.lower()

    def test_explain_why_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "explain WHY" in sent

    def test_simple_language_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "non-technical person immediately understands" in sent

    def test_actionable_tip_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "concrete, doable tip" in sent
        assert "vague platitudes" in sent

    def test_anti_generic_padding_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "generic statements that could apply to anyone" in sent


class TestQuestionAndTimeframeHandling:
    def test_specific_question_included_when_provided(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE, question="Will I get promoted this year?")
        sent = _sent_prompt(mock_post)
        assert "Will I get promoted this year?" in sent

    def test_falls_back_to_evidence_timeframe_when_no_question(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE, question=None)
        sent = _sent_prompt(mock_post)
        assert "whatever timeframe the evidence itself" in sent


class TestLanguageAndParsing:
    def test_marathi_instruction_still_included(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE, language="mr")
        sent = _sent_prompt(mock_post)
        assert "Marathi" in sent

    def test_score_summary_details_still_parsed_correctly(self):
        with patch("requests.post", return_value=_gemini_response(
                "8\n---SPLIT---\nShort summary.\n---SPLIT---\nAstrology suggests...\n\nFinal synthesis.")):
            score, summary, details = gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        assert score == 8
        assert summary == "Short summary."
        assert "Astrology suggests" in details

    def test_evidence_data_reaches_the_prompt(self):
        with patch("requests.post", return_value=_gemini_response("7\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_unified_reading("Life", _SAMPLE_EVIDENCE)
        sent = _sent_prompt(mock_post)
        assert "career" in sent
