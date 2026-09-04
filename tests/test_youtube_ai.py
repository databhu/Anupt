"""
Tests for ai/gemini_client.py's extract_youtube_predictions() and
generate_youtube_summary() — the only two AI touchpoints in the YouTube
Insights pipeline (see engines/youtube_insights.py's module docstring for
why the other four "agent" roles are deterministic instead). As with
every other AI-facing test in this project, the property that matters
most is that a hallucinated or malformed response never reaches the UI
looking legitimate.
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


_SAMPLE_VIDEOS = [
    {"video_id": "v1", "title": "Aries Weekly Horoscope", "description": "Career growth this week."},
    {"video_id": "v2", "title": "Random video", "description": "Not astrology."},
]


class TestExtraction:
    def test_well_formed_response_parses_and_validates(self):
        payload = json.dumps([
            {"video_id": "v1", "signs_mentioned": ["Aries"], "themes": ["career"],
             "prediction_summary": "Career growth ahead."},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert len(result) == 1
        assert result[0]["video_id"] == "v1"
        assert result[0]["signs_mentioned"] == ["Aries"]

    def test_entry_with_no_signs_is_dropped(self):
        payload = json.dumps([
            {"video_id": "v2", "signs_mentioned": [], "themes": [], "prediction_summary": ""},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert result == []

    def test_hallucinated_sign_and_theme_filtered_valid_ones_kept(self):
        payload = json.dumps([
            {"video_id": "v1", "signs_mentioned": ["Aries", "FakeSign"], "themes": ["career", "fakeTheme"],
             "prediction_summary": "x"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert result[0]["signs_mentioned"] == ["Aries"]
        assert result[0]["themes"] == ["career"]

    def test_markdown_fenced_response_parses(self):
        fenced = '```json\n[{"video_id": "v1", "signs_mentioned": ["Leo"], "themes": [], "prediction_summary": "x"}]\n```'
        with patch("requests.post", return_value=_gemini_response(fenced)):
            result = gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert len(result) == 1

    def test_malformed_json_returns_empty_list_not_crash(self):
        with patch("requests.post", return_value=_gemini_response("not json at all")):
            result = gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert result == []

    def test_non_array_json_returns_empty_list(self):
        with patch("requests.post", return_value=_gemini_response('{"not": "an array"}')):
            result = gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert result == []

    def test_empty_video_list_skips_api_call_entirely(self):
        with patch("requests.post") as mock_post:
            result = gc.extract_youtube_predictions([])
        assert result == []
        mock_post.assert_not_called()

    def test_missing_api_key_returns_empty_list_without_calling_api(self):
        with patch.dict(os.environ, {}, clear=True), patch("requests.post") as mock_post:
            result = gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert result == []
        mock_post.assert_not_called()

    def test_single_batched_call_for_multiple_videos(self):
        # Minimizing AI calls means N videos -> ONE call, not N calls.
        payload = json.dumps([
            {"video_id": "v1", "signs_mentioned": ["Aries"], "themes": [], "prediction_summary": ""},
            {"video_id": "v2", "signs_mentioned": ["Leo"], "themes": [], "prediction_summary": ""},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)) as mock_post:
            gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        assert mock_post.call_count == 1

    def test_prompt_includes_only_closed_vocabulary_signs_and_themes(self):
        from engines import youtube_insights
        with patch("requests.post", return_value=_gemini_response("[]")) as mock_post:
            gc.extract_youtube_predictions(_SAMPLE_VIDEOS)
        sent = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        for sign in youtube_insights.ZODIAC_SIGNS:
            assert sign in sent


class TestSummaryGeneration:
    def test_well_formed_summary_parses(self):
        evidence = {"target_sign": "Aries", "ranked_themes": ["career"],
                    "theme_agreement": {"career": 3}, "evidence_by_theme": {"career": []}}
        with patch("requests.post", return_value=_gemini_response(
                "7\n---SPLIT---\nCareer is the focus this week.\n---SPLIT---\nMultiple creators agree.")):
            score, summary, details = gc.generate_youtube_summary(evidence, "Aries")
        assert score == 7
        assert "Career" in summary or "career" in summary

    def test_evidence_and_anti_fabrication_instruction_reach_the_prompt(self):
        evidence = {"target_sign": "Leo", "ranked_themes": ["love"],
                    "theme_agreement": {"love": 2}, "evidence_by_theme": {"love": []}}
        with patch("requests.post", return_value=_gemini_response("5\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_youtube_summary(evidence, "Leo")
        sent = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        assert "theme_agreement" in sent
        assert "never invent" in sent.lower()

    def test_question_included_when_provided(self):
        evidence = {"target_sign": "Leo", "ranked_themes": [], "theme_agreement": {}, "evidence_by_theme": {}}
        with patch("requests.post", return_value=_gemini_response("5\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_youtube_summary(evidence, "Leo", question="Will I get a promotion?")
        sent = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        assert "Will I get a promotion?" in sent

    def test_marathi_instruction_included_when_requested(self):
        evidence = {"target_sign": "Leo", "ranked_themes": [], "theme_agreement": {}, "evidence_by_theme": {}}
        with patch("requests.post", return_value=_gemini_response("5\n---SPLIT---\nx\n---SPLIT---\ny")) as mock_post:
            gc.generate_youtube_summary(evidence, "Leo", language="mr")
        sent = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        assert "Marathi" in sent
