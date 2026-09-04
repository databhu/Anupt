"""
Tests for ai/gemini_client.py's classify_navigation_intent() — the AI
fallback for the "Ask ANUPT" chatbot, only ever called when
utils.nav_router's rule-based matching finds nothing. The property under
test throughout: a destination outside the closed vocabulary must never
be trusted, no matter how plausible the model's response looks — the
navigation router can only ever point somewhere that actually exists.
"""

import logging
import os
from unittest.mock import MagicMock, patch

logging.disable(logging.CRITICAL)
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")

from ai import gemini_client as gc
from utils import nav_router as nr


def _gemini_response(text: str):
    r = MagicMock()
    r.status_code = 200
    r.json = lambda: {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return r


class TestWellFormedResponses:
    def test_valid_destination_returned(self):
        with patch("requests.post", return_value=_gemini_response('{"destination": "Astrology"}')):
            result = gc.classify_navigation_intent("what do the stars say", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] == "Astrology"

    def test_markdown_fenced_response_parses(self):
        fenced = '```json\n{"destination": "Tarot"}\n```'
        with patch("requests.post", return_value=_gemini_response(fenced)):
            result = gc.classify_navigation_intent("cards please", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] == "Tarot"

    def test_explicit_null_destination(self):
        with patch("requests.post", return_value=_gemini_response('{"destination": null}')):
            result = gc.classify_navigation_intent("asdkjfh nonsense", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] is None


class TestHallucinationSafety:
    """The core property: an invalid destination is never trusted, however
    plausible-looking the model's output is."""

    def test_destination_outside_vocabulary_is_rejected(self):
        with patch("requests.post", return_value=_gemini_response('{"destination": "SecretAdminPanel"}')):
            result = gc.classify_navigation_intent("something", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] is None

    def test_lowercase_variant_of_real_destination_is_rejected(self):
        # Case must match exactly — "astrology" is not "Astrology" and
        # shouldn't be silently coerced, since app.py sets st.session_state.nav
        # to the exact string.
        with patch("requests.post", return_value=_gemini_response('{"destination": "astrology"}')):
            result = gc.classify_navigation_intent("something", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] is None

    def test_every_accepted_destination_is_in_the_closed_vocabulary(self):
        for name in nr.DESTINATIONS:
            with patch("requests.post", return_value=_gemini_response(f'{{"destination": "{name}"}}')):
                result = gc.classify_navigation_intent("test", nr.DESTINATION_DESCRIPTIONS)
            assert result["destination"] == name


class TestMalformedResponses:
    def test_non_json_response_returns_none_not_crash(self):
        with patch("requests.post", return_value=_gemini_response("I'm not sure what you mean.")):
            result = gc.classify_navigation_intent("test", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] is None

    def test_malformed_json_returns_none_not_crash(self):
        with patch("requests.post", return_value=_gemini_response("{not valid json,,,}")):
            result = gc.classify_navigation_intent("test", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] is None

    def test_missing_destination_key_returns_none(self):
        with patch("requests.post", return_value=_gemini_response('{"foo": "bar"}')):
            result = gc.classify_navigation_intent("test", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] is None


class TestApiKeyHandling:
    def test_missing_api_key_returns_none_without_calling_api(self):
        with patch.dict(os.environ, {}, clear=True), patch("requests.post") as mock_post:
            result = gc.classify_navigation_intent("test", nr.DESTINATION_DESCRIPTIONS)
        assert result["destination"] is None
        mock_post.assert_not_called()


class TestPromptConstruction:
    def test_all_destinations_included_in_prompt(self):
        with patch("requests.post", return_value=_gemini_response('{"destination": "Home"}')) as mock_post:
            gc.classify_navigation_intent("test query", nr.DESTINATION_DESCRIPTIONS)
        sent = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        for name in nr.DESTINATIONS:
            assert name in sent

    def test_user_query_included_in_prompt(self):
        with patch("requests.post", return_value=_gemini_response('{"destination": "Home"}')) as mock_post:
            gc.classify_navigation_intent("a very specific unusual query", nr.DESTINATION_DESCRIPTIONS)
        sent = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        assert "a very specific unusual query" in sent
