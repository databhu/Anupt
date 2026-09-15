"""
Tests for ai/gemini_client.py's chat_reply() — carries the same
astrologer-persona/explain-why/one-tip instructions as the other reading
prompts (test_unified_reading_prompt.py, test_engine_reading_prompt.py),
for consistency across every place the app produces reading-related text.
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


def _sent_prompt(mock_post) -> str:
    return mock_post.call_args.kwargs["json"]["contents"][-1]["parts"][0]["text"]


class TestChatReplyPersona:
    def test_astrologer_persona_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("A reply.")) as mock_post:
            gc.chat_reply([], "What about my career?", {"theme": "career"})
        sent = _sent_prompt(mock_post)
        assert "experienced astrologer" in sent.lower()

    def test_why_and_tip_instruction_present(self):
        with patch("requests.post", return_value=_gemini_response("A reply.")) as mock_post:
            gc.chat_reply([], "What about my career?", {"theme": "career"})
        sent = _sent_prompt(mock_post)
        assert "WHY" in sent
        assert "concrete thing" in sent

    def test_evidence_reaches_the_prompt(self):
        with patch("requests.post", return_value=_gemini_response("A reply.")) as mock_post:
            gc.chat_reply([], "What about my career?", {"unique_marker_xyz": "career"})
        sent = _sent_prompt(mock_post)
        assert "unique_marker_xyz" in sent

    def test_question_reaches_the_prompt(self):
        with patch("requests.post", return_value=_gemini_response("A reply.")) as mock_post:
            gc.chat_reply([], "Will I get promoted this year?", {})
        sent = _sent_prompt(mock_post)
        assert "Will I get promoted this year?" in sent

    def test_history_included_as_prior_turns(self):
        history = [{"role": "user", "text": "Hi"}, {"role": "model", "text": "Hello!"}]
        with patch("requests.post", return_value=_gemini_response("A reply.")) as mock_post:
            gc.chat_reply(history, "What about love?", {})
        sent_contents = mock_post.call_args.kwargs["json"]["contents"]
        assert len(sent_contents) == 3  # 2 history turns + the new question
        assert sent_contents[0]["parts"][0]["text"] == "Hi"

    def test_returns_the_reply_text(self):
        with patch("requests.post", return_value=_gemini_response("Career looks strong this year.")):
            reply = gc.chat_reply([], "What about my career?", {})
        assert reply == "Career looks strong this year."
