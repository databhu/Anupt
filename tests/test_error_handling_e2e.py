"""
End-to-end regression test: when the AI service fails for any reason, the
actual reading page must show a friendly message and never leak the raw
technical error (model names, HTTP internals, tracebacks) into the UI —
verified through the real Astrology page flow via AppTest, not just at
the ai/gemini_client.py unit level, since what matters is what the user
actually sees rendered on the page.
"""

import logging
from datetime import date, time
from unittest.mock import MagicMock, patch

import pytest

logging.disable(logging.CRITICAL)

from streamlit.testing.v1 import AppTest

import auth.store as store

pytestmark = pytest.mark.skipif(
    __import__("os").environ.get("DATABASE_URL") is None,
    reason="requires a live Postgres connection (DATABASE_URL)",
)


def _response(status: int, body: dict):
    r = MagicMock()
    r.status_code = status
    r.json = lambda: body
    r.text = str(body)
    return r


def _make_test_user(username: str) -> int:
    store.init_db()
    store.create_user(username, "password123")
    uid = store.verify_user(username, "password123")
    store.save_profile(uid, {
        "name": "Error Test", "dob": date(1995, 8, 8), "birth_time": time(14, 30),
        "city": "Mumbai, Maharashtra, India", "latitude": 19.076, "longitude": 72.8777,
        "utc_offset": 5.5, "interests": ["Career"],
    })
    return uid


class TestAstrologyPageNeverLeaksTechnicalErrors:
    def test_server_error_shows_friendly_message_not_raw_error(self):
        uid = _make_test_user("errtest_servererr")
        technical_body = {"error": "internal server error, model gemini-xyz-123 crashed: traceback follows..."}

        with patch("requests.post", return_value=_response(500, technical_body)):
            at = AppTest.from_file("../app.py")
            at.run(timeout=30)
            at.session_state["user_id"] = uid
            at.session_state["username"] = "errtest_servererr"
            at.session_state["nav"] = "Astrology"
            at.run(timeout=30)
            for b in at.button:
                if "Get my Astrology summary" in (b.label or ""):
                    b.click()
                    break
            at.run(timeout=30)

        assert len(at.exception) == 0, "a None score or error text must not crash the page"
        rendered_text = " ".join(m.value for m in at.markdown)
        assert "gemini-xyz-123" not in rendered_text
        assert "traceback" not in rendered_text.lower()
        assert "internal server error" not in rendered_text.lower()
        assert "temporarily" in rendered_text.lower() or "unavailable" in rendered_text.lower() or \
               "reading service" in rendered_text.lower()

    def test_missing_api_key_shows_config_error_not_crash(self, monkeypatch):
        uid = _make_test_user("errtest_nokey")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        for i in range(2, 11):
            monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)

        at = AppTest.from_file("../app.py")
        at.run(timeout=30)
        at.session_state["user_id"] = uid
        at.session_state["username"] = "errtest_nokey"
        at.session_state["nav"] = "Astrology"
        at.run(timeout=30)
        for b in at.button:
            if "Get my Astrology summary" in (b.label or ""):
                b.click()
                break
        at.run(timeout=30)

        assert len(at.exception) == 0
        rendered_text = " ".join(m.value for m in at.markdown)
        assert "configuration" in rendered_text.lower()

    def test_rate_limited_shows_temporary_message_not_crash(self):
        uid = _make_test_user("errtest_ratelimit")

        with patch("requests.post", return_value=_response(429, {})):
            at = AppTest.from_file("../app.py")
            at.run(timeout=30)
            at.session_state["user_id"] = uid
            at.session_state["username"] = "errtest_ratelimit"
            at.session_state["nav"] = "Numerology"
            at.run(timeout=30)
            for b in at.button:
                if "Get my Numerology summary" in (b.label or ""):
                    b.click()
                    break
            at.run(timeout=30)

        assert len(at.exception) == 0
        rendered_text = " ".join(m.value for m in at.markdown)
        assert "busy" in rendered_text.lower() or "moment" in rendered_text.lower()
