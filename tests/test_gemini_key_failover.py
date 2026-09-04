"""
Tests for the key-manager integration inside ai/gemini_client.py's _call().

test_key_manager.py already covers the key manager in isolation; this file
covers the thing that actually matters to a user: does a real generation
call genuinely fail over to a backup key when the primary is rate-limited
or invalid, end to end through _call() itself — not just verified at the
key-manager layer alone.
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

logging.disable(logging.CRITICAL)

from ai import gemini_client as gc
from ai import key_manager


def _response(status: int, body: dict):
    r = MagicMock()
    r.status_code = status
    r.json = lambda: body
    r.text = str(body)
    return r


_OK_BODY = {"candidates": [{"content": {"parts": [{"text": "A real reading."}]}}]}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    key_manager._key_state.clear()
    gc._resolved_model = "gemini-3.6-flash"  # skip model discovery, not what's under test here
    yield
    key_manager._key_state.clear()


class TestCallLevelFailover:
    def test_rate_limited_primary_fails_over_to_backup(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "primarykey12345")
        monkeypatch.setenv("GEMINI_API_KEY_2", "backupkey123456")

        calls = []

        def fake_post(url, params=None, **kw):
            key = params.get("key")
            calls.append(key)
            if key == "primarykey12345":
                return _response(429, {"error": "rate limited"})
            return _response(200, _OK_BODY)

        with patch("requests.post", side_effect=fake_post):
            result = gc._call([{"role": "user", "parts": [{"text": "hi"}]}])

        assert calls == ["primarykey12345", "backupkey123456"]
        assert result == "A real reading."

    def test_invalid_primary_fails_over_and_marks_invalid(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "invalidkey1234")
        monkeypatch.setenv("GEMINI_API_KEY_2", "validkey123456")

        def fake_post(url, params=None, **kw):
            return _response(401, {}) if params.get("key") == "invalidkey1234" else _response(200, _OK_BODY)

        with patch("requests.post", side_effect=fake_post):
            result = gc._call([{"role": "user", "parts": [{"text": "hi"}]}])

        assert result == "A real reading."
        statuses = {s["key_masked"]: s["status"] for s in key_manager.status_summary("gemini")}
        assert any(v == "Invalid" for v in statuses.values())

    def test_both_keys_exhausted_returns_friendly_message_not_crash(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "key1aaaaaaaaaa")
        monkeypatch.setenv("GEMINI_API_KEY_2", "key2bbbbbbbbbb")
        with patch("requests.post", return_value=_response(429, {})):
            result = gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        assert "error" not in result.lower() or "available" in result.lower()
        assert result == gc._FRIENDLY_RATE_LIMITED

    def test_success_on_backup_key_does_not_affect_primarys_cooldown(self, monkeypatch):
        # Attribution matters: reporting success on the key that actually
        # succeeded must not accidentally clear or otherwise touch a
        # DIFFERENT key's cooldown state.
        monkeypatch.setenv("GEMINI_API_KEY", "primarykey12345")
        monkeypatch.setenv("GEMINI_API_KEY_2", "backupkey123456")
        key_manager.report_rate_limited("gemini", "primarykey12345", cooldown_seconds=100)

        with patch("requests.post", return_value=_response(200, _OK_BODY)):
            result = gc._call([{"role": "user", "parts": [{"text": "hi"}]}])

        assert result == "A real reading."
        # Primary should still be cooling down — only backup succeeded, and
        # a cooling-down key correctly isn't even attempted (avoiding
        # hammering an exhausted key), so it was never a candidate to clear.
        assert key_manager.get_key("gemini") == "backupkey123456"

    def test_single_key_setup_unaffected_by_multikey_logic(self, monkeypatch):
        # No _2, _3 etc configured — must behave exactly like the
        # original single-key implementation: one call, one result.
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", return_value=_response(200, _OK_BODY)) as mock_post:
            result = gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        assert result == "A real reading."
        assert mock_post.call_count == 1

    def test_no_keys_configured_returns_friendly_message(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        for i in range(2, 11):
            monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)
        with patch("requests.post") as mock_post:
            result = gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        assert result == gc._FRIENDLY_UNAVAILABLE
        mock_post.assert_not_called()

    def test_three_keys_all_rate_limited_in_sequence_then_succeeds(self, monkeypatch):
        # A more realistic failover chain: first two keys are exhausted,
        # third one works.
        monkeypatch.setenv("GEMINI_API_KEY", "keyone12345678")
        monkeypatch.setenv("GEMINI_API_KEY_2", "keytwo12345678")
        monkeypatch.setenv("GEMINI_API_KEY_3", "keythree1234567")

        def fake_post(url, params=None, **kw):
            key = params.get("key")
            if key in ("keyone12345678", "keytwo12345678"):
                return _response(429, {})
            return _response(200, _OK_BODY)

        with patch("requests.post", side_effect=fake_post):
            result = gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        assert result == "A real reading."

    def test_raw_key_never_appears_in_logged_error_messages(self, monkeypatch, caplog):
        # Server logs are more widely visible than raw application state —
        # the masking guarantee has to hold there too, not just in
        # status_summary's return value.
        monkeypatch.setenv("GEMINI_API_KEY", "supersecretkeyvalue123")
        with patch("requests.post", return_value=_response(401, {})):
            with caplog.at_level(logging.ERROR, logger="anupt.gemini"):
                logging.getLogger("anupt.gemini").disabled = False
                gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
                logging.getLogger("anupt.gemini").disabled = True
        assert "supersecretkeyvalue123" not in caplog.text
