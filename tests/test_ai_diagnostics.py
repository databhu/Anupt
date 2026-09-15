"""
Tests for ai/gemini_client.py's get_last_call_diagnostics() — the
instrumentation Developer Mode reads from. Never shown to a regular user;
these tests confirm it accurately reflects what actually happened in the
most recent _call(), across success, failover, and every failure path.
"""

import logging
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


_OK_BODY = {"candidates": [{"content": {"parts": [{"text": "A reading."}]}}]}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    key_manager._key_state.clear()
    gc._resolved_model = "gemini-3.6-flash"
    yield
    key_manager._key_state.clear()


class TestDiagnosticsSuccess:
    def test_success_case_recorded_correctly(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", return_value=_response(200, _OK_BODY)):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "success"
        assert diag["fallback_attempts"] == 0
        assert diag["last_error"] is None
        assert diag["model"] == "gemini-3.6-flash"

    def test_latency_is_tracked_and_nonnegative(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", return_value=_response(200, _OK_BODY)):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert isinstance(diag["latency_ms"], int)
        assert diag["latency_ms"] >= 0

    def test_request_id_changes_between_calls(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", return_value=_response(200, _OK_BODY)):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
            first_id = gc.get_last_call_diagnostics()["request_id"]
            gc._call([{"role": "user", "parts": [{"text": "hi again"}]}])
            second_id = gc.get_last_call_diagnostics()["request_id"]
        assert first_id != second_id


class TestDiagnosticsFailover:
    def test_fallback_attempts_counted_correctly(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "primarykey12345")
        monkeypatch.setenv("GEMINI_API_KEY_2", "backupkey123456")

        def fake_post(url, headers=None, **kw):
            return _response(429, {}) if headers.get("x-goog-api-key") == "primarykey12345" else _response(200, _OK_BODY)

        with patch("requests.post", side_effect=fake_post):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "success"
        assert diag["fallback_attempts"] == 1


class TestDiagnosticsFailureModes:
    def test_no_key_configured(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        for i in range(2, 11):
            monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)
        with patch("requests.post") as mock_post:
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "no_key_configured"
        mock_post.assert_not_called()

    def test_rate_limited_all_keys(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", return_value=_response(429, {})):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "rate_limited"
        assert "429" in diag["last_error"]

    def test_invalid_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "badkey1234567")
        with patch("requests.post", return_value=_response(401, {})):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "invalid_key"

    def test_http_error(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", return_value=_response(500, {})):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "http_error"
        assert "500" in diag["last_error"]

    def test_http_error_diagnostics_include_the_response_body_not_just_status(self, monkeypatch):
        # Developer Mode needs the actual reason Google gave, not just the
        # bare status code -- a bare "HTTP 500" doesn't distinguish a
        # transient server hiccup from something the app owner needs to
        # actually fix, and the server log isn't always reachable for
        # every hosting setup.
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        error_body = {"error": {"message": "The model is overloaded. Please try again later."}}
        with patch("requests.post", return_value=_response(503, error_body)):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert "overloaded" in diag["last_error"]

    def test_no_candidates(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", return_value=_response(200, {"candidates": []})):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "no_candidates"

    def test_empty_text(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        empty = {"candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "SAFETY"}]}
        with patch("requests.post", return_value=_response(200, empty)):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "empty_text"
        assert "SAFETY" in diag["last_error"]

    def test_network_error(self, monkeypatch):
        import requests
        monkeypatch.setenv("GEMINI_API_KEY", "onlykey1234567")
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("timeout")):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert diag["outcome"] == "network_error"


class TestDiagnosticsSecurity:
    def test_raw_key_never_appears_in_diagnostics(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "supersecretkeyvalue123456")
        with patch("requests.post", return_value=_response(401, {})):
            gc._call([{"role": "user", "parts": [{"text": "hi"}]}])
        diag = gc.get_last_call_diagnostics()
        assert "supersecretkeyvalue123456" not in str(diag)
