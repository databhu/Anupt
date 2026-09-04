"""
Tests for engines/youtube_client.py — the deterministic YouTube research
stage. Every test uses a mocked requests.get, since this sandbox cannot
reach googleapis.com (confirmed host_not_allowed at the network egress
layer — the same restriction already true of the Gemini API throughout
this project). What's under test is that the code is CORRECTLY structured
to talk to the real API when deployed, and that it degrades safely for
every failure mode a real deployment could hit.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

logging.disable(logging.CRITICAL)

from ai import key_manager
from engines import youtube_client as yc


def _response(status: int, body: dict):
    r = MagicMock()
    r.status_code = status
    r.json = lambda: body
    r.text = str(body)
    return r


_GOOD_ITEM = {
    "id": {"videoId": "abc123"},
    "snippet": {
        "publishedAt": "2026-09-01T10:00:00Z", "title": "Aries Weekly Horoscope",
        "description": "This week brings career opportunities for Aries.",
        "channelTitle": "AstroChannel", "thumbnails": {"default": {"url": "http://thumb.jpg"}},
    },
}


@pytest.fixture(autouse=True)
def _clean_state():
    key_manager._key_state.clear()
    yield
    key_manager._key_state.clear()


class TestBasicSearch:
    def test_well_formed_response_parsed_correctly(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {"items": [_GOOD_ITEM]})):
            result = yc.search_astrology_videos("Aries", "week")
        assert result["error"] is None
        assert len(result["videos"]) == 1
        assert result["videos"][0]["video_id"] == "abc123"
        assert result["videos"][0]["url"] == "https://www.youtube.com/watch?v=abc123"
        assert result["videos"][0]["title"] == "Aries Weekly Horoscope"

    def test_query_includes_the_zodiac_sign(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {"items": []})) as mock_get:
            yc.search_astrology_videos("Scorpio", "week")
        sent_query = mock_get.call_args.kwargs["params"]["q"]
        assert "Scorpio" in sent_query

    def test_query_targets_vedic_astrology_not_generic_astrology(self, monkeypatch):
        # A deliberate signal choice: Vedic astrology is predominantly an
        # Indian tradition and matches what the rest of this app does
        # (sidereal, Nakshatra-based) — a stronger signal than country
        # targeting alone for the KIND of creator this app wants.
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {"items": []})) as mock_get:
            yc.search_astrology_videos("Scorpio", "week")
        sent_query = mock_get.call_args.kwargs["params"]["q"]
        assert "vedic astrology" in sent_query.lower()

    def test_empty_results_handled_cleanly(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {"items": []})):
            result = yc.search_astrology_videos("Aries", "week")
        assert result["videos"] == []
        assert result["error"] is None


class TestRegionCode:
    def test_defaults_to_india(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {"items": []})) as mock_get:
            yc.search_astrology_videos("Aries", "week")
        assert mock_get.call_args.kwargs["params"]["regionCode"] == "IN"

    def test_region_code_is_configurable(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {"items": []})) as mock_get:
            yc.search_astrology_videos("Aries", "week", region_code="US")
        assert mock_get.call_args.kwargs["params"]["regionCode"] == "US"

    def test_none_omits_region_code_entirely(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {"items": []})) as mock_get:
            yc.search_astrology_videos("Aries", "week", region_code=None)
        assert "regionCode" not in mock_get.call_args.kwargs["params"]

    def test_region_code_preserved_through_key_failover_retry(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytprimary123456")
        monkeypatch.setenv("YOUTUBE_API_KEY_2", "ytbackup1234567")
        calls = []

        def fake_get(url, params=None, **kw):
            calls.append(params)
            if params["key"] == "ytprimary123456":
                return _response(403, {"error": "quotaExceeded"})
            return _response(200, {"items": []})

        with patch("requests.get", side_effect=fake_get):
            yc.search_astrology_videos("Leo", "week", region_code="IN")
        assert all(c.get("regionCode") == "IN" for c in calls)


class TestKeyFailover:
    def test_quota_exceeded_fails_over_to_backup_key(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytprimary123456")
        monkeypatch.setenv("YOUTUBE_API_KEY_2", "ytbackup1234567")
        calls = []

        def fake_get(url, params=None, **kw):
            calls.append(params["key"])
            if params["key"] == "ytprimary123456":
                return _response(403, {"error": "quotaExceeded"})
            return _response(200, {"items": [_GOOD_ITEM]})

        with patch("requests.get", side_effect=fake_get):
            result = yc.search_astrology_videos("Leo", "week")
        assert calls == ["ytprimary123456", "ytbackup1234567"]
        assert result["error"] is None
        assert len(result["videos"]) == 1

    def test_all_keys_exhausted_returns_friendly_error(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytonlykey123456")
        with patch("requests.get", return_value=_response(403, {"error": "quotaExceeded"})):
            result = yc.search_astrology_videos("Leo", "week")
        assert result["videos"] == []
        assert result["error"] is not None

    def test_missing_key_returns_error_without_calling_api(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        with patch("requests.get") as mock_get:
            result = yc.search_astrology_videos("Leo", "week")
        assert result["videos"] == []
        mock_get.assert_not_called()


class TestErrorHandling:
    def test_network_error_handled_gracefully(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("timeout")):
            result = yc.search_astrology_videos("Leo", "week")
        assert result["videos"] == []
        assert result["error"] is not None

    def test_non_json_response_handled_gracefully(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.json = MagicMock(side_effect=ValueError("not json"))
        with patch("requests.get", return_value=bad_response):
            result = yc.search_astrology_videos("Leo", "week")
        assert result["videos"] == []
        assert result["error"] is not None

    def test_other_http_error_handled_gracefully(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(500, {"error": "server error"})):
            result = yc.search_astrology_videos("Leo", "week")
        assert result["videos"] == []
        assert result["error"] is not None


class TestMalformedDataResilience:
    def test_item_missing_video_id_is_skipped(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        malformed = {"items": [
            {"id": {}, "snippet": {"title": "no video id"}},
            _GOOD_ITEM,
        ]}
        with patch("requests.get", return_value=_response(200, malformed)):
            result = yc.search_astrology_videos("Leo", "week")
        assert len(result["videos"]) == 1
        assert result["videos"][0]["video_id"] == "abc123"

    def test_item_missing_title_is_skipped(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        malformed = {"items": [
            {"id": {"videoId": "notitle"}, "snippet": {}},
            _GOOD_ITEM,
        ]}
        with patch("requests.get", return_value=_response(200, malformed)):
            result = yc.search_astrology_videos("Leo", "week")
        assert len(result["videos"]) == 1

    def test_missing_items_key_entirely_does_not_crash(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "ytkey1234567890")
        with patch("requests.get", return_value=_response(200, {})):
            result = yc.search_astrology_videos("Leo", "week")
        assert result["videos"] == []
        assert result["error"] is None


class TestKeyStatusSecurity:
    def test_raw_key_never_appears_in_status(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "supersecretytkey123456")
        status = yc.youtube_key_status()
        assert "supersecretytkey123456" not in str(status)
