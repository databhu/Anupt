"""
Tests for ai/gemini_client.py's model discovery/resolution/fallback logic
(_discover_models, _resolve_model, _mark_model_bad, _post_with_fallback).

This is the machinery behind several explicit acceptance criteria: "No
recurring 503/404 model error caused by using an inappropriate/unavailable
model", "App automatically selects a supported lightweight model",
"Fallback model logic works" — none of which had any test coverage before
this file, despite being core infrastructure with a real documented
incident driving its design (see _known_bad_models' docstring in
gemini_client.py: gemini-2.5-flash kept being listed as available long
after Google started hard-rejecting it for new keys).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

logging.disable(logging.CRITICAL)

from ai import gemini_client as gc


def _list_models_response(model_names_and_methods: list):
    """model_names_and_methods: list of (name, methods_list) tuples."""
    r = MagicMock()
    r.status_code = 200
    r.json = lambda: {
        "models": [
            {"name": f"models/{name}", "supportedGenerationMethods": methods}
            for name, methods in model_names_and_methods
        ]
    }
    r.text = ""
    return r


def _generate_response(status: int, text: str = ""):
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json = lambda: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    return r


@pytest.fixture(autouse=True)
def _clean_model_state():
    gc._resolved_model = None
    gc._known_bad_models.clear()
    yield
    gc._resolved_model = None
    gc._known_bad_models.clear()


def _blocklist_alias():
    """Most tests in this file exercise the DEFAULT_MODEL/discovery
    fallback chain specifically, which only kicks in once the stable
    alias (tried first — see STABLE_ALIAS's docstring) has already been
    ruled out. Pre-blocklisting it here lets those tests focus on the
    layer they're actually testing without every one of them needing to
    separately account for the alias attempt in front of it."""
    gc._known_bad_models.add(gc.STABLE_ALIAS)


class TestStableAliasTriedFirst:
    """STABLE_ALIAS is Google's own maintained alias, tried before
    DEFAULT_MODEL/discovery specifically because it's designed to survive
    version-specific deprecations without any code change on this app's
    part — these tests confirm it's actually used that way, not just
    defined and forgotten."""

    def test_resolves_to_the_alias_on_a_fresh_process(self):
        with patch("requests.get") as mock_get:
            resolved = gc._resolve_model("fake-key")
        assert resolved == gc.STABLE_ALIAS
        mock_get.assert_not_called()  # no discovery needed when the alias hasn't failed yet

    def test_generate_call_targets_the_alias_in_its_url(self):
        with patch("requests.post", return_value=_generate_response(200)) as mock_post:
            gc._post_with_fallback("fake-key", {"contents": []})
        called_url = mock_post.call_args.args[0]
        assert gc.STABLE_ALIAS in called_url

    def test_alias_404_falls_through_to_default_model(self):
        response = _list_models_response([(gc.DEFAULT_MODEL, ["generateContent"])])
        call_log = []

        def fake_post(url, headers=None, json=None, timeout=None):
            call_log.append(url)
            if gc.STABLE_ALIAS in url:
                return _generate_response(404, "alias temporarily unavailable")
            return _generate_response(200)

        with patch("requests.get", return_value=response), patch("requests.post", side_effect=fake_post):
            resp = gc._post_with_fallback("fake-key", {"contents": []})

        assert resp.status_code == 200
        assert len(call_log) == 2
        assert gc.STABLE_ALIAS in call_log[0]
        assert gc.DEFAULT_MODEL in call_log[1]

    def test_explicit_override_still_beats_the_alias(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "operator-pinned-model")
        with patch("requests.get") as mock_get:
            resolved = gc._resolve_model("fake-key")
        assert resolved == "operator-pinned-model"
        mock_get.assert_not_called()


class TestModelDiscoveryFiltering:
    def test_only_generatecontent_capable_models_included(self):
        response = _list_models_response([
            ("gemini-3.6-flash", ["generateContent"]),
            ("text-embedding-004", ["embedContent"]),  # not generateContent
        ])
        with patch("requests.get", return_value=response):
            models = gc._discover_models("fake-key")
        assert "gemini-3.6-flash" in models
        assert "text-embedding-004" not in models

    def test_specialist_variants_excluded(self):
        response = _list_models_response([
            ("gemini-3.6-flash", ["generateContent"]),
            ("gemini-embedding-001", ["generateContent"]),
            ("aqa", ["generateContent"]),
            ("imagen-3", ["generateContent"]),
            ("veo-2", ["generateContent"]),
        ])
        with patch("requests.get", return_value=response):
            models = gc._discover_models("fake-key")
        assert models == ["gemini-3.6-flash"]

    def test_models_prefix_stripped(self):
        response = _list_models_response([("gemini-3.6-flash", ["generateContent"])])
        with patch("requests.get", return_value=response):
            models = gc._discover_models("fake-key")
        assert models[0] == "gemini-3.6-flash"
        assert "models/" not in models[0]

    def test_http_error_returns_empty_list_not_crash(self):
        bad_response = MagicMock()
        bad_response.status_code = 500
        bad_response.text = "server error"
        with patch("requests.get", return_value=bad_response):
            models = gc._discover_models("fake-key")
        assert models == []

    def test_network_error_returns_empty_list_not_crash(self):
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("timeout")):
            models = gc._discover_models("fake-key")
        assert models == []


class TestModelResolution:
    def test_prefers_default_model_when_available(self):
        _blocklist_alias()
        response = _list_models_response([
            (gc.DEFAULT_MODEL, ["generateContent"]),
            ("gemini-other-model", ["generateContent"]),
        ])
        with patch("requests.get", return_value=response):
            resolved = gc._resolve_model("fake-key")
        assert resolved == gc.DEFAULT_MODEL

    def test_falls_back_to_best_available_when_default_missing(self):
        _blocklist_alias()
        response = _list_models_response([("gemini-alternative-flash", ["generateContent"])])
        with patch("requests.get", return_value=response):
            resolved = gc._resolve_model("fake-key")
        assert resolved == "gemini-alternative-flash"

    def test_falls_back_to_compiled_default_when_discovery_fails_entirely(self):
        _blocklist_alias()
        bad_response = MagicMock()
        bad_response.status_code = 500
        bad_response.text = ""
        with patch("requests.get", return_value=bad_response):
            resolved = gc._resolve_model("fake-key")
        # A transient ListModels blip must not take the feature down —
        # falls back to the compiled-in default rather than erroring.
        assert resolved == gc.DEFAULT_MODEL

    def test_result_is_cached_not_rediscovered_every_call(self):
        _blocklist_alias()
        response = _list_models_response([(gc.DEFAULT_MODEL, ["generateContent"])])
        with patch("requests.get", return_value=response) as mock_get:
            gc._resolve_model("fake-key")
            gc._resolve_model("fake-key")
            gc._resolve_model("fake-key")
        assert mock_get.call_count == 1

    def test_force_refresh_bypasses_cache(self):
        _blocklist_alias()
        response = _list_models_response([(gc.DEFAULT_MODEL, ["generateContent"])])
        with patch("requests.get", return_value=response) as mock_get:
            gc._resolve_model("fake-key")
            gc._resolve_model("fake-key", force_refresh=True)
        assert mock_get.call_count == 2

    def test_explicit_override_always_wins(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "operator-pinned-model")
        with patch("requests.get") as mock_get:
            resolved = gc._resolve_model("fake-key")
        assert resolved == "operator-pinned-model"
        mock_get.assert_not_called()  # no discovery needed at all when pinned


class TestMarkModelBad:
    def test_bad_model_excluded_from_future_resolution(self):
        _blocklist_alias()
        gc._mark_model_bad(gc.DEFAULT_MODEL)
        response = _list_models_response([
            (gc.DEFAULT_MODEL, ["generateContent"]),
            ("gemini-backup-model", ["generateContent"]),
        ])
        with patch("requests.get", return_value=response):
            resolved = gc._resolve_model("fake-key")
        assert resolved == "gemini-backup-model"

    def test_marking_currently_resolved_model_bad_clears_cache(self):
        gc._resolved_model = "some-model"
        gc._mark_model_bad("some-model")
        assert gc._resolved_model is None

    def test_marking_a_different_model_bad_does_not_clear_cache(self):
        gc._resolved_model = "some-model"
        gc._mark_model_bad("a-different-model")
        assert gc._resolved_model == "some-model"


class TestPostWithFallback404Recovery:
    """The core regression this guards: a model that's listed as available
    but actually 404s on generateContent (the real gemini-2.5-flash
    incident) must not silently break every reading — the fallback must
    try a different model instead."""

    def test_404_on_first_model_retries_with_second(self):
        _blocklist_alias()
        response = _list_models_response([
            (gc.DEFAULT_MODEL, ["generateContent"]),
            ("gemini-backup-model", ["generateContent"]),
        ])
        call_log = []

        def fake_post(url, headers=None, json=None, timeout=None):
            call_log.append(url)
            if gc.DEFAULT_MODEL in url:
                return _generate_response(404, "model retired")
            return _generate_response(200)

        with patch("requests.get", return_value=response), patch("requests.post", side_effect=fake_post):
            resp = gc._post_with_fallback("fake-key", {"contents": []})

        assert resp.status_code == 200
        assert len(call_log) == 2
        assert gc.DEFAULT_MODEL in call_log[0]
        assert "gemini-backup-model" in call_log[1]

    def test_404_model_gets_blocklisted(self):
        _blocklist_alias()
        response = _list_models_response([
            (gc.DEFAULT_MODEL, ["generateContent"]),
            ("gemini-backup-model", ["generateContent"]),
        ])

        def fake_post(url, headers=None, json=None, timeout=None):
            return _generate_response(404) if gc.DEFAULT_MODEL in url else _generate_response(200)

        with patch("requests.get", return_value=response), patch("requests.post", side_effect=fake_post):
            gc._post_with_fallback("fake-key", {"contents": []})

        assert gc.DEFAULT_MODEL in gc._known_bad_models

    def test_does_not_loop_forever_when_no_alternative_exists(self):
        _blocklist_alias()
        # Only one model exists at all, and it 404s -- must stop after a
        # bounded number of attempts, not spin indefinitely.
        response = _list_models_response([(gc.DEFAULT_MODEL, ["generateContent"])])
        call_count = [0]

        def fake_post(url, headers=None, json=None, timeout=None):
            call_count[0] += 1
            return _generate_response(404, "gone")

        with patch("requests.get", return_value=response), patch("requests.post", side_effect=fake_post):
            resp = gc._post_with_fallback("fake-key", {"contents": []})

        assert resp.status_code == 404  # surfaces the real failure rather than hanging
        assert call_count[0] <= 3  # bounded, never infinite

    def test_non_404_error_returned_immediately_without_retry(self):
        _blocklist_alias()
        response = _list_models_response([(gc.DEFAULT_MODEL, ["generateContent"])])
        call_count = [0]

        def fake_post(url, headers=None, json=None, timeout=None):
            call_count[0] += 1
            return _generate_response(429)  # rate limit, not a dead-model issue

        with patch("requests.get", return_value=response), patch("requests.post", side_effect=fake_post):
            resp = gc._post_with_fallback("fake-key", {"contents": []})

        assert resp.status_code == 429
        assert call_count[0] == 1  # no model-fallback retry for a non-404
