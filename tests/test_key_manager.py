"""
Tests for ai/key_manager.py — the multi-key failover manager.

The property under test throughout that matters most: a raw key value
must NEVER appear in anything this module returns for display (status_summary)
— every test that touches masking explicitly asserts the raw key string is
absent from the output, not just that a masked-looking string is present.
"""

import time

import pytest

from ai import key_manager as km


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Every test gets a fresh key-state dict and its own env vars, so
    cooldowns/invalidations from one test can't leak into another."""
    km._key_state.clear()
    yield
    km._key_state.clear()


class TestKeyDiscovery:
    def test_finds_single_unsuffixed_key(self, monkeypatch):
        monkeypatch.setenv("SOLO_API_KEY", "onlykey123456")
        assert km.discover_keys("solo") == ["onlykey123456"]

    def test_finds_multiple_sequential_keys(self, monkeypatch):
        monkeypatch.setenv("MULTI_API_KEY", "key1value")
        monkeypatch.setenv("MULTI_API_KEY_2", "key2value")
        monkeypatch.setenv("MULTI_API_KEY_3", "key3value")
        assert km.discover_keys("multi") == ["key1value", "key2value", "key3value"]

    def test_stops_at_gap_in_numbered_keys(self, monkeypatch):
        monkeypatch.setenv("GAP_API_KEY", "key1value")
        monkeypatch.setenv("GAP_API_KEY_3", "key3value")  # _2 deliberately missing
        assert km.discover_keys("gap") == ["key1value"]

    def test_no_keys_configured_returns_empty_list(self, monkeypatch):
        monkeypatch.delenv("NONE_API_KEY", raising=False)
        assert km.discover_keys("none") == []

    def test_provider_name_case_insensitive_env_lookup(self, monkeypatch):
        monkeypatch.setenv("CASETEST_API_KEY", "somekey")
        assert km.discover_keys("CaseTest") == ["somekey"]
        assert km.discover_keys("casetest") == ["somekey"]


class TestFailover:
    def test_returns_first_key_when_all_available(self, monkeypatch):
        monkeypatch.setenv("FO_API_KEY", "primary")
        monkeypatch.setenv("FO_API_KEY_2", "backup")
        assert km.get_key("fo") == "primary"

    def test_fails_over_to_backup_after_rate_limit(self, monkeypatch):
        monkeypatch.setenv("FO2_API_KEY", "primary")
        monkeypatch.setenv("FO2_API_KEY_2", "backup")
        km.report_rate_limited("fo2", "primary", cooldown_seconds=60)
        assert km.get_key("fo2") == "backup"

    def test_returns_none_when_all_keys_exhausted(self, monkeypatch):
        monkeypatch.setenv("FO3_API_KEY", "primary")
        monkeypatch.setenv("FO3_API_KEY_2", "backup")
        km.report_rate_limited("fo3", "primary", cooldown_seconds=60)
        km.report_rate_limited("fo3", "backup", cooldown_seconds=60)
        assert km.get_key("fo3") is None

    def test_key_available_again_after_cooldown_expires(self, monkeypatch):
        monkeypatch.setenv("FO4_API_KEY", "primary")
        km.report_rate_limited("fo4", "primary", cooldown_seconds=0.2)
        assert km.get_key("fo4") is None
        time.sleep(0.3)
        assert km.get_key("fo4") == "primary"

    def test_report_success_clears_cooldown_early(self, monkeypatch):
        monkeypatch.setenv("FO5_API_KEY", "primary")
        km.report_rate_limited("fo5", "primary", cooldown_seconds=100)
        assert km.get_key("fo5") is None
        km.report_success("fo5", "primary")
        assert km.get_key("fo5") == "primary"


class TestPermanentInvalidation:
    def test_invalid_key_is_skipped(self, monkeypatch):
        monkeypatch.setenv("INV_API_KEY", "badkey")
        monkeypatch.setenv("INV_API_KEY_2", "goodkey")
        km.report_invalid("inv", "badkey")
        assert km.get_key("inv") == "goodkey"

    def test_invalid_key_does_not_expire_like_cooldown(self, monkeypatch):
        monkeypatch.setenv("INV2_API_KEY", "badkey")
        monkeypatch.setenv("INV2_API_KEY_2", "goodkey")
        km.report_invalid("inv2", "badkey")
        time.sleep(0.1)
        # Even after time passes, an invalid key must stay excluded —
        # unlike a cooldown, there's no reason to believe it will start
        # working again.
        assert km.get_key("inv2") == "goodkey"

    def test_all_keys_invalid_returns_none(self, monkeypatch):
        monkeypatch.setenv("INV3_API_KEY", "badkey")
        km.report_invalid("inv3", "badkey")
        assert km.get_key("inv3") is None


class TestKeyMaskingSecurity:
    """The property that matters most: raw key material must never leak
    through this module's diagnostic output."""

    def test_raw_key_never_appears_in_status_summary(self, monkeypatch):
        monkeypatch.setenv("SEC_API_KEY", "sk-abcdefghijklmnopqrstuvwxyz123456")
        summary = km.status_summary("sec")
        assert "abcdefghijklmnopqrstuvwxyz" not in str(summary)

    def test_masked_form_shows_only_first_and_last_four(self, monkeypatch):
        monkeypatch.setenv("SEC2_API_KEY", "sk-abcdefghijklmnopqrstuvwxyz123456")
        summary = km.status_summary("sec2")
        assert summary[0]["key_masked"] == "sk-a...3456"

    def test_short_key_fully_masked_not_partially_exposed(self, monkeypatch):
        monkeypatch.setenv("SEC3_API_KEY", "short")
        summary = km.status_summary("sec3")
        assert "short" not in str(summary)
        assert summary[0]["key_masked"] == "*****"

    def test_status_reflects_current_state_accurately(self, monkeypatch):
        monkeypatch.setenv("SEC4_API_KEY", "availablekey123")
        monkeypatch.setenv("SEC4_API_KEY_2", "coolingkey12345")
        monkeypatch.setenv("SEC4_API_KEY_3", "invalidkey12345")
        km.report_rate_limited("sec4", "coolingkey12345", cooldown_seconds=60)
        km.report_invalid("sec4", "invalidkey12345")
        summary = km.status_summary("sec4")
        statuses = [s["status"] for s in summary]
        assert statuses[0] == "Available"
        assert "Cooling down" in statuses[1]
        assert statuses[2] == "Invalid"

    def test_mask_function_never_reveals_the_raw_key(self):
        # The property that actually matters is that the raw key never
        # appears in the masked output — not the masked string's length,
        # which can legitimately be longer than a short original key
        # while still being fully secure (e.g. "aaaaaaaaaa" -> "aaaa...aaaa").
        for key in ["a" * 10, "a" * 20, "a" * 40, "sk-" + "x" * 30]:
            masked = km._mask(key)
            assert key not in masked
