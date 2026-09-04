"""
ANUPT — Multi-key API manager.

Built for RELIABILITY, not for circumventing any single key's legitimate
rate limits. Every key this module rotates through is a real, separately-
provisioned credential the app owner controls (e.g. multiple free-tier
Gemini keys from different Google accounts they own, or a backup YouTube
Data API key) — its job is to fail over to another already-legitimate
key when one is temporarily exhausted or briefly unreachable, and to
respect EACH key's own quota independently. It does not pool keys to
make one account's limit look larger than it is, and it does not retry
a single key faster or more aggressively to work around its rate limit —
a rate-limited key goes into a cooldown and simply isn't used again
until that cooldown expires or the provider's own quota window resets.

Security: a raw key value is never logged, displayed, or returned from
this module — every public-facing view is masked (first 4 / last 4
characters only), and even that masked form is meant for an app owner's
diagnostics panel, never for a regular user.
"""

import os
import time as _time

_COOLDOWN_SECONDS_DEFAULT = 60
# (provider, key) -> {"cooldown_until": float (monotonic seconds), "permanently_failed": bool}
_key_state: dict = {}


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def discover_keys(provider: str) -> list[str]:
    """Every configured key for `provider`, checking <PROVIDER>_API_KEY,
    then <PROVIDER>_API_KEY_2, _3, ... in order. Checks os.environ first,
    then st.secrets (matching the single-key lookup already used
    elsewhere in this app). Stops at the first gap in numbered keys —
    e.g. if _2 is missing but _3 is present, _3 is not picked up, since
    a gap like that is much more likely a misconfiguration worth
    surfacing than something to silently work around."""
    keys = []
    base = f"{provider.upper()}_API_KEY"
    for suffix in [""] + [f"_{i}" for i in range(2, 11)]:
        name = base + suffix
        value = os.environ.get(name)
        if not value:
            try:
                import streamlit as st
                value = st.secrets.get(name)
            except Exception:
                value = None
        if value:
            keys.append(value)
        elif suffix != "":
            break
    return keys


def get_key(provider: str) -> str | None:
    """The next usable key for `provider` — the first configured one that
    isn't currently cooling down or permanently failed. None if every
    configured key is currently unusable (caller should show a friendly
    'temporarily unavailable' message, the same pattern used for the
    single-key case elsewhere in this app)."""
    now = _time.monotonic()
    for key in discover_keys(provider):
        state = _key_state.get((provider, key))
        if state is None:
            return key
        if state.get("permanently_failed"):
            continue
        if state.get("cooldown_until", 0) <= now:
            return key
    return None


def report_rate_limited(provider: str, key: str, cooldown_seconds: int = _COOLDOWN_SECONDS_DEFAULT):
    """Call after a 429 from this specific key. Takes it out of rotation
    for `cooldown_seconds` — long enough to stop hammering an exhausted
    key, short enough that a transient limit doesn't strand the key
    forever, since most free-tier quotas reset on their own."""
    _key_state.setdefault((provider, key), {})
    _key_state[(provider, key)]["cooldown_until"] = _time.monotonic() + cooldown_seconds


def report_invalid(provider: str, key: str):
    """Call after a definitive 'this key is invalid/revoked' response
    (e.g. 401/403 that isn't a rate limit) — excluded for the rest of
    this process's lifetime rather than retried indefinitely."""
    _key_state.setdefault((provider, key), {})
    _key_state[(provider, key)]["permanently_failed"] = True


def report_success(provider: str, key: str):
    """Clears any cooldown on a successful call — a working response
    means the key is healthy again, no need to wait out the rest of a
    cooldown that was really just caution."""
    if (provider, key) in _key_state:
        _key_state[(provider, key)].pop("cooldown_until", None)


def status_summary(provider: str) -> list[dict]:
    """Owner-facing diagnostic view for a settings panel — masked keys
    only. Never shown to a regular end user."""
    now = _time.monotonic()
    result = []
    for key in discover_keys(provider):
        state = _key_state.get((provider, key), {})
        if state.get("permanently_failed"):
            status = "Invalid"
        elif state.get("cooldown_until", 0) > now:
            status = f"Cooling down ({round(state['cooldown_until'] - now)}s)"
        else:
            status = "Available"
        result.append({"key_masked": _mask(key), "status": status})
    return result
