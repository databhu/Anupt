"""
ANUPT — AI writer, on Google Gemini.

Same job as before, different provider: every function here receives
already-computed, structured, deterministic evidence from the engines and
is only asked to write and correlate — never to invent a planetary
position, a numerology number, or a card draw.

The API key is entirely backend configuration — never a UI field, never
logged or shown to the person using the app. It comes from GEMINI_API_KEY
(Streamlit secrets or an environment variable); the model is a fixed
constant here, not user-selectable. See README.md for where to set the key.

Model choice: prefers gemini-2.5-flash (Google's free tier covers Flash-class
models; Pro needs billing), but does NOT hard-depend on it. Google retires
model aliases regularly, so on startup the client asks the API which models
the key can actually use and picks the best available Flash-class one; if a
call still 404s, it re-discovers and retries once. GEMINI_MODEL pins a
specific model and skips discovery — a backend override, never a user setting.

Free-tier rate limits (roughly 10 requests/minute, a few hundred/day) are
handled two ways: keep every response short — max_tokens is deliberately
modest, which also serves the "concise summary" product requirement — and
recognize HTTP 429 specifically to show "you're going a bit fast, give it
a moment" instead of a generic error.

Errors: users always see a calm, non-technical message, but the real cause
(HTTP status, API error body, which model failed) goes to the server log so
the app owner can actually diagnose it. diagnose() exposes the same check
from the Profile page for when log access is inconvenient.
"""

import base64
import json
import logging
import os
import re
import time as _time

import requests

from ai import key_manager

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
API_BASE = f"{API_ROOT}/models"
# Starting preference only — NOT a hard dependency. Google retires model aliases
# regularly (gemini-1.5-*, gemini-pro and even gemini-2.5-* have all returned 404
# on v1beta at various points), so if this name isn't available to the key in use,
# _resolve_model() discovers a working one from the live ListModels endpoint
# instead of failing. That's the difference between "the app breaks when Google
# renames something" and "the app keeps working". Google's own 404 error for the
# previous default (gemini-2.5-flash, deprecated for new users) pointed here.
DEFAULT_MODEL = "gemini-3.6-flash"
# Google's own maintained alias — "gets hot-swapped with every new
# release" per the official model docs, specifically so callers don't
# have to track version-specific deprecations themselves. Tried BEFORE
# DEFAULT_MODEL/discovery, since aliases are meant to be called directly
# and generally aren't listed as their own entry in ListModels (so
# checking "is this in the discovered list" would almost always miss it).
# If it 404s, _post_with_fallback's existing retry logic blocklists it
# like any other model and falls through to DEFAULT_MODEL/discovery below
# — this is an additional first attempt layered on top of the existing
# fallback chain, not a replacement for it.
STABLE_ALIAS = "gemini-flash-latest"
TIMEOUT = 45

log = logging.getLogger("anupt.gemini")

SYSTEM_GUARDRAILS = (
    "You are the AI writer for ANUPT, a spiritual-reflection consumer app. You will be given "
    "structured, already-calculated evidence from deterministic Astrology, Numerology, Tarot "
    "or Palmistry engines. Your ONLY job is to interpret, correlate and communicate this "
    "evidence clearly, warmly and CONCISELY, in plain natural-language prose a general "
    "consumer will read on a phone screen. Rules:\n"
    "1. Never invent a planetary position, number, card, or finding — use only what's given "
    "in the structured evidence; never fabricate information beyond it.\n"
    "2. Explicitly mention which systems support each insight you raise.\n"
    "3. Frame everything as spiritual/personal-reflection guidance, never as a guaranteed "
    "factual outcome — clearly distinguish an 'indication' or 'tendency' from a certainty. "
    "Avoid presenting any astrological or spiritual interpretation as a guaranteed fact.\n"
    "4. Avoid unsafe high-stakes advice (medical, legal, financial-investment certainty) — "
    "encourage the user to consult a relevant professional for those.\n"
    "5. Keep tone encouraging, honest, and grounded — not vague fortune-cookie text.\n"
    "6. Write in plain prose only — no markdown (**, __, #, bullet dashes, etc.), no JSON, "
    "no Python dictionaries or other code/data structures, since your output is displayed "
    "as-is to a consumer, not parsed or rendered from markdown. Use plain paragraph breaks "
    "(a blank line) instead of headings or emphasis marks.\n"
    "7. Never expose internal implementation details: no raw numeric scores, house numbers, "
    "confidence percentages, variable names, engine names as code identifiers, or any other "
    "technical/calculation artifact — translate all of it into natural language a non-technical "
    "reader understands. Never repeat raw calculation steps back verbatim; synthesize them "
    "into what they MEAN instead.\n"
    "8. Never mention API errors, model names, HTTP statuses, rate limits, or any other "
    "backend/technical failure — if something went wrong, that is handled entirely outside "
    "of what you write; you only ever write as if the evidence you were given is all there is.\n"
    "9. Be concise: a few short, well-chosen paragraphs beat an exhaustive one — this is a "
    "consumer-facing summary, not a technical report."
)

_FRIENDLY_UNAVAILABLE = (
    "⚠️ The AI reading service isn't available right now — the calculated data above "
    "is still fully accurate. Please try again shortly, or let the app owner know if "
    "this keeps happening."
)
_FRIENDLY_RATE_LIMITED = (
    "⚠️ AI is temporarily busy — please try again in a moment. "
    "The calculated data above is still fully accurate in the meantime."
)
_FRIENDLY_CONFIG_ERROR = (
    "⚠️ AI interpretation is currently unavailable — please check the AI configuration. "
    "The calculated data above is still fully accurate in the meantime."
)

# Simple in-process throttle: free-tier RPM is tight (~10/min), so if two calls land
# within this many seconds of each other we wait rather than firing both and eating
# a 429. This is a courtesy spacing, not a hard queue — it only smooths bursts from
# a single running app instance.
_MIN_INTERVAL_SECONDS = 2.0
_last_call_at = 0.0


def _get_api_key() -> str | None:
    """Backward-compatible single-key lookup, now backed by the multi-key
    manager: with only GEMINI_API_KEY configured this behaves exactly as
    before; with GEMINI_API_KEY_2, _3, ... also configured, this
    transparently starts returning whichever configured key is currently
    least likely to be rate-limited. See ai/key_manager.py."""
    return key_manager.get_key("gemini")


def _configured_model() -> str | None:
    """Explicit backend override (GEMINI_MODEL). Never user-facing. When set, it's
    used as-is and no discovery happens — the operator has said exactly what they want."""
    model = os.environ.get("GEMINI_MODEL")
    if model:
        return model
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_MODEL") or None
    except Exception:
        return None


def _model_sort_key(name: str):
    """Rank candidate models: prefer flash (free-tier friendly), then the highest
    version number, then stable over dated/preview builds."""
    version = 0.0
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    if m:
        try:
            version = float(m.group(1))
        except ValueError:
            version = 0.0
    is_flash = "flash" in name
    # A bare alias like "gemini-2.5-flash" is preferable to "…-flash-001" / "-preview-…"
    is_bare = not re.search(r"(preview|exp|\d{3,}|latest)", name)
    return (is_flash, version, is_bare)


def _discover_models(api_key: str) -> list[str]:
    """Ask the API which models this key can actually use with generateContent.
    Returns bare model names (no 'models/' prefix), best candidate first."""
    try:
        resp = requests.get(f"{API_BASE}", headers={"x-goog-api-key": api_key}, params={"pageSize": 200}, timeout=15)
        if resp.status_code != 200:
            log.warning("ListModels failed: HTTP %s %s", resp.status_code, resp.text[:300])
            return []
        names = []
        for m in resp.json().get("models", []):
            if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                continue
            name = (m.get("name") or "").removeprefix("models/")
            # Skip specialist variants that can't serve general text+vision readings
            if any(bad in name for bad in ("embedding", "aqa", "tts", "imagen", "veo", "live")):
                continue
            if name:
                names.append(name)
        names.sort(key=_model_sort_key, reverse=True)
        return names
    except requests.exceptions.RequestException as e:
        log.warning("ListModels request error: %s", e)
        return []


_resolved_model: str | None = None
# Models confirmed dead for this process — see _mark_model_bad(). This exists
# because of a real, confirmed case: Google's ListModels endpoint kept listing
# gemini-2.5-flash as generateContent-capable long after generateContent itself
# started hard-rejecting it for new-user keys with 404 "no longer available to
# new users". Re-running discovery after that 404 returned the exact same
# answer (the listing hadn't changed), so a naive "rediscover and retry" loops
# forever on the same dead model. Blocklisting the specific model that just
# 404'd guarantees each retry makes real progress toward a model that works.
_known_bad_models: set[str] = set()


def _mark_model_bad(model: str):
    _known_bad_models.add(model)
    global _resolved_model
    if _resolved_model == model:
        _resolved_model = None


def _resolve_model(api_key: str, force_refresh: bool = False) -> str:
    """The model actually used for calls. Explicit override wins; otherwise
    try Google's own stable alias (see STABLE_ALIAS) first, then
    DEFAULT_MODEL if the key really has it (and it isn't known-bad), else
    the best discovered alternative. Cached per process so we don't call
    ListModels on every reading."""
    global _resolved_model
    override = _configured_model()
    if override:
        return override
    if _resolved_model and not force_refresh and _resolved_model not in _known_bad_models:
        return _resolved_model
    if STABLE_ALIAS not in _known_bad_models:
        _resolved_model = STABLE_ALIAS
        return _resolved_model

    available = [m for m in _discover_models(api_key) if m not in _known_bad_models]
    if not available:
        # Nothing usable left (or discovery itself failed) — fall back to the
        # compiled-in default so a transient ListModels blip doesn't take the
        # feature down. If DEFAULT_MODEL is itself known-bad, _post_with_fallback
        # will still surface the real error rather than looping.
        _resolved_model = DEFAULT_MODEL
    elif DEFAULT_MODEL in available:
        _resolved_model = DEFAULT_MODEL
    else:
        _resolved_model = available[0]
        log.warning(
            "Preferred model %r unavailable for this key; using %r instead. Available: %s",
            DEFAULT_MODEL, _resolved_model, ", ".join(available[:8]),
        )
    return _resolved_model


def _get_model() -> str:
    """Back-compat shim for anything still calling the old name."""
    key = _get_api_key()
    return _resolve_model(key) if key else (_configured_model() or DEFAULT_MODEL)


def _throttle():
    global _last_call_at
    elapsed = _time.monotonic() - _last_call_at
    if elapsed < _MIN_INTERVAL_SECONDS:
        _time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
    _last_call_at = _time.monotonic()


def _post_once(api_key: str, model: str, payload: dict):
    url = f"{API_BASE}/{model}:generateContent"
    # Header-based auth (x-goog-api-key) per Google's current official API
    # reference ("All requests to the Gemini API must include a
    # x-goog-api-key header") — switched from query-param auth (?key=),
    # which older/some third-party examples still show and which may be
    # legacy-only rather than guaranteed going forward.
    return requests.post(url, headers={"x-goog-api-key": api_key}, json=payload, timeout=TIMEOUT)


def _post_with_fallback(api_key: str, payload: dict):
    """POST generateContent, transparently working around a listed-but-actually-
    dead model. Excludes any model that 404s and moves to the next best option,
    up to a few attempts, so one stale/deprecated listing can't take the whole
    feature down — and so it can't loop forever on the same dead model either,
    since each attempt permanently blocklists whatever just failed."""
    model = _resolve_model(api_key)
    resp = None
    for _ in range(3):
        resp = _post_once(api_key, model, payload)
        if resp.status_code != 404:
            return resp
        log.warning("Model %r returned 404 (%s); excluding it and trying the next option.",
                    model, resp.text[:200])
        _mark_model_bad(model)
        next_model = _resolve_model(api_key, force_refresh=True)
        if next_model == model:
            break  # discovery has nothing else to offer — stop, don't spin
        model = next_model
    return resp


_last_call_diagnostics: dict = {
    "model": None, "latency_ms": None, "fallback_attempts": 0,
    "outcome": None, "last_error": None, "request_id": None,
}
_request_counter = 0


def get_last_call_diagnostics() -> dict:
    """Developer Mode reads this — never shown to a regular user. Reflects
    only the most recently completed _call(), reset on every new call."""
    return dict(_last_call_diagnostics)


def _call(contents: list, system: str = SYSTEM_GUARDRAILS, max_tokens: int = 900) -> str:
    """Routes every Gemini request through the multi-key manager: tries the
    current best key via _post_with_fallback() (which separately handles a
    dead MODEL name, see that function's docstring — a different concern
    from a dead KEY, composed here rather than tangled together). If a key
    is rate-limited, it's reported to the key manager (so it cools down
    and the NEXT call skips it) and this retries once with whatever key
    the manager offers next — real failover, not just a single attempt
    before giving up."""
    global _request_counter
    _request_counter += 1
    start_time = _time.monotonic()
    _last_call_diagnostics.update({
        "model": _resolved_model, "latency_ms": None, "fallback_attempts": 0,
        "outcome": None, "last_error": None, "request_id": f"req-{_request_counter}",
    })

    def _finish(outcome: str, error: str | None = None):
        _last_call_diagnostics["outcome"] = outcome
        _last_call_diagnostics["last_error"] = error
        _last_call_diagnostics["latency_ms"] = round((_time.monotonic() - start_time) * 1000)
        _last_call_diagnostics["model"] = _resolved_model

    api_key = _get_api_key()
    if not api_key:
        log.error("No GEMINI_API_KEY configured — no AI readings will be generated.")
        _finish("no_key_configured", "no API key")
        return _FRIENDLY_CONFIG_ERROR

    _throttle()
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system}]},
        # temperature/topP/topK are deprecated for the Gemini 3 model
        # family (Google's own docs: "will ignore them entirely") — left
        # out entirely rather than sent-and-ignored, since a future model
        # generation may not be as forgiving about unrecognized/deprecated
        # parameters.
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    tried_keys = set()
    try:
        for _ in range(len(key_manager.discover_keys("gemini")) or 1):
            if api_key in tried_keys:
                break
            tried_keys.add(api_key)
            _last_call_diagnostics["fallback_attempts"] = len(tried_keys) - 1
            resp = _post_with_fallback(api_key, payload)

            if resp.status_code == 429:
                log.info("Gemini rate limited (429) on key %s.", key_manager._mask(api_key))
                key_manager.report_rate_limited("gemini", api_key)
                next_key = _get_api_key()
                if next_key and next_key not in tried_keys:
                    log.info("Failing over to next available key.")
                    api_key = next_key
                    continue
                _finish("rate_limited", "HTTP 429 on all configured keys")
                return _FRIENDLY_RATE_LIMITED

            if resp.status_code in (401, 403):
                log.error("Gemini key %s rejected (HTTP %s) — marking invalid.",
                          key_manager._mask(api_key), resp.status_code)
                key_manager.report_invalid("gemini", api_key)
                next_key = _get_api_key()
                if next_key and next_key not in tried_keys:
                    api_key = next_key
                    continue
                _finish("invalid_key", f"HTTP {resp.status_code} on all configured keys")
                return _FRIENDLY_CONFIG_ERROR

            if resp.status_code != 200:
                # Full detail to the server log (visible to the app owner in
                # Streamlit Cloud's "Manage app" logs) — never to the end user.
                log.error("Gemini HTTP %s (model in use: %r): %s",
                          resp.status_code, _resolved_model, resp.text[:500])
                _finish("http_error", f"HTTP {resp.status_code}")
                return _FRIENDLY_UNAVAILABLE

            key_manager.report_success("gemini", api_key)
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                log.error("Gemini returned no candidates. Response: %s", json.dumps(data)[:500])
                _finish("no_candidates", "empty candidates list")
                return _FRIENDLY_UNAVAILABLE
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip():
                finish_reason = candidates[0].get("finishReason")
                log.error("Gemini returned empty text (finishReason=%s).", finish_reason)
                _finish("empty_text", f"finishReason={finish_reason}")
                return _FRIENDLY_UNAVAILABLE
            _finish("success")
            return text.strip()

        _finish("all_keys_exhausted", "loop completed without a returned result")
        return _FRIENDLY_UNAVAILABLE  # every configured key was tried and failed
    except requests.exceptions.RequestException as e:
        log.error("Gemini request failed: %s", e)
        _finish("network_error", str(e)[:200])
        return _FRIENDLY_UNAVAILABLE


def diagnose() -> dict:
    """Owner-facing connectivity check, surfaced in Profile → Settings.
    Returns plain status info (never the key itself) so a deployment problem
    can be identified without digging through server logs. Uses the exact same
    _post_with_fallback() path a real reading does, so this reflects what users
    actually experience rather than a simpler, more optimistic check."""
    api_key = _get_api_key()
    result = {
        "api_key_configured": bool(api_key),
        "model_override": _configured_model(),
        "preferred_model": DEFAULT_MODEL,
        "keys_configured": key_manager.status_summary("gemini"),
    }
    if not api_key:
        result["status"] = "No GEMINI_API_KEY found in secrets or environment."
        return result

    available = _discover_models(api_key)
    result["models_visible_to_key"] = len(available)
    result["example_models"] = available[:8]
    if not available:
        result["status"] = ("Couldn't list models — the key may be invalid, restricted, "
                            "or the Generative Language API isn't enabled for its project.")
        return result

    try:
        resp = _post_with_fallback(api_key, {
            "contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}],
            "generationConfig": {"maxOutputTokens": 10},
        })
        result["model_in_use"] = _resolved_model or _configured_model() or DEFAULT_MODEL
        result["known_bad_models"] = sorted(_known_bad_models)
        result["test_call_http_status"] = resp.status_code
        result["status"] = "Working" if resp.status_code == 200 else f"Test call failed: {resp.text[:300]}"
    except requests.exceptions.RequestException as e:
        result["status"] = f"Test call could not reach the API: {e}"
    return result


def _user_turn(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


_SPLIT_INSTRUCTION = (
    "\n\nRespond in exactly this three-part format, with those literal '---SPLIT---' "
    "lines between the parts and nothing else on those lines:\n"
    "A single whole number from 1 to 10 rating how favorable the evidence looks for "
    "this reading overall (1 = very challenging, 10 = highly favorable). Output ONLY "
    "the digit(s), nothing else on this line.\n"
    "---SPLIT---\n"
    "A 2-3 sentence direct, concrete answer — no preamble, no restating the question, "
    "just the meaning in plain warm language.\n"
    "---SPLIT---\n"
    "A fuller reading (2-4 short paragraphs) going deeper into the same evidence."
)


def _parse_score(text: str) -> int | None:
    m = re.search(r"\d+", text)
    if not m:
        return None
    return max(1, min(10, int(m.group())))


def _split_score_summary_details(text: str) -> tuple[int | None, str, str]:
    """One generation call produces the favorability score, the short summary, and
    the fuller reading — not three separate calls — since free-tier rate limits make
    every extra request costly. Degrades gracefully if the model doesn't follow the
    exact format: a missing score becomes None (UI hides the score badge rather than
    showing a made-up number), and completely unstructured output still becomes a
    reasonable summary/details split rather than an empty response."""
    parts = text.split("---SPLIT---")
    if len(parts) == 3:
        score = _parse_score(parts[0])
        return score, parts[1].strip(), parts[2].strip()
    if len(parts) == 2:
        # Model produced a summary/details split but skipped the score somehow.
        return None, parts[0].strip(), parts[1].strip()
    stripped = text.strip()
    summary = stripped[:280] + ("…" if len(stripped) > 280 else "")
    return None, summary, stripped


def generate_unified_narrative(reading_type: str, unified_evidence: dict, question: str | None = None) -> str:
    """Combined mode, full narrative only (used by Home's quick 'today' card, which
    doesn't need the summary/details split)."""
    prompt = (
        f"Reading type: {reading_type}.\n"
        + (f"User's specific question: {question}\n" if question else "")
        + "Structured cross-system evidence (JSON):\n"
        + json.dumps(unified_evidence, indent=2, default=str)
        + "\n\nWrite a unified reading. Structure it as: an opening synthesis (2-3 sentences), "
        "then the top 2-3 themes in order of strength, each naming which systems agree, "
        "then one grounded closing reflection. Do not restate raw numbers — translate them "
        "into meaning."
    )
    return _call([_user_turn(prompt)], max_tokens=1100)


def generate_unified_reading(reading_type: str, unified_evidence: dict,
                              question: str | None = None, language: str = "en") -> tuple[int | None, str, str]:
    """ANUPT combined page: one call, returns (score, summary, details).
    Structured so the fuller reading always works through each
    contributing system's own answer first, then closes with a genuine
    synthesis — never a single generic paragraph that blurs which system
    said what, and never just a restatement of the raw evidence."""
    systems_present = ["Astrology", "Numerology", "Tarot"]
    if "palmistry" in unified_evidence:
        systems_present.append("Palmistry")
    systems_line = ", ".join(systems_present)

    prompt = (
        f"Reading type: {reading_type}.\n"
        + (f"User's specific question: {question}\n" if question else
           "The user asked for a general reading — use whatever timeframe the evidence itself "
           "centers on (e.g. current dasha/transits, this year's numerology, today's cards).\n")
        + "Structured cross-system evidence — Astrology, Numerology, Tarot, and "
        "Palmistry when a palm reading is included (JSON):\n"
        + json.dumps(unified_evidence, indent=2, default=str)
        + "\n\nYou are writing as an experienced astrologer/reader speaking directly to this "
        "person — warm, direct, plain-spoken, never clinical or like a report. For every point "
        "you raise: explain WHY it's indicated (what in the evidence points to it, in a sentence "
        "a non-technical person immediately understands, not jargon), and offer ONE simple, "
        "concrete, doable tip connected to it — something they could actually act on this week, "
        "not vague platitudes like 'stay positive'. Do not pad the reading with generic "
        "statements that could apply to anyone; every sentence should trace back to something "
        "specific in the evidence above.\n"
        + _SPLIT_INSTRUCTION
        + f"\n\nFor the fuller reading (the third part), structure it exactly as follows: first, "
        f"one short paragraph EACH for {systems_line} — what that system specifically indicates "
        f"for this question/timeframe, why, and one tip. Name the system at the start of its "
        f"paragraph (e.g. 'Astrology suggests...'). Then close with a final short paragraph "
        f"that genuinely synthesizes them — not a recap, but what emerges when you hold all of "
        f"them together (where they reinforce each other, and where one adds nuance another "
        f"doesn't cover)."
        + ("\nRespond entirely in Marathi (मराठी), not English." if language == "mr" else "")
    )
    text = _call([_user_turn(prompt)], max_tokens=1400)
    return _split_score_summary_details(text)


def generate_single_engine_narrative(engine_name: str, engine_data: dict, reading_type: str,
                                      question: str | None = None) -> str:
    """Full narrative only, single engine (kept for the reading-history/full-text case)."""
    prompt = (
        f"Reading type: {reading_type}. Engine: {engine_name} ONLY — do not reference "
        f"other systems, this is a single-engine reading by the user's choice.\n"
        + (f"User's specific question: {question}\n" if question else "")
        + f"Structured {engine_name} data (JSON):\n"
        + json.dumps(engine_data, indent=2, default=str)
        + f"\n\nWrite a clear, warm {engine_name} reading based only on this data."
    )
    return _call([_user_turn(prompt)], max_tokens=900)


def generate_engine_reading(engine_name: str, engine_data: dict, reading_type: str,
                             question: str | None = None, focus: str | None = None,
                             language: str = "en") -> tuple[int | None, str, str]:
    """Single-engine page (Astrology/Numerology/Tarot): one call, returns
    (score, summary, details) — the summary answers the question directly,
    details are there for whoever wants to explore.

    `focus` narrows the reading to one life area (e.g. "career", "finance",
    "relationships", "personal growth") for the Numerology page's specialized
    readings — still built from the exact same structured data, the AI is
    just asked to weight its interpretation toward that area rather than
    calculate anything new. `language` == "mr" asks for a Marathi response;
    the underlying data and prompt stay in English either way, since the
    numbers themselves aren't language-dependent.

    Rule Engine -> Evidence -> AI -> Final Reading: when `engine_data`
    includes a "rule_based_findings" key (from engines.astrology_interpretation
    or engines.numerology_interpretation), that's pre-computed, already-
    interpreted evidence — deterministic, not AI output. The AI's job then
    shifts from independently interpreting raw numbers to synthesizing and
    personalizing findings that already exist, which is both more
    consistent (the same placement always gets the same base interpretation)
    and more transparent (a user can see the rule-based finding directly,
    independent of whatever the AI goes on to say about it)."""
    has_rule_findings = "rule_based_findings" in engine_data
    prompt = (
        f"Reading type: {reading_type}. Engine: {engine_name} ONLY — do not reference "
        f"other systems, this is a single-engine reading by the user's choice.\n"
        + (f"Focus this reading specifically on: {focus}.\n" if focus else "")
        + (f"User's question: {question}\n" if question else "User asked for a general read.\n")
        + f"Structured {engine_name} data (JSON) — the ONLY numbers you may reference; "
        "never state a number that isn't in this data:\n"
        + json.dumps(engine_data, indent=2, default=str)
        + (
            "\n\nThe 'rule_based_findings' field above is NOT something to re-derive — it's "
            "already-computed, deterministic interpretation (strength tiers, exceptions like "
            "combustion, number relationships) from a rule engine, not from you. Your job is to "
            "SYNTHESIZE and PERSONALIZE these existing findings toward the user's question — weave "
            "the strongest, most relevant ones into a coherent answer — not to independently "
            "reinterpret the raw positions as if this evidence didn't already exist."
            if has_rule_findings else ""
        )
        + "\n\nYou are writing as an experienced astrologer/reader speaking directly to this "
        "person — warm, direct, plain-spoken, never clinical. Explain WHY each point you raise "
        "is indicated (what in the evidence points to it, in a sentence a non-technical person "
        "immediately understands, not jargon), and offer ONE simple, concrete, doable tip "
        "connected to it — something they could actually act on this week, not vague platitudes. "
        "Do not pad the reading with generic statements that could apply to anyone; every "
        "sentence should trace back to something specific in the evidence above."
        + _SPLIT_INSTRUCTION
        + ("\nRespond entirely in Marathi (मराठी), not English." if language == "mr" else "")
    )
    text = _call([_user_turn(prompt)], max_tokens=1000)
    return _split_score_summary_details(text)



def chat_reply(history: list, question: str, context_evidence: dict) -> str:
    """
    AI Astrologer chat. `history` is a list of {"role": "user"|"model", "text": str}.
    `context_evidence` should already be filtered to what's relevant to the question
    (topic-specific selection happens in app.py, not here).
    """
    contents = []
    for turn in history[-10:]:
        contents.append({"role": turn["role"], "parts": [{"text": turn["text"]}]})
    context_prefix = (
        "Relevant structured evidence for this question (JSON):\n"
        + json.dumps(context_evidence, indent=2, default=str)
        + "\n\nAnswer as an experienced astrologer speaking directly to this person — plain, "
        "warm, conversational, a couple of sentences unless the question genuinely needs more. "
        "Where relevant, briefly note WHY (what in the evidence points to it) and offer one "
        "small, concrete thing they could actually do about it — not a vague platitude.\n\n"
        "Question: "
    )
    contents.append(_user_turn(context_prefix + question))
    return _call(contents, max_tokens=700)


def _palm_vocabulary_prompt_block() -> str:
    """Renders engines.palmistry.PALM_FEATURES as prompt text — the closed
    vocabulary the model is instructed to choose from, so a hallucinated
    feature name is something the app can actually detect and drop rather
    than an open-ended risk."""
    from engines import palmistry
    lines = []
    for name, data in palmistry.PALM_FEATURES.items():
        lines.append(f"- {name} ({data['category']}): typically {data['typical_location']}")
    return "\n".join(lines)


def palm_vision_reading_structured(image_bytes: bytes, mime_type: str, hand_label: str,
                                    question: str | None = None, language: str = "en") -> dict:
    """
    Structured, evidence-linked palm reading: IMAGE -> DETECTION -> VISUAL
    PROOF -> TRADITIONAL INTERPRETATION -> PERSONALIZED READING. Replaces
    the old free-text palm_vision_reading() — instead of prose the model
    is asked to invent unsupervised, this asks for a closed JSON structure
    naming only features from a fixed vocabulary, each with a confidence
    level and (when the model can reasonably localize it) an approximate
    bounding box the app uses to actually draw on the user's photo. A
    feature the model can't confidently see should be omitted entirely,
    not guessed at with low confidence — confidence describes uncertainty
    about a real observation, not permission to speculate.

    Returns a dict; on any failure (missing key, bad JSON, model declines)
    returns {"error": <user-safe message>} instead of raising, so the UI
    always has something safe to branch on.
    """
    from engines import palmistry

    api_key = _get_api_key()
    if not api_key:
        return {"error": _FRIENDLY_UNAVAILABLE}

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    marking_types = ", ".join(palmistry.MARKING_TYPES)
    prompt = (
        f"This is a photo of a {hand_label} palm submitted to a palmistry app. "
        + (f"The user specifically asked: {question}\n" if question else "User asked for a general read.\n")
        + "\nFIRST, judge the photo itself: is it clear enough to make out individual line paths and "
        "mount areas (not just a blurry hand shape)? If not, say so — do not analyze a photo you can't "
        "actually read clearly.\n\n"
        + "If it IS clear enough, identify ONLY features from this exact list that you can actually see "
        "on THIS photo — do not name anything not on this list, and skip anything you can't confidently "
        "make out rather than guessing:\n" + _palm_vocabulary_prompt_block()
        + f"\n\nMarking types that may appear ON a line or mount, if you see them: {marking_types}.\n\n"
        + "Respond with ONLY a single JSON object (no markdown fences, no text before or after), "
        "in exactly this shape:\n"
        '{\n'
        '  "image_quality_sufficient": true or false,\n'
        '  "quality_issue": "specific reason if false, else null",\n'
        '  "findings": [\n'
        '    {\n'
        '      "feature": "<exact name from the list above>",\n'
        '      "confidence": "High" | "Medium" | "Low",\n'
        '      "location_description": "<where on THIS hand, in plain words>",\n'
        '      "bbox": [x_min, y_min, x_max, y_max] using 0-1 normalized coordinates '
        '(fraction of image width/height) roughly bounding where this feature is in the photo, '
        'or null if you cannot reasonably localize it,\n'
        '      "markings": ["<marking type near/on this feature, if any>"],\n'
        '      "note": "<what specifically stands out about THIS one on THIS hand>"\n'
        '    }\n'
        '  ],\n'
        '  "life_areas": {\n'
        '    "personality": {"narrative": "2-3 sentences", "linked_features": ["<feature names this draws on>"]},\n'
        '    "career": {"narrative": "...", "linked_features": [...]},\n'
        '    "finance": {"narrative": "...", "linked_features": [...]},\n'
        '    "relationships": {"narrative": "...", "linked_features": [...]},\n'
        '    "strengths": {"narrative": "...", "linked_features": [...]},\n'
        '    "challenges": {"narrative": "...", "linked_features": [...]},\n'
        '    "life_phases": {"narrative": "...", "linked_features": [...]}\n'
        '  },\n'
        '  "score": <1-10 favorability integer, or null if image_quality_sufficient is false>,\n'
        '  "summary": "<2-3 sentence plain-language headline for this reading>"\n'
        '}\n\n'
        "Every life_areas narrative MUST only reference findings that are actually in your "
        "findings list — never introduce a claim that isn't tied to something you reported seeing. "
        "If image_quality_sufficient is false, leave findings empty and every life_areas narrative "
        "should explain you can't respond directly from this photo rather than reading anyway. "
        "Frame every narrative as spiritual/personal-reflection guidance, not a factual or medical claim."
        + ("\nWrite every narrative and note in Marathi (मराठी), not English — feature names in the "
           "JSON structure itself should stay in English." if language == "mr" else "")
    )
    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}},
        ],
    }]
    raw_text = _call(contents, max_tokens=2200)
    if raw_text == _FRIENDLY_UNAVAILABLE or raw_text == _FRIENDLY_RATE_LIMITED:
        return {"error": raw_text}
    return _parse_structured_palm_reading(raw_text)


def _parse_structured_palm_reading(raw_text: str) -> dict:
    """Extracts and validates the JSON object from the model's response.
    Strips markdown code fences if present, finds the outermost {...} if
    there's stray text around it, and filters findings down to the closed
    vocabulary — a feature name the model invented despite instructions
    gets dropped rather than displayed as if it were legitimate."""
    from engines import palmistry

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"error": "The AI's response wasn't in the expected format. Please try again."}

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"error": "The AI's response wasn't in the expected format. Please try again."}

    valid_names = set(palmistry.PALM_FEATURES.keys())
    valid_confidence = {"High", "Medium", "Low"}
    clean_findings = []
    for f in data.get("findings", []):
        if not isinstance(f, dict) or f.get("feature") not in valid_names:
            continue  # hallucinated or malformed feature name — dropped, not displayed
        bbox = f.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            bbox = None
        elif not all(0 <= v <= 1 for v in bbox):
            bbox = None  # out-of-range coordinates aren't usable for drawing — treat as unlocalized
        clean_findings.append({
            "feature": f["feature"],
            "category": palmistry.PALM_FEATURES[f["feature"]]["category"],
            "confidence": f.get("confidence") if f.get("confidence") in valid_confidence else "Low",
            "location_description": str(f.get("location_description", "")),
            "bbox": bbox,
            "markings": [m for m in f.get("markings", []) if m in palmistry.MARKING_TYPES] if isinstance(f.get("markings"), list) else [],
            "note": str(f.get("note", "")),
            "traditional_meaning": palmistry.PALM_FEATURES[f["feature"]]["traditional_meaning"],
        })

    clean_life_areas = {}
    for area in palmistry.LIFE_AREAS:
        entry = data.get("life_areas", {}).get(area, {})
        if not isinstance(entry, dict):
            entry = {}
        linked = entry.get("linked_features", [])
        clean_life_areas[area] = {
            "narrative": str(entry.get("narrative", "")),
            "linked_features": [f for f in linked if f in valid_names] if isinstance(linked, list) else [],
        }

    score = data.get("score")
    score = max(1, min(10, int(score))) if isinstance(score, (int, float)) else None

    return {
        "image_quality_sufficient": bool(data.get("image_quality_sufficient", False)),
        "quality_issue": data.get("quality_issue"),
        "findings": clean_findings,
        "life_areas": clean_life_areas,
        "score": score,
        "summary": str(data.get("summary", "")),
    }


def palm_comparison_reading(dominant_reading: dict, non_dominant_reading: dict,
                             dominant_label: str, language: str = "en") -> dict:
    """Compares two ALREADY-COMPUTED structured readings (no new image
    analysis — reuses each hand's own findings) into a dominant-vs-non-
    dominant narrative, per the traditional convention that the dominant
    hand shows how you express yourself day to day and the non-dominant
    hand shows innate tendencies you were 'born with'."""
    api_key = _get_api_key()
    if not api_key:
        return {"error": _FRIENDLY_UNAVAILABLE}

    prompt = (
        f"Compare two structured palm readings for the same person — their {dominant_label} "
        "(dominant) hand and their other (non-dominant) hand. Traditionally the dominant hand shows "
        "how someone expresses themselves day to day, while the non-dominant hand shows more innate, "
        "'born with' tendencies. Only compare features present in BOTH structured readings below; "
        "do not invent a feature that isn't in either.\n\n"
        f"Dominant hand findings (JSON):\n{json.dumps(dominant_reading.get('findings', []), default=str)}\n\n"
        f"Non-dominant hand findings (JSON):\n{json.dumps(non_dominant_reading.get('findings', []), default=str)}\n\n"
        "Respond with ONLY a JSON object: "
        '{"narrative": "3-5 sentences on the most meaningful differences/similarities found", '
        '"differences": [{"feature": "<name present in both>", "dominant_note": "...", '
        '"non_dominant_note": "...", "interpretation": "..."}]}'
        + ("\nWrite in Marathi (मराठी), not English." if language == "mr" else "")
    )
    raw_text = _call([_user_turn(prompt)], max_tokens=900)
    if raw_text == _FRIENDLY_UNAVAILABLE or raw_text == _FRIENDLY_RATE_LIMITED:
        return {"error": raw_text}

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"error": "The AI's response wasn't in the expected format. Please try again."}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"error": "The AI's response wasn't in the expected format. Please try again."}


def classify_navigation_intent(query: str, destinations: dict) -> dict:
    """The AI fallback for the 'Ask ANUPT' navigation chatbot — called ONLY
    when utils.nav_router's rule-based keyword matching finds nothing at
    all. `destinations` is {name: description}, the same closed vocabulary
    the rule router uses, so the model can only ever pick a page that
    actually exists — a lightweight classification task, not a generative
    one, kept cheap and fast on purpose (this runs on every unmatched chat
    message, unlike a reading which only runs when the user asks for one).

    Returns {"destination": <name>} or {"destination": None} if even the
    AI can't find a reasonable match (the UI should fall back to showing
    every option rather than guessing)."""
    api_key = _get_api_key()
    if not api_key:
        return {"destination": None}

    options_text = "\n".join(f"- {name}: {desc}" for name, desc in destinations.items())
    prompt = (
        f"A user of an astrology/numerology/palmistry/tarot app typed this into a "
        f"navigation search box: \"{query}\"\n\n"
        f"Which ONE of these app sections do they most likely want? Only these exist:\n"
        f"{options_text}\n\n"
        f"Respond with ONLY a JSON object, no other text: "
        f'{{"destination": "<exact name from the list above>"}} — or '
        f'{{"destination": null}} if none of them are a reasonable match for this request.'
    )
    text = _call([_user_turn(prompt)], max_tokens=60)
    if text in (_FRIENDLY_UNAVAILABLE, _FRIENDLY_RATE_LIMITED):
        return {"destination": None}

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"```$", "", stripped.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1:
        return {"destination": None}
    try:
        parsed = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return {"destination": None}

    destination = parsed.get("destination")
    if destination not in destinations:
        return {"destination": None}  # hallucinated/invalid destination — never trusted
    return {"destination": destination}


def extract_youtube_predictions(videos: list) -> list:
    """The Extraction stage (the one place raw video text meets the AI):
    ONE batched call across every video's title + description — not one
    call per video — reading unstructured text and pulling out which
    zodiac signs it discusses and what themes it touches, as closed-
    vocabulary structured JSON. Deliberately works from title+description
    only, never a transcript — see engines/youtube_client.py's module
    docstring for why. Every entry is validated against the closed
    vocabulary in engines.youtube_insights before being trusted; anything
    that fails validation (hallucinated sign, missing fields) is silently
    dropped rather than surfaced as if it were real."""
    from engines import youtube_insights

    if not videos:
        return []

    api_key = _get_api_key()
    if not api_key:
        return []

    videos_text = "\n\n".join(
        f"video_id: {v['video_id']}\ntitle: {v['title']}\ndescription: {v['description'][:300]}"
        for v in videos
    )
    signs_list = ", ".join(youtube_insights.ZODIAC_SIGNS)
    themes_list = ", ".join(youtube_insights.THEMES)
    prompt = (
        "Below are YouTube video titles and descriptions from astrology content creators. "
        "For EACH video, identify which zodiac signs (if any) it discusses and which themes "
        "it touches, using ONLY these exact values:\n"
        f"Valid signs: {signs_list}\n"
        f"Valid themes: {themes_list}\n\n"
        "If a video isn't really about a specific sign or has no clear astrological "
        "prediction content, give it an empty signs_mentioned list.\n\n"
        f"{videos_text}\n\n"
        "Respond with ONLY a JSON array (no markdown fences, no other text), one object per "
        "video, in exactly this shape:\n"
        '[{"video_id": "<id>", "signs_mentioned": ["<sign>", ...], "themes": ["<theme>", ...], '
        '"prediction_summary": "<1-2 sentence summary of what this video predicts, in your own words>"}]'
    )
    raw_text = _call([_user_turn(prompt)], max_tokens=1500)
    if raw_text in (_FRIENDLY_UNAVAILABLE, _FRIENDLY_RATE_LIMITED):
        return []

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip())
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        log.error("YouTube extraction response wasn't a JSON array.")
        return []
    try:
        raw_predictions = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        log.error("YouTube extraction response wasn't valid JSON.")
        return []
    if not isinstance(raw_predictions, list):
        return []

    validated = [youtube_insights.validate_extracted_prediction(p) for p in raw_predictions]
    return [v for v in validated if v is not None]


def generate_youtube_summary(evidence: dict, sign: str, question: str | None = None,
                              language: str = "en") -> tuple[int | None, str, str]:
    """The Summary stage: same (score, summary, details) shape as every
    other reading in this app. Takes the ALREADY-AGGREGATED evidence from
    engines.youtube_insights.build_insights_evidence() — themes, their
    cross-source agreement counts, and their source videos — and writes
    the final reading. The AI never decides what the themes are or how
    much sources agree; it only synthesizes what the rule-based pipeline
    already determined."""
    prompt = (
        f"Reading type: YouTube astrology insights for {sign}.\n"
        + (f"User's question: {question}\n" if question else "User asked for a general read.\n")
        + "Structured evidence aggregated from real astrology YouTube videos (JSON) — themes "
        "are ranked by how many DIFFERENT creators independently raised them, which is the "
        "signal to emphasize (a theme several creators agree on is more noteworthy than one "
        "only a single video mentions):\n"
        + json.dumps(evidence, indent=2, default=str)
        + _SPLIT_INSTRUCTION
        + "\nMention the level of cross-creator agreement where it's notable (e.g. 'several "
        "astrologers this week are focused on...'). Never invent a theme or prediction that "
        "isn't in the evidence above."
        + ("\nRespond entirely in Marathi (मराठी), not English." if language == "mr" else "")
    )
    text = _call([_user_turn(prompt)], max_tokens=900)
    return _split_score_summary_details(text)
