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

Model choice: gemini-2.5-flash. Google's free tier only covers Flash-class
models (Pro requires billing) — Flash is the stable, widely-documented,
non-preview name that's consistently still free as of this build, and it's
natively multimodal (handles the palmistry photo the same call shape as
text). GEMINI_MODEL is an optional backend override, never a user setting.

Free-tier rate limits (roughly 10 requests/minute, a few hundred/day) are
handled two ways: keep every response short — max_tokens is deliberately
modest, which also serves the "concise summary" product requirement — and
recognize HTTP 429 specifically to show "you're going a bit fast, give it
a moment" instead of a generic error.
"""

import base64
import json
import os
import time as _time

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"
TIMEOUT = 45

SYSTEM_GUARDRAILS = (
    "You are the AI writer for ANUPT, a spiritual-reflection app. You will be given "
    "structured, already-calculated evidence from deterministic Astrology, Numerology "
    "and Tarot engines. Your job is ONLY to interpret, correlate and communicate this "
    "evidence clearly and warmly. Rules:\n"
    "1. Never invent a planetary position, number, or card — use only what's given.\n"
    "2. Explicitly mention which systems support each insight you raise.\n"
    "3. Frame everything as spiritual/personal-reflection guidance, never as a "
    "guaranteed factual outcome.\n"
    "4. Avoid unsafe high-stakes advice (medical, legal, financial-investment "
    "certainty) — encourage the user to consult a relevant professional for those.\n"
    "5. Keep tone encouraging, honest, and grounded — not vague fortune-cookie text.\n"
    "6. Write in plain prose only — no markdown (**, __, #, bullet dashes, etc.), "
    "since your output is displayed as-is, not rendered from markdown. Use plain "
    "paragraph breaks (a blank line) instead of headings or emphasis marks."
)

_FRIENDLY_UNAVAILABLE = (
    "⚠️ The AI reading service isn't available right now — the calculated data above "
    "is still fully accurate. Please try again shortly, or let the app owner know if "
    "this keeps happening."
)
_FRIENDLY_RATE_LIMITED = (
    "⚠️ Lots of requests right now — please wait a few seconds and try again. "
    "The calculated data above is still fully accurate in the meantime."
)

# Simple in-process throttle: free-tier RPM is tight (~10/min), so if two calls land
# within this many seconds of each other we wait rather than firing both and eating
# a 429. This is a courtesy spacing, not a hard queue — it only smooths bursts from
# a single running app instance.
_MIN_INTERVAL_SECONDS = 2.0
_last_call_at = 0.0


def _get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _get_model() -> str:
    """Never user-facing — an optional backend override (GEMINI_MODEL) for when
    Google retires DEFAULT_MODEL, not a setting anyone using the app ever sees."""
    model = os.environ.get("GEMINI_MODEL")
    if model:
        return model
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_MODEL") or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def _throttle():
    global _last_call_at
    elapsed = _time.monotonic() - _last_call_at
    if elapsed < _MIN_INTERVAL_SECONDS:
        _time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
    _last_call_at = _time.monotonic()


def _call(contents: list, system: str = SYSTEM_GUARDRAILS, max_tokens: int = 900) -> str:
    api_key = _get_api_key()
    if not api_key:
        return _FRIENDLY_UNAVAILABLE

    _throttle()
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": max_tokens},
    }
    url = f"{API_BASE}/{_get_model()}:generateContent"
    try:
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=TIMEOUT)
        if resp.status_code == 429:
            return _FRIENDLY_RATE_LIMITED
        if resp.status_code != 200:
            return _FRIENDLY_UNAVAILABLE
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return _FRIENDLY_UNAVAILABLE
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        return text.strip() or _FRIENDLY_UNAVAILABLE
    except requests.exceptions.RequestException:
        return _FRIENDLY_UNAVAILABLE


def _user_turn(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


_SPLIT_INSTRUCTION = (
    "\n\nRespond in exactly this two-part format, with that literal '---SPLIT---' "
    "line between the parts and nothing else on that line:\n"
    "A 2-3 sentence direct, concrete answer — no preamble, no restating the question, "
    "just the meaning in plain warm language.\n"
    "---SPLIT---\n"
    "A fuller reading (2-4 short paragraphs) going deeper into the same evidence."
)


def _split_summary_and_details(text: str) -> tuple[str, str]:
    """One generation call produces both the short summary and the fuller reading —
    not two separate calls — since free-tier rate limits make every extra request
    costly. Falls back gracefully if the model didn't use the exact delimiter."""
    if "---SPLIT---" in text:
        summary, details = text.split("---SPLIT---", 1)
        return summary.strip(), details.strip()
    # Fallback: no delimiter found — use the whole thing as details and a
    # truncated version as the summary, so the UI never ends up with nothing.
    stripped = text.strip()
    summary = stripped[:280] + ("…" if len(stripped) > 280 else "")
    return summary, stripped


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
                              question: str | None = None) -> tuple[str, str]:
    """ANUPT combined page: one call, returns (summary, details)."""
    prompt = (
        f"Reading type: {reading_type}.\n"
        + (f"User's specific question: {question}\n" if question else "")
        + "Structured cross-system evidence — Astrology, Numerology, Tarot, and "
        "Palmistry when a palm reading is included (JSON):\n"
        + json.dumps(unified_evidence, indent=2, default=str)
        + _SPLIT_INSTRUCTION
        + "\nThe fuller part should name which systems agree on each point you raise, "
        "including the palm reading if 'palmistry' is present in the evidence."
    )
    text = _call([_user_turn(prompt)], max_tokens=1200)
    return _split_summary_and_details(text)


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
                             question: str | None = None) -> tuple[str, str]:
    """Single-engine page (Astrology/Numerology/Tarot): one call, returns (summary, details) —
    the summary answers the question directly, details are there for whoever wants to explore."""
    prompt = (
        f"Reading type: {reading_type}. Engine: {engine_name} ONLY — do not reference "
        f"other systems, this is a single-engine reading by the user's choice.\n"
        + (f"User's question: {question}\n" if question else "User asked for a general read.\n")
        + f"Structured {engine_name} data (JSON):\n"
        + json.dumps(engine_data, indent=2, default=str)
        + _SPLIT_INSTRUCTION
    )
    text = _call([_user_turn(prompt)], max_tokens=1000)
    return _split_summary_and_details(text)


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
        + json.dumps(context_evidence, indent=2, default=str) + "\n\nQuestion: "
    )
    contents.append(_user_turn(context_prefix + question))
    return _call(contents, max_tokens=700)


def palm_vision_reading(image_bytes: bytes, mime_type: str, hand_label: str,
                         question: str | None = None) -> tuple[str, str]:
    """
    AI-assisted feature extraction for palmistry — see engines/palmistry.py's
    docstring for why this is the one AI-vision-assisted step. Always labeled
    'AI-assisted' in the UI, never claimed as a deterministic measurement.
    One call, returns (summary, details) — same pattern as the other engines.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        f"This is a photo of a {hand_label} palm submitted to a palmistry app. "
        + (f"The user specifically asked: {question}\n" if question else "User asked for a general read.\n")
        + "First, silently note what you can observe about the major lines (Life, Head, "
        "Heart, Fate, Sun if visible), general mounts, and finger/thumb proportions — "
        "be honest if a feature isn't clearly visible rather than guessing confidently. "
        "Then give a traditional palmistry interpretation from those observations."
        + _SPLIT_INSTRUCTION
        + "\nFrame both parts as spiritual/personal-reflection guidance, not a factual or medical claim."
    )
    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}},
        ],
    }]
    text = _call(contents, max_tokens=1000)
    return _split_summary_and_details(text)
