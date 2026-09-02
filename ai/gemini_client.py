"""
ANUPT — AI Writer (Gemini)

Strict separation of concerns: every function here receives already-computed,
structured, deterministic evidence from the engines and is only asked to
write and correlate — never to invent a planetary position, a numerology
number, or a card draw. Palm vision reading is the one exception where the
"feature extraction" itself is AI-assisted (see engines/palmistry.py) — that
is always labeled as such in the UI.

Uses the plain REST endpoint so no SDK version pinning is required.
Default model is configurable — Gemini model names change over time, so the
UI exposes the model field rather than hardcoding one permanently.
"""

import base64
import json
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
    "5. Keep tone encouraging, honest, and grounded — not vague fortune-cookie text."
)


def _endpoint(model: str) -> str:
    return f"{API_BASE}/{model}:generateContent"


def _call(model: str, api_key: str, contents: list, system_instruction: str | None = None) -> str:
    if not api_key:
        return ("⚠️ No Gemini API key set. Add your key in the sidebar to enable "
                "AI-written interpretations — the calculated data above is still "
                "fully accurate without it.")
    payload = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    payload["generationConfig"] = {"temperature": 0.8, "maxOutputTokens": 1600}

    try:
        resp = requests.post(
            _endpoint(model), params={"key": api_key}, json=payload, timeout=TIMEOUT
        )
        if resp.status_code != 200:
            return f"⚠️ AI request failed ({resp.status_code}): {resp.text[:300]}"
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "⚠️ The AI returned no content — it may have blocked this prompt. Try again."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip() or "⚠️ Empty AI response."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Couldn't reach the Gemini API: {e}"


def generate_unified_narrative(model: str, api_key: str, reading_type: str,
                                unified_evidence: dict, question: str | None = None) -> str:
    """Combined mode: synthesize across all engines using the Unified Insight Engine's evidence."""
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
    return _call(model, api_key, [{"role": "user", "parts": [{"text": prompt}]}], SYSTEM_GUARDRAILS)


def generate_single_engine_narrative(model: str, api_key: str, engine_name: str,
                                      engine_data: dict, reading_type: str,
                                      question: str | None = None) -> str:
    """Single-engine mode: interpret only one system's structured output, no cross-referencing."""
    prompt = (
        f"Reading type: {reading_type}. Engine: {engine_name} ONLY — do not reference "
        f"other systems, this is a single-engine reading by the user's choice.\n"
        + (f"User's specific question: {question}\n" if question else "")
        + f"Structured {engine_name} data (JSON):\n"
        + json.dumps(engine_data, indent=2, default=str)
        + f"\n\nWrite a clear, warm {engine_name} reading based only on this data."
    )
    return _call(model, api_key, [{"role": "user", "parts": [{"text": prompt}]}], SYSTEM_GUARDRAILS)


def chat_reply(model: str, api_key: str, history: list, question: str, context_evidence: dict) -> str:
    """
    AI Astrologer chat. `history` is a list of {"role": "user"|"model", "text": str}.
    `context_evidence` should already be filtered to what's relevant to the question
    (topic-specific selection happens in app.py, not here) — this function does not
    decide relevance, it only writes from what it's handed.
    """
    contents = []
    for turn in history[-10:]:
        contents.append({"role": turn["role"], "parts": [{"text": turn["text"]}]})
    context_prefix = (
        "Relevant structured evidence for this question (JSON):\n"
        + json.dumps(context_evidence, indent=2, default=str) + "\n\nQuestion: "
    )
    contents.append({"role": "user", "parts": [{"text": context_prefix + question}]})
    return _call(model, api_key, contents, SYSTEM_GUARDRAILS)


def palm_vision_reading(model: str, api_key: str, image_bytes: bytes, mime_type: str,
                         hand_label: str) -> str:
    """
    The one AI-assisted feature-extraction step (see engines/palmistry.py docstring
    for why). Always clearly labeled 'AI-assisted' in the UI, never claimed as a
    deterministic measurement.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        f"This is a photo of a {hand_label} palm submitted to a palmistry app. "
        "Identify what you can observe about the major lines (Life, Head, Heart, "
        "Fate, Sun if visible), general mounts, and finger/thumb proportions. "
        "Then give a brief traditional palmistry interpretation. Be honest about "
        "uncertainty — if a feature isn't clearly visible in the image, say so "
        "rather than guessing confidently. Frame this as spiritual/personal-reflection "
        "guidance, not a factual or medical claim."
    )
    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}},
        ],
    }]
    return _call(model, api_key, contents, SYSTEM_GUARDRAILS)
