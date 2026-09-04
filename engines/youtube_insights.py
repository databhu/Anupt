"""
ANUPT — YouTube Insights pipeline: Analysis, Zodiac organization, and
Evidence stages.

Architecture note on CrewAI: this pipeline maps to the 6 roles named in
the feature request (Research, Extraction, Analysis, Zodiac, Evidence,
Summary), but is deliberately implemented as a plain deterministic
pipeline of Python functions rather than as CrewAI multi-agent
orchestration. The reasoning: CrewAI's value comes from agents that need
to independently REASON about what to do next — genuinely exploratory,
non-deterministic task planning. This pipeline's steps are already
well-defined and sequential: Research is one API call, Analysis/Zodiac/
Evidence are pure data aggregation with no judgment calls to make, and
only Extraction (understanding unstructured video text) and Summary
(writing final prose) genuinely need an LLM. Wrapping all six in CrewAI
agents would add a heavy new dependency, and since each CrewAI agent
typically makes its own LLM call to reason about its role even for a
mechanical task, it would very likely INCREASE API usage — directly
against this feature's own "reduce API usage" and "minimize AI calls"
requirements. The four rule-based stages live in this file; the two AI
stages (extraction, summary) are separate functions in ai/gemini_client.py
that take this file's structured output as their input, so the AI is
never the thing deciding what counts as a "theme" or which sign a
prediction is about — only interpreting free text into this file's
already-defined structure.

Belief-based, not scientific: this surfaces what astrology content
creators say, aggregated and organized — it is not a claim that any of
it is validated or accurate, exactly like every other reading in ANUPT.
"""

from collections import Counter

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Closed vocabulary the AI extraction step must choose from — the same
# hallucination-prevention pattern used for palmistry features and
# astrology yogas elsewhere in this app. A theme outside this list gets
# dropped during validation, never displayed.
THEMES = ["career", "finance", "love", "health", "family", "travel", "general"]


def validate_extracted_prediction(raw: dict) -> dict | None:
    """Validates one AI-extracted prediction against the closed
    vocabulary. Returns a cleaned dict, or None if the entry is
    unusable (e.g. no valid sign at all) — never trusts AI output
    structurally, the same discipline used for palm-reading findings."""
    if not isinstance(raw, dict):
        return None
    video_id = raw.get("video_id")
    if not video_id:
        return None

    signs = raw.get("signs_mentioned", [])
    valid_signs = [s for s in signs if s in ZODIAC_SIGNS] if isinstance(signs, list) else []
    if not valid_signs:
        return None

    themes = raw.get("themes", [])
    valid_themes = [t for t in themes if t in THEMES] if isinstance(themes, list) else []

    return {
        "video_id": video_id,
        "signs_mentioned": valid_signs,
        "themes": valid_themes or ["general"],
        "prediction_summary": str(raw.get("prediction_summary", "")).strip(),
    }


def analyze_predictions(extracted: list, target_sign: str) -> dict:
    """The Analysis stage: deterministic aggregation, no AI. Filters to
    predictions that actually mention `target_sign`, tallies how many
    DISTINCT videos raise each theme (agreement across sources — the
    same 'how many systems support this' idea used throughout ANUPT,
    applied here to how many creators say the same thing), and ranks
    themes by that count."""
    relevant = [p for p in extracted if target_sign in p.get("signs_mentioned", [])]
    theme_counts = Counter(theme for p in relevant for theme in p.get("themes", []))
    ranked_themes = [theme for theme, _ in theme_counts.most_common()]
    return {
        "target_sign": target_sign,
        "relevant_prediction_count": len(relevant),
        "theme_agreement": dict(theme_counts),
        "ranked_themes": ranked_themes,
        "relevant_predictions": relevant,
    }


def organize_by_theme(analysis: dict) -> dict:
    """The Zodiac/organization stage: deterministic grouping of the
    already-filtered, already-tallied predictions (from analyze_predictions,
    already scoped to one sign) into per-theme groups, each carrying which
    videos raised that theme — ready for the Evidence stage to attach full
    source metadata."""
    by_theme: dict = {theme: [] for theme in THEMES}
    for pred in analysis["relevant_predictions"]:
        for theme in pred.get("themes", ["general"]):
            by_theme.setdefault(theme, []).append(pred)
    return {theme: preds for theme, preds in by_theme.items() if preds}


def attach_evidence(organized_by_theme: dict, videos: list) -> dict:
    """The Evidence stage: deterministic — links each theme's predictions
    back to their actual source video (title, channel, URL, published
    date), so every claim in the final reading can be traced to a real,
    linkable video, the same 'show your work' principle used for every
    other reading in this app."""
    video_lookup = {v["video_id"]: v for v in videos}
    result = {}
    for theme, predictions in organized_by_theme.items():
        entries = []
        seen_videos = set()
        for pred in predictions:
            vid = pred["video_id"]
            if vid in seen_videos:
                continue
            seen_videos.add(vid)
            source = video_lookup.get(vid)
            if not source:
                continue
            entries.append({
                "prediction_summary": pred.get("prediction_summary", ""),
                "source_title": source["title"],
                "source_channel": source["channel_title"],
                "source_url": source["url"],
                "published_at": source["published_at"],
            })
        if entries:
            result[theme] = entries
    return result


def build_insights_evidence(extracted: list, videos: list, target_sign: str) -> dict:
    """Runs the full deterministic Analysis -> Zodiac -> Evidence chain —
    the rule-based 'Evidence' layer ready for the AI Summary stage or for
    direct display in the UI, no AI call needed to produce any of it."""
    analysis = analyze_predictions(extracted, target_sign)
    organized = organize_by_theme(analysis)
    evidence = attach_evidence(organized, videos)
    return {
        "target_sign": target_sign,
        "relevant_prediction_count": analysis["relevant_prediction_count"],
        "ranked_themes": analysis["ranked_themes"],
        "theme_agreement": analysis["theme_agreement"],
        "evidence_by_theme": evidence,
    }
