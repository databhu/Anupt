"""
ANUPT — YouTube research client (the "Research Agent" role, implemented as
a plain deterministic function rather than a CrewAI agent — see the
architecture note in engines/youtube_insights.py for why).

Uses ONLY the official YouTube Data API v3 (search.list, videos.list) —
never scrapes video pages or transcripts. This is a deliberate legal/
technical boundary: titles, descriptions, channel names, and publish
dates are metadata the official API explicitly returns for third-party
display; a video's actual spoken transcript is not, and pulling it would
mean scraping rather than using the API as intended. Because of that
boundary, the "Extraction" stage that follows this one works from a
video's title + description, not its transcript.

Free tier: the YouTube Data API v3 gives every project 10,000 quota
units/day at no cost; a search.list call costs 100 units, so roughly 100
searches/day per key before hitting the free limit — exactly the kind of
budget the multi-key manager (ai/key_manager.py) helps stretch further
across multiple legitimately-owned keys, with real failover rather than
any attempt to work around a single key's quota.
"""

import logging

import requests

from ai import key_manager

log = logging.getLogger("anupt.youtube")

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_TIMEOUT = 15

# Deterministic query templates per time period — no AI needed to build a
# search query, this is plain string composition. Deliberately includes
# "vedic astrology" rather than just "astrology": Vedic (sidereal,
# Nakshatra-based) astrology is predominantly an Indian tradition and is
# what the rest of this app is built around, whereas plain "astrology"
# pulls in Western tropical content just as readily — this is a stronger,
# more direct signal for the kind of creator this app wants than country
# targeting alone.
_QUERY_TEMPLATES = {
    "week": "{sign} vedic astrology horoscope this week prediction",
    "month": "{sign} vedic astrology horoscope {month_name} prediction",
    "year": "{sign} vedic astrology horoscope {year} prediction",
}

FRIENDLY_UNAVAILABLE = (
    "⚠️ Couldn't reach YouTube's astrology content right now — please try again shortly."
)


def _build_query(zodiac_sign: str, time_period: str, month_name: str | None = None, year: int | None = None) -> str:
    template = _QUERY_TEMPLATES.get(time_period, _QUERY_TEMPLATES["week"])
    return template.format(sign=zodiac_sign, month_name=month_name or "", year=year or "")


def _video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _normalize_item(item: dict) -> dict | None:
    video_id = item.get("id", {}).get("videoId")
    snippet = item.get("snippet", {})
    if not video_id or not snippet.get("title"):
        return None
    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "url": _video_url(video_id),
        "thumbnail_url": (snippet.get("thumbnails", {}).get("default") or {}).get("url", ""),
    }


def search_astrology_videos(zodiac_sign: str, time_period: str = "week",
                             max_results: int = 8, month_name: str | None = None,
                             year: int | None = None, region_code: str | None = "IN") -> dict:
    """The Research stage: a deterministic, rule-based YouTube search — no
    AI involved in deciding what to search for or how to filter results.
    Returns {"videos": [...], "error": None} on success, or
    {"videos": [], "error": <friendly message>} on failure — the caller
    always has a safe shape to branch on, never an exception to catch.

    `region_code` (default "IN") sets YouTube's regionCode parameter,
    documented by Google as showing "content as seen by viewers in this
    country" — a real, legitimate API parameter, but a POPULARITY/
    RELEVANCE bias for that region, not a hard filter guaranteeing a
    result's creator is actually based there. Pass None to search without
    any regional bias at all."""
    api_key = key_manager.get_key("youtube")
    if not api_key:
        log.error("No YOUTUBE_API_KEY configured — YouTube Insights unavailable.")
        return {"videos": [], "error": FRIENDLY_UNAVAILABLE}

    query = _build_query(zodiac_sign, time_period, month_name, year)
    params = {
        "part": "snippet", "q": query, "type": "video", "order": "date",
        "maxResults": max_results, "relevanceLanguage": "en", "key": api_key,
    }
    if region_code:
        params["regionCode"] = region_code

    try:
        resp = requests.get(SEARCH_URL, params=params, timeout=_TIMEOUT)
    except requests.exceptions.RequestException as e:
        log.error("YouTube search request failed: %s", e)
        return {"videos": [], "error": FRIENDLY_UNAVAILABLE}

    if resp.status_code == 403:
        # YouTube Data API returns 403 for both quota-exhausted and a
        # genuinely invalid/restricted key — treat as rate-limited (the far
        # more common case for a free-tier key) and let the key manager's
        # cooldown handle it; a key that's actually invalid will simply
        # keep getting 403s until its cooldown-then-retry cycle surfaces
        # that pattern to the app owner via youtube_key_status().
        log.warning("YouTube API 403 (quota or key issue) on key %s.", key_manager._mask(api_key))
        key_manager.report_rate_limited("youtube", api_key, cooldown_seconds=300)
        next_key = key_manager.get_key("youtube")
        if next_key and next_key != api_key:
            return search_astrology_videos(zodiac_sign, time_period, max_results, month_name, year, region_code)
        return {"videos": [], "error": FRIENDLY_UNAVAILABLE}

    if resp.status_code != 200:
        log.error("YouTube API HTTP %s: %s", resp.status_code, resp.text[:400])
        return {"videos": [], "error": FRIENDLY_UNAVAILABLE}

    key_manager.report_success("youtube", api_key)
    try:
        data = resp.json()
    except ValueError:
        log.error("YouTube API returned non-JSON response.")
        return {"videos": [], "error": FRIENDLY_UNAVAILABLE}

    videos = [v for v in (_normalize_item(item) for item in data.get("items", [])) if v]
    return {"videos": videos, "error": None}


def youtube_key_status() -> list[dict]:
    """Owner-facing diagnostic — masked keys only, same pattern as
    ai.gemini_client.diagnose()."""
    return key_manager.status_summary("youtube")
