"""
ANUPT — city autocomplete and coordinate/timezone resolution.

Uses Open-Meteo's Geocoding API (geocoding-api.open-meteo.com) — free,
no API key, no rate-limit headaches for a small app, and it returns the
IANA timezone name directly alongside lat/lon, so no second lookup is
needed. This is what replaced the old manual latitude/longitude/UTC-offset
fields: the user picks a city from real suggestions, and everything else
is derived automatically.

The UTC offset the astrology engine needs is computed from the timezone
at the person's actual birth date/time (not "today"), so historical DST
rules are respected — this matters for anyone born before a timezone
changed its rules.
"""

from datetime import date, time, datetime
from zoneinfo import ZoneInfo

import requests

SEARCH_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIMEOUT = 6


def search_cities(query: str, count: int = 6) -> list[dict]:
    """
    Returns up to `count` candidate cities for a (partial) name, each as:
        {"label": "Mumbai, Maharashtra, India", "name": ..., "admin1": ...,
         "country": ..., "latitude": ..., "longitude": ..., "timezone": ...}
    Returns [] on no matches or if the service is unreachable — callers
    should treat that as "no suggestions yet", not a hard failure.
    """
    query = (query or "").strip()
    if len(query) < 2:
        return []
    try:
        resp = requests.get(
            SEARCH_URL,
            params={"name": query, "count": count, "language": "en", "format": "json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except (requests.exceptions.RequestException, ValueError):
        return []

    out = []
    for r in results:
        parts = [r.get("name")]
        if r.get("admin1") and r["admin1"] != r.get("name"):
            parts.append(r["admin1"])
        if r.get("country"):
            parts.append(r["country"])
        out.append({
            "label": ", ".join(p for p in parts if p),
            "name": r.get("name"),
            "admin1": r.get("admin1", ""),
            "country": r.get("country", ""),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "timezone": r.get("timezone"),
        })
    return out


def utc_offset_for(timezone_name: str, on_date: date, at_time: time) -> float:
    """
    UTC offset in hours (e.g. 5.5 for IST) that was actually in effect for
    the given timezone at the given local date+time — computed from real
    DST/historical rules via the stdlib zoneinfo database, not a guess.
    """
    local_dt = datetime.combine(on_date, at_time, tzinfo=ZoneInfo(timezone_name))
    offset = local_dt.utcoffset()
    return offset.total_seconds() / 3600 if offset is not None else 0.0
