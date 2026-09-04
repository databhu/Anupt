"""
ANUPT — account + data store, on hosted Postgres.

Migrated from a local SQLite file so this can run on Streamlit Community
Cloud (free tier) — its filesystem is ephemeral and wipes on every restart
or redeploy, which would silently delete every account. A hosted Postgres
(Neon or Supabase both have permanent free tiers) survives that.

Connection string comes from DATABASE_URL — either an environment variable
(handy for local dev / Docker / testing) or Streamlit's own secrets store
(`st.secrets`, the standard way to hand a deployed Streamlit Cloud app a
secret). See README.md for how to get one from Neon/Supabase and where to
put it.

The public functions here (create_user, verify_user, save_profile, etc.)
are unchanged from the SQLite version — app.py doesn't need to know which
database is behind them. Passwords are still PBKDF2-HMAC-SHA256 hashed
with a random per-user salt, never stored in plain text.
"""

import hashlib
import json
import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

import psycopg2
import psycopg2.errors
import psycopg2.extras

PBKDF2_ITERATIONS = 260_000


def get_dsn() -> str:
    """DATABASE_URL from the environment first (simplest for local dev/Docker/tests),
    falling back to Streamlit secrets (how a deployed Community Cloud app gets it)."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        try:
            import streamlit as st
            dsn = st.secrets.get("DATABASE_URL")
        except Exception:
            dsn = None
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Locally, add it to .streamlit/secrets.toml. "
            "On Streamlit Community Cloud, add it in your app's Settings -> Secrets. "
            "It should be a standard postgres:// connection string from Neon or Supabase "
            "(use the *pooled* connection string if the provider offers one)."
        )
    return dsn


@contextmanager
def _conn():
    conn = psycopg2.connect(get_dsn(), cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                name TEXT, dob DATE, birth_time TIME, city TEXT,
                latitude DOUBLE PRECISION, longitude DOUBLE PRECISION, utc_offset DOUBLE PRECISION,
                interests TEXT, updated_at TIMESTAMPTZ,
                numerology_system TEXT DEFAULT 'pythagorean',
                current_name TEXT,
                language TEXT DEFAULT 'en',
                house_system TEXT DEFAULT 'Placidus'
            )
        """)
        # Migration safety: an existing deployment's `profiles` table predates
        # these three columns. CREATE TABLE above only applies to a brand-new
        # database, so anyone upgrading needs these added explicitly — IF NOT
        # EXISTS makes this a no-op on a database that already has them.
        for col_def in (
            "numerology_system TEXT DEFAULT 'pythagorean'",
            "current_name TEXT",
            "language TEXT DEFAULT 'en'",
            "house_system TEXT DEFAULT 'Placidus'",
        ):
            cur.execute(f"ALTER TABLE profiles ADD COLUMN IF NOT EXISTS {col_def}")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                created_at TIMESTAMPTZ NOT NULL,
                reading_type TEXT, mode TEXT, engine TEXT,
                narrative TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS palm_photos (
                user_id INTEGER NOT NULL REFERENCES users(id),
                hand TEXT NOT NULL,
                image_bytes BYTEA NOT NULL,
                mime_type TEXT NOT NULL,
                uploaded_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (user_id, hand)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS insights_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)


def get_cached(cache_key: str, max_age_hours: float = 24.0):
    """Generic reusable-results cache — currently used by the YouTube
    Insights pipeline (engines/youtube_insights.py) to avoid re-running
    Research+Extraction (an API call plus an AI call) for the same sign
    within the same day, but written generically enough for any other
    pipeline that wants the same 'minimize API calls via reuse' pattern.
    Returns None on a miss OR an expired entry — the caller can't tell
    the difference, and doesn't need to; either way it should recompute."""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT data, created_at FROM insights_cache WHERE cache_key = %s", (cache_key,))
        row = cur.fetchone()
    if row is None:
        return None
    age = datetime.now(timezone.utc) - row["created_at"]
    if age > timedelta(hours=max_age_hours):
        return None
    try:
        return json.loads(row["data"])
    except (json.JSONDecodeError, TypeError):
        return None


def set_cached(cache_key: str, data) -> None:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO insights_cache (cache_key, data, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE SET data = EXCLUDED.data, created_at = EXCLUDED.created_at
        """, (cache_key, json.dumps(data, default=str), datetime.now(timezone.utc)))


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    )
    return salt_hex, dk.hex()


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt, pw_hash = _hash_password(password)
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, salt, password_hash, created_at) VALUES (%s, %s, %s, %s)",
                (username, salt, pw_hash, datetime.now(timezone.utc)),
            )
        return True, "Account created."
    except psycopg2.errors.UniqueViolation:
        return False, "That username is already taken."


def verify_user(username: str, password: str) -> int | None:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username.strip(),))
        row = cur.fetchone()
    if row is None:
        return None
    _, computed_hash = _hash_password(password, row["salt"])
    if secrets.compare_digest(computed_hash, row["password_hash"]):
        return row["id"]
    return None


def save_profile(user_id: int, profile: dict):
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO profiles (user_id, name, dob, birth_time, city, latitude, longitude, utc_offset,
                                   interests, updated_at, numerology_system, current_name, language,
                                   house_system)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                name=EXCLUDED.name, dob=EXCLUDED.dob, birth_time=EXCLUDED.birth_time, city=EXCLUDED.city,
                latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude, utc_offset=EXCLUDED.utc_offset,
                interests=EXCLUDED.interests, updated_at=EXCLUDED.updated_at,
                numerology_system=EXCLUDED.numerology_system, current_name=EXCLUDED.current_name,
                language=EXCLUDED.language, house_system=EXCLUDED.house_system
        """, (
            user_id, profile["name"], profile["dob"], profile["birth_time"],
            profile.get("city", ""), profile["latitude"], profile["longitude"], profile["utc_offset"],
            json.dumps(profile.get("interests", [])), datetime.now(timezone.utc),
            profile.get("numerology_system", "pythagorean"), profile.get("current_name") or None,
            profile.get("language", "en"), profile.get("house_system", "Placidus"),
        ))


def get_profile(user_id: int) -> dict | None:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM profiles WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "name": row["name"],
        "dob": row["dob"],              # psycopg2 hands back real date/time objects
        "birth_time": row["birth_time"],  # for DATE/TIME columns — no manual parsing needed
        "city": row["city"] or "",
        "latitude": row["latitude"], "longitude": row["longitude"], "utc_offset": row["utc_offset"],
        "interests": json.loads(row["interests"] or "[]"),
        "numerology_system": row["numerology_system"] or "pythagorean",
        "current_name": row["current_name"] or "",
        "language": row["language"] or "en",
        "house_system": row["house_system"] or "Placidus",
    }


def save_reading(user_id: int, reading_type: str, mode: str, engine: str, narrative: str):
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO readings (user_id, created_at, reading_type, mode, engine, narrative) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, datetime.now(timezone.utc), reading_type, mode, engine, narrative),
        )


def get_readings(user_id: int, limit: int = 30) -> list:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM readings WHERE user_id = %s ORDER BY created_at DESC LIMIT %s", (user_id, limit)
        )
        rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["created_at"] = d["created_at"].isoformat()  # keep app.py's string-slicing (e.g. [:16]) working
        out.append(d)
    return out


def account_created_at(user_id: int) -> str | None:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT created_at FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    return row["created_at"].isoformat() if row else None


def save_palm_photo(user_id: int, hand: str, image_bytes: bytes, mime_type: str):
    """hand is 'left' or 'right'. Overwrites any previous photo for that hand —
    one stored photo per hand per account, matching the profile's own upsert pattern."""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO palm_photos (user_id, hand, image_bytes, mime_type, uploaded_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, hand) DO UPDATE SET
                image_bytes=EXCLUDED.image_bytes, mime_type=EXCLUDED.mime_type,
                uploaded_at=EXCLUDED.uploaded_at
        """, (user_id, hand, psycopg2.Binary(image_bytes), mime_type, datetime.now(timezone.utc)))


def get_palm_photo(user_id: int, hand: str) -> dict | None:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT image_bytes, mime_type, uploaded_at FROM palm_photos WHERE user_id = %s AND hand = %s",
            (user_id, hand),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "image_bytes": bytes(row["image_bytes"]), "mime_type": row["mime_type"],
        "uploaded_at": row["uploaded_at"].isoformat(),
    }


def delete_palm_photo(user_id: int, hand: str):
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM palm_photos WHERE user_id = %s AND hand = %s", (user_id, hand))


def delete_account(user_id: int):
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM readings WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM palm_photos WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM profiles WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
