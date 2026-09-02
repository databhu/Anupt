# ✦ ANUPT — Astrology · Numerology · Palmistry · Tarot

An AI-unified personal reading app. Four systems calculate independently;
a Unified Insight Engine finds where they agree or conflict; an AI writer
turns that evidence into a reading — it never invents the calculations.

Built as a runnable Python/Streamlit web app rather than a native Android
app — that lets you actually run and test the whole product loop today, on
any machine, with no Android Studio / Play Store pipeline required.

## What's real vs. simplified here

| System | Status |
|---|---|
| **Numerology** | Fully deterministic (Pythagorean system) — Life Path, Destiny, Soul Urge, Personality, Birthday, Maturity, Personal Year/Month/Day. |
| **Astrology** | Real Vedic calculations via the Swiss Ephemeris (sidereal/Lahiri) — planets, houses (whole-sign), nakshatra, and Vimshottari Dasha. |
| **Tarot** | Full, complete 78-card deck. Draws are seeded per profile+period, so a given day's card is reproducible, not re-randomized on every refresh. |
| **Palmistry** | The image **quality gate** (blur/brightness/hand-presence) is deterministic. The **reading** is AI-vision-assisted, not a trained CV model — flagged honestly rather than faked. Take or upload a photo once and it's saved to your account (Postgres), so you don't need to re-upload it every visit. |
| **Unified Insight Engine** | Real cross-system theme scoring (career/finance/relationships/family/growth/spirituality) with an agreement-strength grid and a full evidence trail. Folds in your saved Palmistry reading too, when you have one. |
| **Question-first summaries** | Every engine page (Astrology, Numerology, Palmistry, Tarot) leads with your question and a concise 2-3 sentence answer — one AI call, not two, since a second call would double free-tier usage. The full reading is one tap away in an expander for anyone who wants to go deeper. |
| **ANUPT** | The dedicated combined-insights tab — fuses every engine (including Palmistry when you've got a saved reading) into one synthesis, shown as a summary plus a visual grid of which themes the systems agree on. |
| **Accounts** | Real sign-up/log-in — passwords are hashed (PBKDF2-SHA256, per-user salt), never stored in plain text. Birth profile, palm photos, and reading history all persist in Postgres and reload automatically next login. |
| **Birth place** | Type a few letters of a city, pick from real suggestions — latitude, longitude and timezone resolve automatically (Open-Meteo's geocoding API). No field anywhere to type in coordinates by hand. |

## AI provider — Google Gemini, fully backend-configured

The AI writer runs on Gemini. The API key and model are **never** exposed in
the UI — there's no field to paste a key into and no model picker anywhere.
Configuration is entirely a backend secret:

```toml
GEMINI_API_KEY = "your-gemini-key-here"
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
The app defaults to `gemini-2.5-flash` — Google's free tier only covers
Flash-class models (Pro requires billing), and Flash is the stable,
well-documented, non-preview name that's natively multimodal (handles the
palmistry photo the same call shape as text). `GEMINI_MODEL` is an optional
override if Google retires that default later — still a secret, never a
user-facing setting.

**Respecting the free tier:** every reading is a *single* API call that
returns both the short summary and the full detailed reading in one shot
(split on a delimiter), rather than two separate calls — that alone roughly
halves usage against the free tier's per-minute request cap. There's also a
small built-in spacing between consecutive calls, and Gemini's specific
"you're going too fast" (HTTP 429) response gets its own friendly message
instead of a generic error.

Without a key configured, every deterministic calculation (charts, numbers,
cards) still displays normally — you just get a friendly notice instead of
an AI-written narrative, never a crash.

## Look & feel

Light, premium theme built around the real ANUPT logo's own colors — the
magenta → indigo → blue crest gradient and gold accents on a soft
lavender-white surface, with genuine card shadows and Fraunces/Manrope
type. Navigation is a fixed bottom bar (Home · Astrology · Numerology ·
Palmistry · Tarot · ANUPT · Profile), tuned for phone-width screens: 44px
minimum touch targets, no horizontal scroll, and breakpoints down to small
phones. See `utils/styling.py` for the full design-token rationale.

## Database — free hosted Postgres

Accounts, birth profiles, palm photos, and reading history live in
Postgres, not a local file — on purpose, so this can be deployed to
Streamlit Community Cloud's free tier, whose filesystem is wiped on every
restart/redeploy.

You need a `DATABASE_URL` connection string. Either provider below has a
free tier that's permanent (not a trial):

1. **[Neon](https://neon.tech)** — create a project, copy the *pooled*
   connection string from the dashboard.
2. **[Supabase](https://supabase.com)** — create a project, go to
   *Project Settings → Database → Connection string*, use the
   **Connection pooling** variant (port 6543), not the direct one.

## Setup

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
fill in both values:

```toml
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
GEMINI_API_KEY = "your-gemini-key-here"
```

(That file is git-ignored — never commit your real one.) For quick local
testing you can use environment variables instead:

```bash
export DATABASE_URL=postgresql://...
export GEMINI_API_KEY=...
```

City search needs no key (Open-Meteo's free geocoding API).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints. Sign up, fill in your name, date/time
of birth, and search for your birth city — no manual coordinates anywhere.
From there, everything is one tap away in the bottom bar: ask a question on
any engine page for a quick summary, expand for the full reading, or head
to **ANUPT** for everything fused together.

## Deploying to Streamlit Community Cloud (free)

1. Push this project to a GitHub repo (`.gitignore` already keeps your
   local `secrets.toml` out of it).
2. On [share.streamlit.io](https://share.streamlit.io), connect the repo
   and point it at `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   DATABASE_URL = "postgresql://your-connection-string-here"
   GEMINI_API_KEY = "your-gemini-key-here"
   ```
4. Deploy. Every visitor signs up for their own account; their data
   persists in Postgres across restarts and redeploys.

## What's next

- A dedicated backend (FastAPI) if this ever needs an API for something
  other than this Streamlit UI (e.g. a future native Android client)
- Native Android app (Kotlin/Compose) calling this same engine logic
- Google/OTP login, password reset, email verification, subscriptions &
  entitlements, admin dashboard
- Push notifications, richer divisional astrology charts, Celtic Cross tarot
- A trained hand-landmark CV model for palmistry, pending licensed
  training data — the AI-vision reading is the honest stand-in until then

Tell me which of these you want built next and I'll pick up from here.
