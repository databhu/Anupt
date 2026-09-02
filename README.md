# ✦ ANUPT — Astrology · Numerology · Palmistry · Tarot

An AI-unified personal reading app. Four systems calculate independently;
a Unified Insight Engine finds where they agree or conflict; Gemini writes
the final reading from that evidence — it never invents the calculations.

This is **Phase 1** of the project plan you uploaded, built as a runnable
Python/Streamlit web app rather than a native Android app — that lets you
actually run and test the whole product loop today, on any machine, with
no Android Studio / Play Store pipeline required. The architecture (engines
→ Unified Insight Engine → AI writer, structured JSON everywhere) is the
same one the plan specifies for the eventual Android + FastAPI backend, so
a native client could call the same engine logic later without a rewrite.

## What's real vs. simplified here

| System | Status |
|---|---|
| **Numerology** | Fully deterministic (Pythagorean system) — Life Path, Destiny, Soul Urge, Personality, Birthday, Maturity, Personal Year/Month/Day. |
| **Astrology** | Real Vedic calculations via the Swiss Ephemeris (sidereal/Lahiri) — planets, houses (whole-sign), nakshatra, and Vimshottari Dasha. Divisional charts beyond D1 aren't built yet (matches the plan's own MVP scope). |
| **Tarot** | Full, complete 78-card deck. Draws are seeded per profile+period, so a given day's card is reproducible, not re-randomized on every refresh. |
| **Palmistry** | The image **quality gate** (blur/brightness/hand-presence) is deterministic. The actual line/mount **reading** is AI-vision-assisted (Gemini), not a trained CV model — that's flagged honestly in the UI rather than faked, matching the plan's own instruction not to fabricate palm results. |
| **Unified Insight Engine** | Real cross-system theme scoring (career/finance/relationships/family/growth/spirituality) with an agreement-strength badge and a full evidence trail you can inspect. |
| **Combined vs. Single-Engine mode** | The toggle you asked for — Combined fuses all systems; Single Engine reads one system in isolation with no cross-referencing. |
| **Accounts** | Real sign-up/log-in — passwords are hashed (PBKDF2-SHA256, per-user salt), never stored in plain text. Each account's birth profile and full reading history persist in Postgres and reload automatically the next time that person logs in. A **My Data** page lets anyone see everything saved under their account, or permanently delete it. |

## Look & feel

Built around the actual ANUPT logo you provided: the magenta → indigo → blue crest gradient,
gold sparkle accents, and the brand's own copy ("Insights for a better you") now run through
the whole app — night-sky background, Fraunces/Manrope type, and per-discipline color coding
lifted straight from the logo (numerology's medallions are magenta like the logo's "739" circle,
the natal wheel picks up the same indigo/blue/magenta trio). The crest itself renders as the
real logo image (background removed, see `assets/`), not a redrawn approximation. Buttons and
the wordmark carry the exact three-stop gradient. See `utils/styling.py` for the full
design-token rationale and the contrast math behind which colors are used for text vs. decoration.

## Database — free hosted Postgres

Accounts, birth profiles, and reading history live in Postgres, not a local file — on
purpose, so this can be deployed to Streamlit Community Cloud's free tier, whose
filesystem is wiped on every restart/redeploy. A local SQLite file would lose every
account each time that happened.

You need a `DATABASE_URL` connection string. Either provider below has a free tier
that's permanent (not a trial):

1. **[Neon](https://neon.tech)** — create a project, copy the *pooled* connection
   string from the dashboard.
2. **[Supabase](https://supabase.com)** — create a project, go to
   *Project Settings → Database → Connection string*, use the **Connection pooling**
   variant (port 6543), not the direct one.

Either works — the app just needs a standard `postgresql://...` URL.

### Local development

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your
connection string in. (That file is git-ignored — never commit your real one.) Or,
simpler for local-only testing: `export DATABASE_URL=postgresql://...` before running.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints. The first thing you'll see is a sign-up
screen — create an account, then enter your Gemini API key in the sidebar (kept only
in that browser session, never written to the database — it's the one thing that
isn't saved to your account). Without a key, all the deterministic data still shows;
you just won't get the AI-written narrative.

## Deploying to Streamlit Community Cloud (free)

1. Push this project to a GitHub repo (`.gitignore` already keeps your local
   `secrets.toml` out of it).
2. On [share.streamlit.io](https://share.streamlit.io), connect the repo and point
   it at `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   DATABASE_URL = "postgresql://your-connection-string-here"
   ```
4. Deploy. Every visitor signs up for their own account; their data persists in
   Postgres across restarts and redeploys — the thing plain SQLite couldn't do here.

## What Phase 2+ would add (from your project plan, deferred deliberately)

- ~~Accounts & persistence~~ — done: sign-up/log-in, Postgres-backed profile and reading history
- ~~Free hosting path~~ — done: deployable to Streamlit Community Cloud with Neon/Supabase
- A dedicated backend (FastAPI) if this ever needs an API for something
  other than this Streamlit UI (e.g. a future native Android client)
- Native Android app (Kotlin/Compose) calling this same engine logic over
  an API — the plan's original target platform
- Google/OTP login, password reset, email verification, subscriptions &
  entitlements, admin dashboard
- Push notifications, richer divisional astrology charts, Celtic Cross tarot
- The proper trained hand-landmark CV model for palmistry (the plan itself
  defers this pending licensed training data)

Tell me which of these you want built next and I'll pick up the next phase.
