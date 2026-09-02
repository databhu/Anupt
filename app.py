"""
ANUPT — AI Astrology, Numerology, Palmistry & Tarot life-reading app.
Phase 1 build: deterministic engines + Unified Insight Engine + Gemini AI writer,
with a Combined (unified) vs Single-Engine reading mode toggle, as requested.

Run: streamlit run app.py
"""

import hashlib
from datetime import date, time, datetime, timezone

import pytz
import streamlit as st
from PIL import Image

from engines import numerology, astrology, tarot, palmistry, unified
from ai import gemini_client
from utils import styling, chart_svg
from auth import store as auth_store

st.set_page_config(page_title="ANUPT", page_icon="assets/logo_mark.png", layout="wide", initial_sidebar_state="expanded")
styling.inject()

try:
    auth_store.init_db()
except Exception as e:
    st.error(
        "Couldn't connect to the database. This app needs a `DATABASE_URL` — a Postgres "
        "connection string from a free host like Neon or Supabase.\n\n"
        "- **Local:** add it to `.streamlit/secrets.toml` or set it as an environment variable.\n"
        "- **Streamlit Community Cloud:** add it under your app's *Settings → Secrets*.\n\n"
        f"Details: {e}"
    )
    st.stop()

# ----------------------------------------------------------------------------
# Session state initialisation
# ----------------------------------------------------------------------------
defaults = {
    "user_id": None,
    "username": None,
    "profile": None,          # dict of birth details
    "numerology_profile": None,
    "chart": None,
    "mode": "Combined (Unified)",
    "gemini_key": "",
    "gemini_model": gemini_client.DEFAULT_MODEL,
    "chat_history": [],
    "last_tarot_draw": None,
    "geocode_prefill": None,
    "nav": "Home",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def render_auth_screen():
    """Login / sign-up gate — nothing else renders until this passes."""
    styling.hero()
    st.markdown("<div style='max-width:420px;margin:0 auto;'>", unsafe_allow_html=True)
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            if not u or not p:
                st.error("Enter both a username and password.")
            else:
                uid = auth_store.verify_user(u, p)
                if uid is not None:
                    st.session_state.user_id = uid
                    st.session_state.username = u.strip()
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

    with tab_signup:
        with st.form("signup_form"):
            su = st.text_input("Choose a username", key="signup_u")
            sp1 = st.text_input("Choose a password", type="password", key="signup_p1")
            sp2 = st.text_input("Confirm password", type="password", key="signup_p2")
            signed_up = st.form_submit_button("Create account", type="primary")
        if signed_up:
            if sp1 != sp2:
                st.error("Passwords don't match.")
            else:
                ok, msg = auth_store.create_user(su, sp1)
                if ok:
                    uid = auth_store.verify_user(su, sp1)
                    st.session_state.user_id = uid
                    st.session_state.username = su.strip()
                    st.rerun()
                else:
                    st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='anupt-disclaimer' style='text-align:center;'>Your account, birth profile, and "
        "reading history are stored in this app's own database, under your account — nothing is "
        "sent anywhere else.</p>",
        unsafe_allow_html=True,
    )


if st.session_state.user_id is None:
    render_auth_screen()
    st.stop()

# Load the saved profile from the database once per login (session_state stays
# empty until now, so this only fires right after logging in, not every rerun).
if st.session_state.profile is None:
    _loaded = auth_store.get_profile(st.session_state.user_id)
    if _loaded is not None:
        st.session_state.profile = _loaded


REQUIRED_PROFILE_FIELDS = ("name", "dob", "birth_time", "latitude", "longitude", "utc_offset")


def profile_is_complete(p: dict | None) -> bool:
    """True only if every field the engines need is present. A profile can end up partial
    (e.g. an interrupted flow) — treat that the same as no profile, rather than crashing
    deep in a page that assumes it's complete."""
    return p is not None and all(k in p for k in REQUIRED_PROFILE_FIELDS)


def reading_id_for(period_label: str) -> str:
    """Stable per-profile, per-period ID so repeat draws for the same period are reproducible."""
    p = st.session_state.profile
    raw = f"{p['name']}|{p['dob']}|{period_label}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ensure_engines_computed():
    p = st.session_state.profile
    if not profile_is_complete(p):
        return
    if st.session_state.numerology_profile is None:
        st.session_state.numerology_profile = numerology.full_profile(p["name"], p["dob"])
    if st.session_state.chart is None:
        st.session_state.chart = astrology.compute_chart(
            p["name"], p["dob"], p["birth_time"], p["latitude"], p["longitude"], p["utc_offset"]
        )


# ----------------------------------------------------------------------------
# Sidebar — settings live globally: AI key, model, mode toggle, navigation
# ----------------------------------------------------------------------------
with st.sidebar:
    styling.sidebar_mark()
    st.markdown(
        f"<p class='anupt-caption' style='text-align:center;'>Signed in as "
        f"<b style='color:var(--parchment)'>{st.session_state.username}</b></p>",
        unsafe_allow_html=True,
    )
    if st.button("Log out", key="logout_btn"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()
    st.markdown("<hr class='anupt-divider'>", unsafe_allow_html=True)

    st.markdown("#### Navigation")
    nav_options = ["Home", "My Life", "Tarot", "Palmistry", "AI Astrologer", "My Data", "Profile"]
    st.session_state.nav = st.radio(
        "Go to", nav_options,
        index=nav_options.index(st.session_state.nav) if st.session_state.nav in nav_options else 0,
        label_visibility="collapsed",
    )
    st.markdown("<hr class='anupt-divider'>", unsafe_allow_html=True)

    st.markdown("#### Reading Mode")
    st.session_state.mode = st.radio(
        "Mode", ["Combined (Unified)", "Single Engine"],
        index=0 if st.session_state.mode == "Combined (Unified)" else 1,
        label_visibility="collapsed",
        help="Combined fuses all four systems into one AI-synthesized reading. "
             "Single Engine shows one system's reading on its own, with no cross-referencing.",
    )
    single_engine_choice = None
    if st.session_state.mode == "Single Engine":
        single_engine_choice = st.selectbox(
            "Which engine?", ["Astrology", "Numerology", "Tarot", "Palmistry"]
        )
    st.session_state["single_engine_choice"] = single_engine_choice

    st.markdown("<hr class='anupt-divider'>", unsafe_allow_html=True)
    st.markdown("#### AI Provider")
    st.session_state.gemini_key = st.text_input(
        "Gemini API key", value=st.session_state.gemini_key, type="password",
        help="Stored only in this browser session — never written to disk. "
             "Get a key at aistudio.google.com/apikey.",
    )
    st.session_state.gemini_model = st.text_input(
        "Model", value=st.session_state.gemini_model,
        help="Gemini model id. Default works as of this build; change it if Google "
             "retires the model — check ai.google.dev/gemini-api/docs/models.",
    )
    st.markdown(
        "<div class='anupt-disclaimer'>Readings are spiritual / personal-reflection "
        "guidance, not guaranteed factual outcomes. Not a substitute for professional "
        "medical, legal or financial advice.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Onboarding gate
# ----------------------------------------------------------------------------
if not profile_is_complete(st.session_state.profile) and st.session_state.nav not in ("Profile", "My Data"):
    st.session_state.nav = "Profile"

styling.hero()

TOPIC_KEYWORDS = {
    "career": ["career", "job", "work", "promotion", "business"],
    "finance": ["money", "finance", "wealth", "income", "invest"],
    "relationships": ["love", "relationship", "partner", "marriage", "dating"],
    "family": ["family", "parents", "children", "home"],
    "personal_growth": ["growth", "purpose", "creativity", "learning"],
    "spirituality": ["spiritual", "spirituality", "meaning", "soul"],
}


def route_topic(question: str) -> str | None:
    q = question.lower()
    for theme, kws in TOPIC_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return theme
    return None


def evidence_for_theme(theme: str | None) -> dict:
    """Topic-specific evidence selection — never send the whole profile to the model."""
    chart = st.session_state.chart
    num = st.session_state.numerology_profile
    if theme is None:
        u = unified.synthesize(num, chart, st.session_state.last_tarot_draw or [])
        return {"top_themes": u["ranked_themes"][:3], "theme_scores": u["theme_scores"]}

    relevant_houses = [h for h, t in unified.HOUSE_THEME_MAP.items() if t == theme]
    relevant_planets = {
        name: data for name, data in chart["planets"].items() if data["house"] in relevant_houses
    }
    return {
        "topic": theme,
        "dasha": chart["dasha"],
        "relevant_houses": relevant_houses,
        "relevant_planets": relevant_planets,
        "numerology_life_path": num["life_path"],
        "numerology_personal_year": num["personal_year"],
    }


# ============================================================================
# PAGE: PROFILE / ONBOARDING
# ============================================================================
if st.session_state.nav == "Profile":
    st.subheader("Your Profile")
    p = st.session_state.profile or {}
    geo = st.session_state.get("geocode_prefill")  # holds a pending auto-locate result, if any

    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full name", value=p.get("name", ""))
            dob = st.date_input("Date of birth", value=p.get("dob", date(1995, 1, 1)),
                                 min_value=date(1900, 1, 1), max_value=date.today())
            birth_time = st.time_input("Exact birth time", value=p.get("birth_time", time(12, 0)))
        with c2:
            city = st.text_input("Birth city", value=(geo or p).get("city", p.get("city", "")),
                                  placeholder="e.g. Mumbai, India")
            lat = st.number_input("Latitude", value=(geo or p).get("latitude", 19.0760), format="%.4f")
            lon = st.number_input("Longitude", value=(geo or p).get("longitude", 72.8777), format="%.4f")
            utc_offset = st.number_input(
                "UTC offset at birth (hours)", value=(geo or p).get("utc_offset", 5.5), step=0.5, format="%.1f",
                help="e.g. India Standard Time = 5.5, US Eastern (standard) = -5",
            )

        st.markdown("**Try auto-locate** (needs internet where this app is running):")
        auto_col1, auto_col2 = st.columns([3, 1])
        with auto_col2:
            geocode_clicked = st.form_submit_button("📍 Resolve city")

        interests = st.multiselect(
            "Areas of interest",
            ["Career", "Money", "Love", "Marriage", "Family", "Personal growth", "Spirituality"],
            default=p.get("interests", ["Career", "Love"]),
        )

        saved = st.form_submit_button("Save profile & generate reading engines", type="primary")

    if geocode_clicked and city:
        try:
            from geopy.geocoders import Nominatim
            from timezonefinder import TimezoneFinder

            geolocator = Nominatim(user_agent="anupt-app")
            loc = geolocator.geocode(city, timeout=8)
            if loc:
                tf = TimezoneFinder()
                tz_name = tf.timezone_at(lat=loc.latitude, lng=loc.longitude)
                tzinfo = pytz.timezone(tz_name)
                offset_seconds = tzinfo.utcoffset(datetime.combine(dob, birth_time)).total_seconds()
                st.success(
                    f"Found **{loc.address}** — lat {loc.latitude:.4f}, lon {loc.longitude:.4f}, "
                    f"timezone {tz_name} (UTC{offset_seconds/3600:+.1f}). "
                    f"Values filled in below — click Save to confirm."
                )
                # Stored separately from st.session_state.profile — a geocode result is not
                # a saved profile, and writing it there directly (as this used to) produced a
                # partial dict missing name/dob/birth_time for first-time users, which crashed
                # every other page with a KeyError the moment they navigated away.
                st.session_state.geocode_prefill = {
                    "city": city, "latitude": loc.latitude, "longitude": loc.longitude,
                    "utc_offset": offset_seconds / 3600,
                }
                st.rerun()
            else:
                st.warning("Couldn't resolve that city — enter latitude/longitude/UTC offset manually.")
        except Exception as e:
            st.warning(f"Auto-locate unavailable here ({e}) — enter coordinates manually.")

    if saved:
        if not name.strip():
            st.error("Please enter a name.")
        else:
            st.session_state.profile = {
                "name": name.strip(), "dob": dob, "birth_time": birth_time, "city": city,
                "latitude": lat, "longitude": lon, "utc_offset": utc_offset, "interests": interests,
            }
            st.session_state.geocode_prefill = None
            auth_store.save_profile(st.session_state.user_id, st.session_state.profile)
            st.session_state.numerology_profile = None
            st.session_state.chart = None
            ensure_engines_computed()
            st.session_state.nav = "Home"
            st.success("Profile saved to your account — engines computed. Head to Home for your reading.")
            st.rerun()

    with st.expander("Privacy note"):
        st.write(
            "Your birth profile and reading history are saved to this app's own database "
            "under your account — nothing is sent to any other server. You can review or "
            "permanently delete everything from the **My Data** page at any time."
        )

# ============================================================================
# PAGE: MY DATA (account info, reading history, delete) — available even
# before a birth profile is set, since it's about the account itself.
# ============================================================================
elif st.session_state.nav == "My Data":
    st.subheader("My Data")
    created = auth_store.account_created_at(st.session_state.user_id)
    st.markdown(
        f"**Account:** {st.session_state.username}"
        + (f"  ·  member since {created[:10]}" if created else "")
    )

    p = st.session_state.profile
    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Saved birth profile")
    if not profile_is_complete(p):
        st.info("No birth profile saved yet — add one on the **Profile** page.")
    else:
        st.markdown(styling.table(
            ["Field", "Value"],
            [["Name", p["name"]], ["Date of birth", p["dob"].isoformat()],
             ["Birth time", p["birth_time"].isoformat()], ["City", p.get("city") or "—"],
             ["Latitude", p["latitude"]], ["Longitude", p["longitude"]],
             ["UTC offset", p["utc_offset"]], ["Interests", ", ".join(p.get("interests", [])) or "—"]],
        ), unsafe_allow_html=True)

    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Reading history")
    readings = auth_store.get_readings(st.session_state.user_id)
    if not readings:
        st.info("No readings saved yet — generate one from Home, My Life, Tarot, or Palmistry.")
    else:
        for r in readings:
            label = f"{r['created_at'][:16].replace('T', ' ')} UTC · {r['reading_type']} · {r['mode']}"
            with st.expander(label):
                st.markdown(r["narrative"])

    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
    with st.expander("Delete my account and all data"):
        st.warning(
            "This permanently deletes your account, saved birth profile, and entire reading "
            "history from the database. This can't be undone."
        )
        confirm = st.checkbox("I understand this is permanent")
        if st.button("Delete my account", disabled=not confirm, key="delete_account_btn"):
            auth_store.delete_account(st.session_state.user_id)
            for k, v in defaults.items():
                st.session_state[k] = v
            st.success("Account deleted.")
            st.rerun()

# ============================================================================
# Guard: everything else needs a birth profile
# ============================================================================
elif not profile_is_complete(st.session_state.profile):
    st.info("Head to **Profile** in the sidebar to enter your birth details first.")

else:
    ensure_engines_computed()
    p = st.session_state.profile
    chart = st.session_state.chart
    num = st.session_state.numerology_profile

    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    if st.session_state.nav == "Home":
        wcol, tcol = st.columns([1, 1.15], gap="large")
        with wcol:
            st.markdown(f'<div class="anupt-wheel-wrap">{chart_svg.natal_wheel_svg(chart)}</div>',
                        unsafe_allow_html=True)
        with tcol:
            st.markdown(f"### {p['name']}")
            st.markdown(
                f'<p class="anupt-caption">Ascendant <b style="color:var(--brass-soft)">{chart["ascendant"]["sign"]}</b> '
                f'&nbsp;·&nbsp; Moon in <b style="color:var(--brass-soft)">{chart["moon_sign"]}</b> '
                f'&nbsp;·&nbsp; Sun in <b style="color:var(--brass-soft)">{chart["sun_sign"]}</b> '
                f'&nbsp;·&nbsp; Life Path <b style="color:var(--brass-soft)">{num["life_path"]["value"]}</b></p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="anupt-caption">Currently in the <b style="color:var(--parchment)">'
                f'{chart["dasha"]["mahadasha"]}</b> Mahadasha, <b style="color:var(--parchment)">'
                f'{chart["dasha"]["antardasha"]}</b> Antardasha.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("#### Today's Reading")

            if st.button("Cast today's reading", type="primary"):
                rid = reading_id_for(f"day-{date.today().isoformat()}")
                cards = tarot.draw_spread(rid, "one_card")
                st.session_state.last_tarot_draw = cards

                if st.session_state.mode == "Combined (Unified)":
                    u = unified.synthesize(num, chart, cards)
                    with st.spinner("Synthesizing across all systems..."):
                        narrative = gemini_client.generate_unified_narrative(
                            st.session_state.gemini_model, st.session_state.gemini_key, "Day", u
                        )
                    auth_store.save_reading(st.session_state.user_id, "Day", "Combined", "all", narrative)
                    styling.scroll_panel("Today · Combined reading", narrative)
                    with st.expander("See the evidence trail"):
                        st.json(u)
                else:
                    engine = st.session_state.single_engine_choice
                    data = {"Astrology": chart, "Numerology": num, "Tarot": cards}.get(engine, chart)
                    with st.spinner(f"Reading via {engine} only..."):
                        narrative = gemini_client.generate_single_engine_narrative(
                            st.session_state.gemini_model, st.session_state.gemini_key, engine, data, "Day"
                        )
                    auth_store.save_reading(st.session_state.user_id, "Day", "Single Engine", engine, narrative)
                    styling.scroll_panel(f"Today · {engine} only", narrative)
                    with st.expander("See the raw data"):
                        st.json(data)

    # ========================================================================
    # PAGE: MY LIFE (Life / Year / Month / Week / Day / Question readings)
    # ========================================================================
    elif st.session_state.nav == "My Life":
        st.subheader("Readings")
        reading_type = st.selectbox("Reading type", ["Life", "Year", "Month", "Week", "Day", "Question"])
        question = None
        if reading_type == "Question":
            question = st.text_input("Ask your question", placeholder="Will this be a good time for a career change?")

        tabs = st.tabs(["Astrology", "Numerology", "Generate Reading"])
        with tabs[0]:
            wc, dc = st.columns([1, 1.2], gap="large")
            with wc:
                st.markdown(f'<div class="anupt-wheel-wrap">{chart_svg.natal_wheel_svg(chart, size=380)}</div>',
                            unsafe_allow_html=True)
            with dc:
                st.markdown(
                    f'<p class="anupt-caption">Nakshatra <b style="color:var(--brass-soft)">{chart["nakshatra"]}</b>, '
                    f'pada {chart["nakshatra_pada"]} &nbsp;·&nbsp; Mahadasha '
                    f'<b style="color:var(--brass-soft)">{chart["dasha"]["mahadasha"]}</b> '
                    f'({chart["dasha"]["mahadasha_start_year"]}–{chart["dasha"]["mahadasha_end_year"]}) '
                    f'→ Antardasha <b style="color:var(--brass-soft)">{chart["dasha"]["antardasha"]}</b></p>',
                    unsafe_allow_html=True,
                )
                rows = [[chart_svg.PLANET_GLYPH.get(k, "•") + " " + k, v["sign"], v["house"],
                         "<span class='retro'>Retrograde</span>" if v["retrograde"] else "Direct"]
                        for k, v in chart["planets"].items()]
                st.markdown(styling.table(["Planet", "Sign", "House", "Motion"], rows), unsafe_allow_html=True)
        with tabs[1]:
            items = [(v["value"], k.replace("_", " ").title()) for k, v in num.items()]
            st.markdown(styling.medallion_grid(items), unsafe_allow_html=True)
            with st.expander("What each number means"):
                rows = [[k.replace("_", " ").title(), v["value"], v["meaning"]] for k, v in num.items()]
                st.markdown(styling.table(["Number", "Value", "Meaning"], rows), unsafe_allow_html=True)
        with tabs[2]:
            if reading_type == "Question" and not question:
                st.info("Type your question above first.")
            elif st.button("Generate reading", type="primary", key="mylife_generate"):
                rid = reading_id_for(f"{reading_type}-{question or ''}-{date.today().isocalendar()}")
                cards = tarot.draw_spread(rid, "three_card")
                st.session_state.last_tarot_draw = cards

                if st.session_state.mode == "Combined (Unified)":
                    u = unified.synthesize(num, chart, cards)
                    with st.spinner("Synthesizing across all systems..."):
                        narrative = gemini_client.generate_unified_narrative(
                            st.session_state.gemini_model, st.session_state.gemini_key,
                            reading_type, u, question,
                        )
                    auth_store.save_reading(st.session_state.user_id, reading_type, "Combined", "all", narrative)
                    st.markdown("##### Where the systems agree")
                    meter_rows = "".join(
                        styling.strength_meter(
                            t.replace("_", " ").title(),
                            u["theme_scores"][t]["strength"],
                            u["theme_scores"][t]["supporting_systems"],
                        )
                        for t in u["ranked_themes"][:4]
                    )
                    st.markdown(meter_rows, unsafe_allow_html=True)
                    styling.scroll_panel(f"{reading_type} · Combined reading", narrative)
                    with st.expander("See the evidence trail"):
                        st.json(u)
                else:
                    engine = st.session_state.single_engine_choice
                    data = {"Astrology": chart, "Numerology": num, "Tarot": cards}.get(engine, chart)
                    with st.spinner(f"Reading via {engine} only..."):
                        narrative = gemini_client.generate_single_engine_narrative(
                            st.session_state.gemini_model, st.session_state.gemini_key,
                            engine, data, reading_type, question,
                        )
                    auth_store.save_reading(st.session_state.user_id, reading_type, "Single Engine", engine, narrative)
                    styling.scroll_panel(f"{reading_type} · {engine} only", narrative)
                    with st.expander("See the raw data"):
                        st.json(data)

    # ========================================================================
    # PAGE: TAROT
    # ========================================================================
    elif st.session_state.nav == "Tarot":
        st.subheader("Tarot")
        spread_label = st.selectbox("Spread", ["Daily guidance (1 card)", "Past · Present · Future (3 cards)",
                                                 "Five-card spread"])
        spread_key = {"Daily guidance (1 card)": "one_card", "Past · Present · Future (3 cards)": "three_card",
                      "Five-card spread": "five_card"}[spread_label]

        colA, colB = st.columns([1, 1])
        reshuffle = colB.checkbox("Reshuffle (new random draw instead of today's fixed card)")
        if colA.button("Draw", type="primary"):
            rid = (reading_id_for(f"tarot-{spread_key}-{date.today().isoformat()}") if not reshuffle
                   else hashlib.sha256(f"{datetime.now(timezone.utc)}".encode()).hexdigest()[:16])
            cards = tarot.draw_spread(rid, spread_key)
            st.session_state.last_tarot_draw = cards

        if st.session_state.last_tarot_draw:
            cards = st.session_state.last_tarot_draw
            row_html = '<div class="anupt-tarot-row">' + "".join(
                styling.tarot_card_html(c) for c in cards
            ) + '</div>'
            st.markdown(row_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Interpret this spread"):
                if st.session_state.mode == "Combined (Unified)":
                    u = unified.synthesize(num, chart, cards)
                    with st.spinner("Synthesizing..."):
                        narrative = gemini_client.generate_unified_narrative(
                            st.session_state.gemini_model, st.session_state.gemini_key, "Tarot spread", u
                        )
                    auth_store.save_reading(st.session_state.user_id, "Tarot spread", "Combined", "all", narrative)
                    styling.scroll_panel("Tarot · Combined reading", narrative)
                else:
                    with st.spinner("Reading via Tarot only..."):
                        narrative = gemini_client.generate_single_engine_narrative(
                            st.session_state.gemini_model, st.session_state.gemini_key,
                            "Tarot", cards, "Tarot spread",
                        )
                    auth_store.save_reading(st.session_state.user_id, "Tarot spread", "Single Engine", "Tarot", narrative)
                    styling.scroll_panel("Tarot · Single-engine reading", narrative)

        with st.expander("Deck integrity check"):
            st.write(f"78-card completeness check: {'✅ passed' if tarot.deck_completeness_check() else '❌ FAILED'}")

    # ========================================================================
    # PAGE: PALMISTRY
    # ========================================================================
    elif st.session_state.nav == "Palmistry":
        st.subheader("Palmistry")
        st.caption(
            "Line/mount reading is AI-vision-assisted, not a trained deterministic detector — "
            "we're upfront about that rather than fabricating precision we don't have."
        )
        hand = st.radio("Which hand?", ["Left", "Right"], horizontal=True)
        uploaded = st.file_uploader("Upload a clear, well-lit palm photo", type=["jpg", "jpeg", "png"])

        if uploaded:
            img = Image.open(uploaded)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(img, caption=f"{hand} palm", width="stretch")
            with c2:
                quality = palmistry.assess_image(img)
                st.markdown("**Deterministic quality gate**")
                st.json(quality)

            if not quality["passed"]:
                st.warning("Please retake the photo:\n\n" + "\n".join(f"- {i}" for i in quality["issues"]))
            else:
                st.success("Image passed the quality gate.")
                if st.button("Get palm reading"):
                    uploaded.seek(0)
                    img_bytes = uploaded.read()
                    mime = uploaded.type or "image/jpeg"
                    with st.spinner("Reading your palm..."):
                        narrative = gemini_client.palm_vision_reading(
                            st.session_state.gemini_model, st.session_state.gemini_key,
                            img_bytes, mime, hand.lower(),
                        )
                    auth_store.save_reading(
                        st.session_state.user_id, f"Palmistry ({hand})", "AI-assisted", "Palmistry", narrative
                    )
                    styling.scroll_panel(f"Palmistry · {hand} hand · AI-assisted", narrative)

    # ========================================================================
    # PAGE: AI ASTROLOGER (chat)
    # ========================================================================
    elif st.session_state.nav == "AI Astrologer":
        st.subheader("AI Astrologer — Ask Anything")
        st.caption("Questions are routed to the relevant systems first, then answered — "
                   "your whole profile isn't dumped into every message.")

        for turn in st.session_state.chat_history:
            with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                st.write(turn["text"])

        question = st.chat_input("Ask about your career, love life, this year, anything...")
        if question:
            st.session_state.chat_history.append({"role": "user", "text": question})
            with st.chat_message("user"):
                st.write(question)
            theme = route_topic(question)
            context = evidence_for_theme(theme)
            with st.chat_message("assistant"):
                with st.spinner("Consulting the charts..."):
                    reply = gemini_client.chat_reply(
                        st.session_state.gemini_model, st.session_state.gemini_key,
                        st.session_state.chat_history, question, context,
                    )
                st.write(reply)
                if theme:
                    st.caption(f"Routed via: {theme.replace('_', ' ')}")
            st.session_state.chat_history.append({"role": "model", "text": reply})

