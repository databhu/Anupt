"""
ANUPT — AI Astrology, Numerology, Palmistry & Tarot life-reading app.

v4: Gemini as the AI writer (backend-configured, never exposed in the
UI), engine pages restructured around a concise question-first summary
with full details tucked behind an expander, a dedicated ANUPT combined-
insights section folding in Palmistry when available, persistent palm
photo storage, and mobile-responsive refinements. Engines and auth are
unchanged from earlier phases — only the UI wiring and AI provider
changed here.

Run: streamlit run app.py
"""

import hashlib
from datetime import date, time, datetime, timezone

import streamlit as st
from PIL import Image

from engines import numerology, astrology, tarot, palmistry, unified
from ai import gemini_client
from utils import styling, chart_svg, geocoding
from auth import store as auth_store

st.set_page_config(page_title="ANUPT", page_icon="assets/logo_mark.png", layout="centered",
                    initial_sidebar_state="collapsed")
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
    "chat_history": [],
    "last_tarot_draw": None,
    "geocode_prefill": None,
    "selected_birth_place": None,   # {label, name, admin1, country, latitude, longitude, timezone}
    "city_query_cache": {"query": "", "results": []},
    "astro_last": None,       # (summary, details) tuple from the last Astrology reading
    "num_last": None,
    "tarot_last": None,
    "palm_last": None,        # {"left": (summary, details) | None, "right": ...}
    "anupt_last": None,       # (summary, details, unified_evidence) from the last ANUPT combined reading
    "nav": "Home",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

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


def save_reading_safely(reading_type: str, mode: str, engine: str, narrative: str):
    """Reading history is a nice-to-have — never let a DB hiccup break the reading the
    person just paid an AI call for."""
    try:
        auth_store.save_reading(st.session_state.user_id, reading_type, mode, engine, narrative)
    except Exception:
        pass


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

styling.top_bar(who=st.session_state.username)

if not profile_is_complete(st.session_state.profile) and st.session_state.nav != "Profile":
    st.session_state.nav = "Profile"

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


def render_city_picker():
    """
    Live city search-as-you-type: types into a plain (non-form) text_input so
    every keystroke reruns the script; results come from Open-Meteo (free,
    keyless) and are cached by query text so an unrelated rerun doesn't
    re-hit the network for the same text. Selecting a suggestion is the ONLY
    way to set latitude/longitude/timezone now — there is no manual
    coordinate entry anymore.
    """
    picked = st.session_state.selected_birth_place
    if picked:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"📍 **{picked['label']}**")
        if c2.button("Change", key="change_city"):
            st.session_state.selected_birth_place = None
            st.rerun()
        return

    query = st.text_input(
        "Place of birth", placeholder="Type a city, then press Enter (e.g. Mumbai)",
        key="city_query_input",
    )
    if len(query.strip()) < 2:
        if query.strip():
            st.caption("Keep typing — need at least 2 characters, then press Enter.")
        return

    cache = st.session_state.city_query_cache
    if cache["query"] == query:
        results = cache["results"]
    else:
        with st.spinner("Searching cities..."):
            results = geocoding.search_cities(query)
        st.session_state.city_query_cache = {"query": query, "results": results}

    if not results:
        st.caption("No matching cities yet — try a different spelling, or keep typing.")
        return

    st.caption("Select your city:")
    for r in results:
        if st.button(r["label"], key=f"cityopt_{r['label']}", use_container_width=True):
            st.session_state.selected_birth_place = r
            st.rerun()


def render_birth_details_form(existing: dict | None):
    """Name / DOB / time / city-autocomplete only — no manual coordinates, per the
    simplified onboarding flow. Plain widgets (not st.form) so the city search can
    live-update on every keystroke."""
    p = existing or {}

    name = st.text_input("Full name", value=p.get("name", ""), key="pf_name")
    c1, c2 = st.columns(2)
    with c1:
        dob = st.date_input("Date of birth", value=p.get("dob", date(1995, 1, 1)),
                             min_value=date(1900, 1, 1), max_value=date.today(), key="pf_dob")
    with c2:
        birth_time = st.time_input("Time of birth", value=p.get("birth_time", time(12, 0)), key="pf_time")

    # Pre-seed the picker from an existing saved profile the first time this renders,
    # so editing an already-complete profile doesn't force a re-search.
    if st.session_state.selected_birth_place is None and existing and existing.get("city"):
        st.session_state.selected_birth_place = {
            "label": existing["city"], "name": existing["city"], "admin1": "", "country": "",
            "latitude": existing["latitude"], "longitude": existing["longitude"],
            "timezone": None,  # unknown for a previously-saved profile; offset below falls back safely
        }

    render_city_picker()

    interests = st.multiselect(
        "Areas of interest",
        ["Career", "Money", "Love", "Marriage", "Family", "Personal growth", "Spirituality"],
        default=p.get("interests", ["Career", "Love"]),
        key="pf_interests",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Save profile", type="primary", key="pf_save"):
        place = st.session_state.selected_birth_place
        if not name.strip():
            st.error("Please enter your name.")
        elif not place:
            st.error("Please select your birth city from the suggestions.")
        else:
            if place.get("timezone"):
                utc_offset = geocoding.utc_offset_for(place["timezone"], dob, birth_time)
            else:
                # Editing an existing profile without re-picking a city keeps its saved offset.
                utc_offset = p.get("utc_offset", 0.0)
            with st.spinner("Saving your profile..."):
                new_profile = {
                    "name": name.strip(), "dob": dob, "birth_time": birth_time,
                    "city": place["label"], "latitude": place["latitude"], "longitude": place["longitude"],
                    "utc_offset": utc_offset, "interests": interests,
                }
                st.session_state.profile = new_profile
                st.session_state.geocode_prefill = None
                auth_store.save_profile(st.session_state.user_id, new_profile)
                st.session_state.numerology_profile = None
                st.session_state.chart = None
                ensure_engines_computed()
            st.session_state.nav = "Home"
            st.success("Profile saved — your reading is ready.")
            st.rerun()


# ============================================================================
# PAGE: PROFILE / SETTINGS (birth details, account, reading history, logout)
# ============================================================================
if st.session_state.nav == "Profile":
    p = st.session_state.profile

    if not profile_is_complete(p):
        st.subheader("Let's set up your profile")
        st.caption("Just your name, birth date, time, and city — we'll work out the rest.")
        render_birth_details_form(p)
    else:
        st.subheader("Profile & Settings")

        with st.expander("Birth details", expanded=False):
            render_birth_details_form(p)

        st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Account")
        created = auth_store.account_created_at(st.session_state.user_id)
        st.markdown(
            f"**{st.session_state.username}**"
            + (f" &nbsp;·&nbsp; member since {created[:10]}" if created else ""),
            unsafe_allow_html=True,
        )
        if st.button("Log out", key="logout_btn"):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

        st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Reading history")
        readings = auth_store.get_readings(st.session_state.user_id)
        if not readings:
            st.info("No readings saved yet — generate one from Astrology, Numerology, Palmistry, Tarot, or ANUPT.")
        else:
            for r in readings:
                label = f"{r['created_at'][:16].replace('T', ' ')} UTC · {r['reading_type']} · {r['mode']}"
                with st.expander(label):
                    st.markdown(r["narrative"])

        st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
        with st.expander("Delete my account and all data"):
            st.warning(
                "This permanently deletes your account, saved birth profile, and entire reading "
                "history. This can't be undone."
            )
            confirm = st.checkbox("I understand this is permanent")
            if st.button("Delete my account", disabled=not confirm, key="delete_account_btn"):
                auth_store.delete_account(st.session_state.user_id)
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.success("Account deleted.")
                st.rerun()

    st.markdown(
        "<p class='anupt-disclaimer'>Readings are spiritual / personal-reflection guidance, not "
        "guaranteed factual outcomes. Not a substitute for professional medical, legal or "
        "financial advice.</p>",
        unsafe_allow_html=True,
    )

# ============================================================================
# Guard: every other page needs a complete birth profile
# ============================================================================
elif not profile_is_complete(st.session_state.profile):
    st.info("Head to **Profile** to enter your birth details first.")

else:
    ensure_engines_computed()
    p = st.session_state.profile
    chart = st.session_state.chart
    num = st.session_state.numerology_profile

    READING_TYPES = ["Life", "Year", "Month", "Week", "Day", "Question"]

    def reading_type_and_question(key_prefix: str):
        reading_type = st.selectbox("Reading type", READING_TYPES, key=f"{key_prefix}_type")
        question = None
        if reading_type == "Question":
            question = st.text_input("Your question", key=f"{key_prefix}_q",
                                      placeholder="Will this be a good time for a career change?")
        return reading_type, question

    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    if st.session_state.nav == "Home":
        styling.hero()
        wcol, tcol = st.columns([1, 1.15], gap="large")
        with wcol:
            st.markdown(f'<div class="anupt-wheel-wrap">{chart_svg.natal_wheel_svg(chart)}</div>',
                        unsafe_allow_html=True)
        with tcol:
            st.markdown(f"### {p['name']}")
            st.markdown(
                f'<p class="anupt-caption">Ascendant <b style="color:var(--indigo)">{chart["ascendant"]["sign"]}</b> '
                f'&nbsp;·&nbsp; Moon in <b style="color:var(--indigo)">{chart["moon_sign"]}</b> '
                f'&nbsp;·&nbsp; Sun in <b style="color:var(--indigo)">{chart["sun_sign"]}</b> '
                f'&nbsp;·&nbsp; Life Path <b style="color:var(--indigo)">{num["life_path"]["value"]}</b></p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="anupt-caption">Currently in the <b style="color:var(--ink)">'
                f'{chart["dasha"]["mahadasha"]}</b> Mahadasha, <b style="color:var(--ink)">'
                f'{chart["dasha"]["antardasha"]}</b> Antardasha.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("#### Today's Reading")

            if st.button("Cast today's reading", type="primary"):
                rid = reading_id_for(f"day-{date.today().isoformat()}")
                cards = tarot.draw_spread(rid, "one_card")
                st.session_state.last_tarot_draw = cards
                u = unified.synthesize(num, chart, cards)
                with st.spinner("Synthesizing across all systems..."):
                    summary, details = gemini_client.generate_unified_reading("Day", u)
                save_reading_safely("Day", "Combined", "all", f"{summary}\n\n{details}")
                styling.summary_card("Today · ANUPT", summary)
                with st.expander("See the full reading & evidence trail"):
                    styling.scroll_panel("Full ANUPT reading", details)
                    st.json(u)

    # ========================================================================
    # PAGE: ASTROLOGY
    # ========================================================================
    elif st.session_state.nav == "Astrology":
        st.subheader("Astrology")
        st.caption("Ask something, or just get a general read — the chart is below if you want to dig in.")
        reading_type, question = reading_type_and_question("astro")

        if reading_type == "Question" and not question:
            st.info("Type your question above first.")
        elif st.button("Get my Astrology summary", type="primary", key="astro_generate"):
            with st.spinner("Reading your chart..."):
                summary, details = gemini_client.generate_engine_reading(
                    "Astrology", chart, reading_type, question
                )
            save_reading_safely(reading_type, "Single Engine", "Astrology", f"{summary}\n\n{details}")
            st.session_state.astro_last = (summary, details)

        if st.session_state.get("astro_last"):
            summary, details = st.session_state.astro_last
            styling.summary_card(f"{reading_type} · Astrology", summary)
            with st.expander("Explore the full reading"):
                styling.scroll_panel("Full Astrology reading", details)

        with st.expander("Explore your chart"):
            st.markdown(f'<div class="anupt-wheel-wrap">{chart_svg.natal_wheel_svg(chart, size=380)}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<p class="anupt-caption">Nakshatra <b style="color:var(--indigo)">{chart["nakshatra"]}</b>, '
                f'pada {chart["nakshatra_pada"]} &nbsp;·&nbsp; Mahadasha '
                f'<b style="color:var(--indigo)">{chart["dasha"]["mahadasha"]}</b> '
                f'({chart["dasha"]["mahadasha_start_year"]}–{chart["dasha"]["mahadasha_end_year"]}) '
                f'→ Antardasha <b style="color:var(--indigo)">{chart["dasha"]["antardasha"]}</b></p>',
                unsafe_allow_html=True,
            )
            rows = [[chart_svg.PLANET_GLYPH.get(k, "•") + " " + k, v["sign"], v["house"],
                     "<span class='retro'>Retrograde</span>" if v["retrograde"] else "Direct"]
                    for k, v in chart["planets"].items()]
            st.markdown(styling.table(["Planet", "Sign", "House", "Motion"], rows), unsafe_allow_html=True)

    # ========================================================================
    # PAGE: NUMEROLOGY
    # ========================================================================
    elif st.session_state.nav == "Numerology":
        st.subheader("Numerology")
        st.caption("Ask something, or just get a general read — your numbers are below if you want to dig in.")
        reading_type, question = reading_type_and_question("num")

        if reading_type == "Question" and not question:
            st.info("Type your question above first.")
        elif st.button("Get my Numerology summary", type="primary", key="num_generate"):
            with st.spinner("Reading your numbers..."):
                summary, details = gemini_client.generate_engine_reading(
                    "Numerology", num, reading_type, question
                )
            save_reading_safely(reading_type, "Single Engine", "Numerology", f"{summary}\n\n{details}")
            st.session_state.num_last = (summary, details)

        if st.session_state.num_last:
            summary, details = st.session_state.num_last
            styling.summary_card(f"{reading_type} · Numerology", summary)
            with st.expander("Explore the full reading"):
                styling.scroll_panel("Full Numerology reading", details)

        with st.expander("Explore your numbers"):
            items = [(v["value"], k.replace("_", " ").title()) for k, v in num.items()]
            st.markdown(styling.medallion_grid(items), unsafe_allow_html=True)
            with st.expander("What each number means"):
                rows = [[k.replace("_", " ").title(), v["value"], v["meaning"]] for k, v in num.items()]
                st.markdown(styling.table(["Number", "Value", "Meaning"], rows), unsafe_allow_html=True)

    # ========================================================================
    # PAGE: PALMISTRY
    # ========================================================================
    elif st.session_state.nav == "Palmistry":
        st.subheader("Palmistry")
        st.caption(
            "Line/mount reading is AI-vision-assisted, not a trained deterministic detector — "
            "we're upfront about that rather than fabricating precision we don't have."
        )
        styling.tips_box("Tips for a reading-friendly photo", [
            "Good, even light — daylight near a window works best; avoid harsh shadows.",
            "Hold your hand flat and open, fingers slightly spread, palm facing the camera.",
            "Fill most of the frame with your palm, in focus, against a plain background.",
            "Remove rings, watches, or anything covering the lines and mounts.",
        ])

        hand = st.radio("Which hand?", ["Left", "Right"], horizontal=True, key="palm_hand")
        hand_key = hand.lower()
        stored = auth_store.get_palm_photo(st.session_state.user_id, hand_key)

        tab_camera, tab_upload = st.tabs(["📷 Take a photo", "📁 Upload a photo"])
        with tab_camera:
            camera_file = st.camera_input("Capture your palm", key=f"palm_cam_{hand_key}",
                                           label_visibility="collapsed")
        with tab_upload:
            uploaded_file = st.file_uploader("Upload a clear, well-lit palm photo",
                                              type=["jpg", "jpeg", "png"], key=f"palm_up_{hand_key}",
                                              label_visibility="collapsed")
        new_file = camera_file or uploaded_file

        img_bytes = mime = None
        if new_file is not None:
            img = Image.open(new_file)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(img, caption=f"{hand} palm (new)", width="stretch")
            with c2:
                with st.spinner("Checking image quality..."):
                    quality = palmistry.assess_image(img)
                st.markdown("**Quality gate**")
                st.json(quality)

            if not quality["passed"]:
                st.warning("Please retake the photo:\n\n" + "\n".join(f"- {i}" for i in quality["issues"]))
            else:
                st.success("Image passed the quality gate.")
                new_file.seek(0)
                img_bytes = new_file.read()
                mime = new_file.type or "image/jpeg"
                if st.button("Save this photo to my profile", key=f"palm_save_{hand_key}"):
                    auth_store.save_palm_photo(st.session_state.user_id, hand_key, img_bytes, mime)
                    st.success(f"Saved — this will be your {hand.lower()} palm photo until you replace it.")
                    st.rerun()
        elif stored is not None:
            st.image(stored["image_bytes"], caption=f"{hand} palm (saved to your profile)", width="stretch")
            img_bytes, mime = stored["image_bytes"], stored["mime_type"]
        else:
            st.info("Take or upload a palm photo above to get a reading.")

        if img_bytes is not None:
            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            palm_question = st.text_input(
                "What would you like to know?", key=f"palm_q_{hand_key}",
                placeholder="e.g. What does my life line say?",
            )
            if st.button("Get my Palmistry summary", type="primary", key=f"palm_generate_{hand_key}"):
                with st.spinner("Reading your palm..."):
                    summary, details = gemini_client.palm_vision_reading(
                        img_bytes, mime, hand_key, palm_question or None
                    )
                save_reading_safely(f"Palmistry ({hand})", "AI-assisted", "Palmistry", f"{summary}\n\n{details}")
                palm_last = dict(st.session_state.palm_last or {})
                palm_last[hand_key] = (summary, details)
                st.session_state.palm_last = palm_last

        current = (st.session_state.palm_last or {}).get(hand_key)
        if current:
            summary, details = current
            styling.summary_card(f"Palmistry · {hand} hand", summary)
            with st.expander("Explore the full reading"):
                styling.scroll_panel(f"Full Palmistry reading · {hand} hand", details)

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
            st.session_state.tarot_last = None

        if st.session_state.last_tarot_draw:
            cards = st.session_state.last_tarot_draw
            row_html = '<div class="anupt-tarot-row">' + "".join(
                styling.tarot_card_html(c) for c in cards
            ) + '</div>'
            st.markdown(row_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            tarot_question = st.text_input(
                "What would you like to know?", key="tarot_q",
                placeholder="e.g. What should I focus on this week?",
            )
            if st.button("Get my Tarot summary", type="primary"):
                with st.spinner("Reading the cards..."):
                    summary, details = gemini_client.generate_engine_reading(
                        "Tarot", cards, "Tarot spread", tarot_question or None
                    )
                save_reading_safely("Tarot spread", "Single Engine", "Tarot", f"{summary}\n\n{details}")
                st.session_state.tarot_last = (summary, details)

            if st.session_state.tarot_last:
                summary, details = st.session_state.tarot_last
                styling.summary_card("Tarot", summary)
                with st.expander("Explore the full reading"):
                    styling.scroll_panel("Full Tarot reading", details)

        with st.expander("Deck integrity check"):
            st.write(f"78-card completeness check: {'✅ passed' if tarot.deck_completeness_check() else '❌ FAILED'}")

    # ========================================================================
    # PAGE: ANUPT (combined insights across every engine + AI Astrologer chat)
    # ========================================================================
    elif st.session_state.nav == "ANUPT":
        st.subheader("ANUPT")
        st.caption("Every engine, fused into one voice — where Astrology, Numerology, "
                   "Tarot and (if you've done one) your Palmistry reading agree.")
        tab_generate, tab_chat = st.tabs(["Get My Insights", "Ask AI Astrologer"])

        with tab_generate:
            reading_type, question = reading_type_and_question("anupt")

            palm_readings = st.session_state.palm_last or {}
            palm_summary_text = " / ".join(
                f"{h.title()} hand: {s}" for h, (s, _d) in palm_readings.items() if s
            ) or None
            if palm_summary_text:
                st.caption("✓ Including your saved Palmistry reading in this synthesis.")

            if reading_type == "Question" and not question:
                st.info("Type your question above first.")
            elif st.button("Get my ANUPT insights", type="primary", key="anupt_generate"):
                rid = reading_id_for(f"{reading_type}-{question or ''}-{date.today().isocalendar()}")
                cards = tarot.draw_spread(rid, "three_card")
                st.session_state.last_tarot_draw = cards
                u = unified.synthesize(num, chart, cards, palm_summary_text)
                with st.spinner("Synthesizing across all systems..."):
                    summary, details = gemini_client.generate_unified_reading(reading_type, u, question)
                save_reading_safely(reading_type, "Combined", "all", f"{summary}\n\n{details}")
                st.session_state.anupt_last = (summary, details, u)

            if st.session_state.anupt_last:
                summary, details, u = st.session_state.anupt_last
                styling.summary_card(f"{reading_type} · ANUPT", summary)

                st.markdown("##### Where the systems agree")
                cards_for_grid = [
                    (t.replace("_", " ").title(), u["theme_scores"][t]["strength"],
                     u["theme_scores"][t]["supporting_systems"])
                    for t in u["ranked_themes"][:4]
                ]
                styling.insight_grid(cards_for_grid)

                with st.expander("Explore the full reading"):
                    styling.scroll_panel("Full ANUPT reading", details)
                with st.expander("See the evidence trail"):
                    st.json(u)

        with tab_chat:
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
                        reply = gemini_client.chat_reply(st.session_state.chat_history, question, context)
                    st.write(reply)
                    if theme:
                        st.caption(f"Routed via: {theme.replace('_', ' ')}")
                st.session_state.chat_history.append({"role": "model", "text": reply})

# ----------------------------------------------------------------------------
# Bottom navigation — rendered last; CSS pins it to the viewport bottom
# regardless of where it appears in the DOM. Each button updates
# st.session_state.nav itself via on_click, so no return-value handling
# or manual rerun is needed here.
# ----------------------------------------------------------------------------
styling.bottom_nav(active=st.session_state.nav)
