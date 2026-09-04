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

from engines import numerology, numerology_scoring, numerology_interpretation, astrology, astrology_scoring, astrology_interpretation, tarot, tarot_interpretation, palmistry, unified, youtube_client
from ai import gemini_client
from utils import styling, chart_svg, geocoding, image_processing, palm_annotation, nav_router
from auth import store as auth_store

st.set_page_config(page_title="ANUPT", page_icon="assets/logo_mark.png", layout="wide",
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
    "astro_last": None,       # (score, summary, details) from the last Astrology reading
    "num_last": None,
    "num_special_last": None,  # (focus_area, score, summary, details) from the last specialized Numerology reading
    "astro_special_last": None,  # (focus_area, score, summary, details) from the last specialized Astrology reading
    "tarot_last": None,
    "palm_last": None,        # {"left": structured_reading_dict | None, "right": ...} — see
                               # ai.gemini_client.palm_vision_reading_structured()'s return shape
    "palm_comparison_last": None,
    "nav_chat_history": [],       # [{"role": "user"/"assistant", "text": str, "action": str|None, "options": list|None}]
    "nav_chat_last_destination": None,  # for resolving follow-ups like "tell me more"
    "yt_insights_last": None,
    "anupt_last": None,       # (score, summary, details, unified_evidence) from the last ANUPT reading
    "force_edit_profile_open": False,  # set True to auto-expand the Edit Profile section on next Profile visit
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
        st.session_state.numerology_profile = numerology.full_profile(
            p["name"], p["dob"], system=p.get("numerology_system", "pythagorean")
        )
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


def run_youtube_insights_pipeline(zodiac_sign: str, time_period: str = "week",
                                   question: str | None = None, language: str = "en",
                                   force_refresh: bool = False) -> dict:
    """Orchestrates the full YouTube Insights pipeline — Research ->
    Extraction -> Analysis -> Zodiac -> Evidence -> AI Summary — with a
    24-hour cache in front of the expensive part (Research + Extraction:
    one YouTube API call plus one AI call) so repeat visits for the same
    sign on the same day reuse prior results instead of re-fetching and
    re-extracting. The Summary stage always runs fresh, since it can be
    personalized to a specific question the cached evidence can't predict.

    Lives here (not in engines/) because it's genuinely cross-layer
    orchestration — YouTube API, AI extraction, deterministic aggregation,
    cache, AI summary — the same role app.py's ensure_engines_computed()
    and save_reading_safely() already play for other cross-cutting
    concerns, keeping engines/ itself limited to pure calculation."""
    from engines import youtube_client, youtube_insights

    cache_key = f"youtube_insights:{zodiac_sign}:{time_period}"
    cached = None if force_refresh else auth_store.get_cached(cache_key, max_age_hours=24)

    if cached:
        evidence = cached
    else:
        search_result = youtube_client.search_astrology_videos(zodiac_sign, time_period)
        if search_result["error"]:
            return {"error": search_result["error"]}
        videos = search_result["videos"]
        if not videos:
            return {"error": f"No recent astrology videos found for {zodiac_sign} right now."}

        extracted = gemini_client.extract_youtube_predictions(videos)
        evidence = youtube_insights.build_insights_evidence(extracted, videos, zodiac_sign)
        auth_store.set_cached(cache_key, evidence)

    if evidence["relevant_prediction_count"] == 0:
        return {"error": f"Found videos, but couldn't extract clear {zodiac_sign} predictions from them "
                         "right now — please try again shortly.", "evidence": evidence}

    score, summary, details = gemini_client.generate_youtube_summary(evidence, zodiac_sign, question, language)
    return {"error": None, "evidence": evidence, "score": score, "summary": summary, "details": details}


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

    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
    st.markdown("##### Numerology settings")
    current_name = st.text_input(
        "Current name (if different from your full name above)",
        value=p.get("current_name", ""), key="pf_current_name",
        placeholder="Leave blank if it's the same — e.g. a married or chosen name",
        help="Lets the Numerology page compare your birth-name vibration to your current one.",
    )
    system_choice = st.radio(
        "Numerology system", ["Pythagorean", "Chaldean"],
        index=0 if p.get("numerology_system", "pythagorean") != "chaldean" else 1,
        key="pf_numerology_system", horizontal=True,
        help="Two traditions with different letter-value tables for name-based numbers "
             "(Life Path and other date-based numbers are identical either way). "
             "Pythagorean is the more common Western system; Chaldean is the older one.",
    )
    language_choice = st.radio(
        "Reading language", ["English", "मराठी (Marathi)"],
        index=0 if p.get("language", "en") != "mr" else 1,
        key="pf_language", horizontal=True,
        help="Applies to your Numerology and Astrology readings.",
    )

    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
    st.markdown("##### Astrology settings")
    house_system_choice = st.selectbox(
        "Western house system", list(astrology.WESTERN_HOUSE_SYSTEMS.keys()),
        index=list(astrology.WESTERN_HOUSE_SYSTEMS.keys()).index(p.get("house_system", "Placidus"))
        if p.get("house_system", "Placidus") in astrology.WESTERN_HOUSE_SYSTEMS else 0,
        key="pf_house_system",
        help="Only affects the Western Chart tab — the Vedic chart always uses whole-sign "
             "houses, per that tradition's own convention.",
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
                    "current_name": current_name.strip(),
                    "numerology_system": "chaldean" if system_choice == "Chaldean" else "pythagorean",
                    "language": "mr" if "Marathi" in language_choice else "en",
                    "house_system": house_system_choice,
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


def render_palm_capture_section(hand: str):
    """One hand's capture UI: shows the stored photo if there is one (with a
    Retake option), otherwise the guide + camera/upload widgets. This is the
    ONLY place palm photos are captured anywhere in the app — the Palmistry
    engine page just reads what's stored here, it never asks for a fresh
    upload itself."""
    hand_key = hand.lower()
    stored = auth_store.get_palm_photo(st.session_state.user_id, hand_key)
    retake_flag = f"pf_palm_retaking_{hand_key}"

    st.markdown(f"**{hand} hand**")

    if stored is not None and not st.session_state.get(retake_flag):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(stored["image_bytes"], width="stretch")
        with c2:
            st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("Retake", key=f"pf_palm_retake_btn_{hand_key}"):
                st.session_state[retake_flag] = True
                st.rerun()
        return

    styling.palm_guide_card()
    tab_cam, tab_up = st.tabs(["📷 Camera", "📁 Upload"])
    with tab_cam:
        cam_file = st.camera_input("Capture", key=f"pf_palm_cam_{hand_key}", label_visibility="collapsed")
    with tab_up:
        up_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"],
                                    key=f"pf_palm_up_{hand_key}", label_visibility="collapsed")
    new_file = cam_file or up_file

    if new_file is not None:
        with st.spinner("Optimizing image..."):
            optimized = image_processing.optimize_image(Image.open(new_file))
        quality = palmistry.assess_image(optimized)
        if not quality["passed"]:
            st.warning("Please retake:\n\n" + "\n".join(f"- {i}" for i in quality["issues"]))
        else:
            img_bytes = image_processing.to_jpeg_bytes(optimized)
            auth_store.save_palm_photo(st.session_state.user_id, hand_key, img_bytes, "image/jpeg")
            st.session_state[retake_flag] = False
            st.success(f"{hand} palm saved — optimized and ready for a Palmistry reading anytime.")
            st.rerun()
    elif stored is not None:
        # They tapped Retake but haven't captured a replacement yet — offer a way back out.
        if st.button("Keep existing photo", key=f"pf_palm_keep_{hand_key}"):
            st.session_state[retake_flag] = False
            st.rerun()


def render_ask_anupt_section():
    """The 'Ask ANUPT' navigation chatbot: User Request -> Intent Detection
    (utils.nav_router, rule-based) -> Router (this function, sets nav) ->
    Correct Feature -> Reading. AI (gemini_client.classify_navigation_intent)
    only runs when the rule-based router finds no match at all."""
    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
    st.markdown("##### 💬 Ask ANUPT")
    st.caption("Tell me what you're looking for, or tap a quick option — "
              "e.g. \"show my palm reading\" or \"life path number\".")

    chip_definitions = [
        ("🔮 Palm reading", "Show my palm reading"),
        ("🔢 Life Path", "Life path number"),
        ("⭐ Birth chart", "My birth chart"),
        ("🃏 Draw a card", "Draw a tarot card"),
        ("💼 Career", "Career prediction"),
        ("✨ Combined", "Combined reading"),
    ]
    chip_query = None
    with st.container(key="anupt_chip_row"):
        chip_cols = st.columns(3)
        for i, (label, phrase) in enumerate(chip_definitions):
            with chip_cols[i % 3]:
                if st.button(label, key=f"nav_chip_{i}", width="stretch", help=phrase):
                    chip_query = phrase

    typed_query = st.chat_input("Ask ANUPT... e.g. 'Show my palm reading'")
    user_query = chip_query or typed_query

    if user_query:
        st.session_state.nav_chat_history.append({"role": "user", "text": user_query})
        result = nav_router.detect_intent(user_query, st.session_state.nav_chat_last_destination)

        if result["matched"] is True:
            destination = result["destination"]
            st.session_state.nav_chat_last_destination = destination
            st.session_state.nav_chat_history.append({
                "role": "assistant",
                "text": f"That sounds like **{destination}** — {nav_router.DESTINATION_DESCRIPTIONS[destination]}.",
                "action": destination, "options": None,
            })
        elif result["matched"] == "ambiguous":
            st.session_state.nav_chat_history.append({
                "role": "assistant",
                "text": f"A few places can help with '{result['term']}' — which would you like?",
                "action": None, "options": result["candidates"],
            })
        else:
            ai_result = gemini_client.classify_navigation_intent(user_query, nav_router.DESTINATION_DESCRIPTIONS)
            if ai_result["destination"]:
                destination = ai_result["destination"]
                st.session_state.nav_chat_last_destination = destination
                st.session_state.nav_chat_history.append({
                    "role": "assistant",
                    "text": f"I think you're looking for **{destination}** — "
                            f"{nav_router.DESTINATION_DESCRIPTIONS[destination]}.",
                    "action": destination, "options": None,
                })
            else:
                st.session_state.nav_chat_history.append({
                    "role": "assistant",
                    "text": "I'm not sure exactly what you're looking for — here's everything ANUPT offers:",
                    "action": None, "options": nav_router.DESTINATIONS,
                })
        st.rerun()

    for i, turn in enumerate(st.session_state.nav_chat_history[-8:]):
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["text"])
            if turn.get("action"):
                if st.button(f"Open {turn['action']} →", key=f"nav_open_{i}", type="primary"):
                    st.session_state.nav = turn["action"]
                    st.rerun()
            if turn.get("options"):
                opt_cols = st.columns(min(len(turn["options"]), 3))
                for j, opt in enumerate(turn["options"]):
                    with opt_cols[j % len(opt_cols)]:
                        if st.button(opt, key=f"nav_option_{i}_{j}"):
                            st.session_state.nav = opt
                            st.session_state.nav_chat_last_destination = opt
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

        st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### 📸 Palm photos (optional)")
        st.caption(
            "Add these now if you'd like Palmistry readings later — the Palmistry "
            "page will use whatever's saved here automatically, no separate upload there."
        )
        render_palm_capture_section("Left")
        render_palm_capture_section("Right")
    else:
        st.subheader("Profile & Settings")

        with st.expander(
            ":material/edit: Edit Profile — name, birth date/time, city",
            expanded=st.session_state.pop("force_edit_profile_open", False),
        ):
            render_birth_details_form(p)

        with st.expander(":material/back_hand: Palm photos — for Palmistry readings"):
            st.caption(
                "Saved here once, used automatically on the Palmistry page — no need "
                "to upload again there."
            )
            render_palm_capture_section("Left")
            render_palm_capture_section("Right")

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
        with st.expander("AI service status (for the app owner)"):
            st.caption(
                "Checks whether the AI reading service is reachable. Useful if readings "
                "show the 'service isn't available' notice. Your API key is never displayed."
            )
            if st.button("Run check", key="diag_btn"):
                with st.spinner("Checking the AI service..."):
                    st.json(gemini_client.diagnose())

        with st.expander("YouTube Insights key status (for the app owner)"):
            st.caption(
                "Shows which configured YouTube API key(s) are currently available vs. "
                "cooling down after a quota limit, for diagnosing the YouTube Insights tab "
                "on the Astrology page. Keys are always shown masked."
            )
            if st.button("Check YouTube keys", key="yt_diag_btn"):
                status = youtube_client.youtube_key_status()
                if not status:
                    st.info("No YOUTUBE_API_KEY configured yet — see secrets.toml.example.")
                else:
                    st.json(status)

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
        render_ask_anupt_section()
        wcol, tcol = st.columns([1, 1.15], gap="large")
        with wcol:
            st.markdown(f'<div class="anupt-wheel-wrap">{chart_svg.natal_wheel_svg(chart)}</div>',
                        unsafe_allow_html=True)
        with tcol:
            name_col, edit_col = st.columns([4, 1.3])
            with name_col:
                st.markdown(f"### {p['name']}")
            with edit_col:
                st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
                if st.button(":material/edit: Edit", key="home_edit_profile",
                             help="Edit your name, birth date/time, or city"):
                    st.session_state.force_edit_profile_open = True
                    st.session_state.nav = "Profile"
                    st.rerun()
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
                    score, summary, details = gemini_client.generate_unified_reading(
                        "Day", u, language=p.get("language", "en")
                    )
                save_reading_safely("Day", "Combined", "all", f"{summary}\n\n{details}")
                styling.summary_card("Today · ANUPT", summary)
                st.markdown(styling.score_gauge(score, "Today"), unsafe_allow_html=True)
                with st.expander("See the full reading & evidence trail"):
                    styling.scroll_panel("Full ANUPT reading", details)
                    st.json(u)

    # ========================================================================
    # PAGE: ASTROLOGY
    # ========================================================================
    elif st.session_state.nav == "Astrology":
        st.subheader("Astrology")
        st.caption(
            "A traditional symbolic system passed down over centuries, not a scientifically "
            "validated method of prediction — the positions below are precise astronomical "
            "calculations; what they're said to mean is interpretive tradition."
        )

        tab_overview, tab_western, tab_cycles, tab_areas, tab_special, tab_youtube = st.tabs(
            ["Overview", "Western Chart", "Life Cycles", "Life Areas", "Specialized Reading", "YouTube Insights"]
        )

        # -------------------------------------------------------------
        with tab_overview:
            st.caption("Vedic (sidereal) system · whole-sign houses")

            chart_evidence = astrology_interpretation.build_chart_evidence(chart)
            st.markdown(styling.rule_based_badge(), unsafe_allow_html=True)
            st.markdown("##### Key Findings")
            st.caption("Your strongest placements, straight from the rule engine — visible instantly, "
                      "no AI call needed to see this much.")
            for placement in chart_evidence["placements"][:4]:
                styling.key_finding_card(
                    f"{placement['planet']} in {placement['sign']} (House {placement['house']})",
                    placement["strength_label"], placement["strength_tier"],
                    placement["interpretation"], placement["basis"], placement["exceptions_applied"],
                )
            with st.expander("See all placements"):
                for placement in chart_evidence["placements"][4:]:
                    styling.key_finding_card(
                        f"{placement['planet']} in {placement['sign']} (House {placement['house']})",
                        placement["strength_label"], placement["strength_tier"],
                        placement["interpretation"], placement["basis"], placement["exceptions_applied"],
                    )

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### AI-Personalized Reading")
            st.caption("Synthesizes the Key Findings above into an answer to your specific question — "
                      "the AI weaves existing evidence together, it doesn't reinterpret your chart from scratch.")
            reading_type, question = reading_type_and_question("astro")

            if reading_type == "Question" and not question:
                st.info("Type your question above first.")
            elif st.button("Get my Astrology summary", type="primary", key="astro_generate"):
                enriched_chart = dict(chart)
                enriched_chart["rule_based_findings"] = chart_evidence
                with st.spinner("Reading your chart..."):
                    score, summary, details = gemini_client.generate_engine_reading(
                        "Astrology", enriched_chart, reading_type, question, language=p.get("language", "en")
                    )
                save_reading_safely(reading_type, "Single Engine", "Astrology", f"{summary}\n\n{details}")
                st.session_state.astro_last = (score, summary, details)

            if st.session_state.get("astro_last"):
                score, summary, details = st.session_state.astro_last
                styling.summary_card(f"{reading_type} · Astrology", summary)
                st.markdown(styling.score_gauge(score, "This reading"), unsafe_allow_html=True)
                with st.expander("Explore the full reading"):
                    styling.scroll_panel("Full Astrology reading", details)

            with st.expander("Advanced Analysis — full chart"):
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
                strengths = astrology.planetary_strength_summary(chart)
                rows = [[chart_svg.PLANET_GLYPH.get(k, "•") + " " + k, v["sign"],
                         f'{v["degree_in_sign"]}°', v["house"],
                         "<span class='retro'>Retrograde</span>" if v["retrograde"] else "Direct",
                         styling.dignity_badge(strengths.get(k, {}).get("dignity", "Neutral"))]
                        for k, v in chart["planets"].items()]
                st.markdown(styling.table(["Planet", "Sign", "Degree", "House", "Motion", "Dignity"], rows),
                            unsafe_allow_html=True)

                st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                st.markdown("**Yogas found in this chart**")
                st.caption("A small, clearly-defined set of classical combinations this app checks for "
                          "— not an exhaustive catalogue (there are hundreds across different texts).")
                st.markdown(styling.yoga_cards(astrology.detect_yogas(chart)), unsafe_allow_html=True)

        # -------------------------------------------------------------
        with tab_western:
            house_sys = p.get("house_system", "Placidus")
            st.caption(f"Tropical (Western) system · {house_sys} houses — change the house "
                      "system anytime in Profile → Edit Profile → Astrology settings.")
            western = astrology.compute_western_chart(
                p["dob"], p["birth_time"], p["latitude"], p["longitude"], p["utc_offset"],
                house_system=house_sys,
            )
            c1, c2, c3, c4 = st.columns(4)
            for col, label, point in [(c1, "Ascendant", western["ascendant"]), (c2, "Descendant", western["descendant"]),
                                       (c3, "Midheaven (MC)", western["midheaven"]), (c4, "IC", western["ic"])]:
                with col:
                    st.metric(label, f'{point["sign"]} {point["degree_in_sign"]}°')

            with st.expander("Planets (Western)", expanded=True):
                rows = [[chart_svg.PLANET_GLYPH.get(k, "•") + " " + k, v["sign"], f'{v["degree_in_sign"]}°',
                         f'{v["house"]} ({v["house_classification"]})',
                         "<span class='retro'>Retrograde</span>" if v["retrograde"] else "Direct",
                         styling.dignity_badge(v["dignity"])]
                        for k, v in western["planets"].items()]
                st.markdown(styling.table(["Planet", "Sign", "Degree", "House", "Motion", "Dignity"], rows),
                            unsafe_allow_html=True)

            with st.expander("Aspects"):
                st.markdown(styling.aspect_list(western["aspects"]), unsafe_allow_html=True)

        # -------------------------------------------------------------
        with tab_cycles:
            st.markdown("##### Dasha (Vedic ruling periods)")
            st.markdown(
                f'<p class="anupt-caption">Currently in the <b style="color:var(--indigo)">'
                f'{chart["dasha"]["mahadasha"]}</b> Mahadasha '
                f'({chart["dasha"]["mahadasha_start_year"]}–{chart["dasha"]["mahadasha_end_year"]}), '
                f'within the <b style="color:var(--indigo)">{chart["dasha"]["antardasha"]}</b> Antardasha.</p>',
                unsafe_allow_html=True,
            )

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### Current transits")
            st.caption("Where the sky is right now, and which of those positions form an aspect "
                      "back to a placement in your natal chart — the standard 'what's active for "
                      "me currently' technique.")
            transits = astrology.compute_transits(chart, as_of=date.today())
            if transits["aspects_to_natal"]:
                st.markdown(
                    styling.aspect_list(transits["aspects_to_natal"],
                                        point_a_key="transiting_point", point_b_key="natal_point"),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No major transit aspects to your natal chart within the current orbs today.")

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### Secondary progressions")
            st.caption("The 'a day for a year' technique — your progressed chart today reflects "
                      "the day this many days after your birth, read as slow inner development "
                      "rather than external events.")
            prog = astrology.secondary_progressions(
                p["dob"], p["birth_time"], p["latitude"], p["longitude"], p["utc_offset"]
            )
            st.caption(f"Progressed to {prog['progressed_date']} (age {prog['age_years']})")
            prog_rows = [[chart_svg.PLANET_GLYPH.get(k, "•") + " " + k, v["sign"], f'{v["degree_in_sign"]}°',
                          "<span class='retro'>Retrograde</span>" if v["retrograde"] else "Direct"]
                         for k, v in prog["planets"].items()]
            st.markdown(styling.table(["Planet", "Sign", "Degree", "Motion"], prog_rows), unsafe_allow_html=True)

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### Solar return")
            st.caption("The moment each year the Sun returns to your exact birth degree — a chart "
                      "for the year ahead. Traditionally read at your current residence; this uses "
                      "your saved birth location, which may differ if you've since moved.")
            sr_year = st.number_input("Year", min_value=1950, max_value=2100,
                                       value=date.today().year, key="sr_year", step=1)
            sr = astrology.solar_return_chart(chart, p["dob"], int(sr_year), p["latitude"], p["longitude"])
            st.caption(f"Exact moment (UTC): {sr['exact_moment_utc']} · Ascendant "
                      f"{sr['ascendant']['sign']} {sr['ascendant']['degree_in_sign']}°")
            sr_rows = [[chart_svg.PLANET_GLYPH.get(k, "•") + " " + k, v["sign"], f'{v["degree_in_sign"]}°',
                       v["house"], "<span class='retro'>Retrograde</span>" if v["retrograde"] else "Direct"]
                      for k, v in sr["planets"].items()]
            st.markdown(styling.table(["Planet", "Sign", "Degree", "House", "Motion"], sr_rows),
                        unsafe_allow_html=True)

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### Directions (Solar Arc)")
            st.caption("Every natal planet advanced by the same arc the Sun itself has "
                      "progressed since birth — a widely-used modern alternative to the older, "
                      "more methodologically disputed Primary Directions technique.")
            sa = astrology.solar_arc_directions(
                chart, p["dob"], p["birth_time"], p["latitude"], p["longitude"], p["utc_offset"]
            )
            st.caption(f"Current arc: {sa['arc_degrees']}°")
            sa_rows = [[chart_svg.PLANET_GLYPH.get(k, "•") + " " + k, v["sign"], f'{v["degree_in_sign"]}°']
                      for k, v in sa["planets"].items()]
            st.markdown(styling.table(["Planet", "Directed Sign", "Degree"], sa_rows), unsafe_allow_html=True)

        # -------------------------------------------------------------
        with tab_areas:
            st.caption("Deterministic — every score below comes from planetary house placement and "
                      "dignity in your actual chart, never a random or AI-generated figure.")
            life_scores = astrology_scoring.theme_scores(chart)
            for theme in astrology_scoring.THEMES:
                d = life_scores[theme]
                st.markdown(
                    styling.score_bar(theme.replace("_", " ").title(), d["score"], d["band"], d["reason"]),
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------------------
        with tab_special:
            st.caption("A reading focused on one area of life, still built only from your "
                      "calculated chart above.")
            focus_choice = st.selectbox(
                "Focus area", ["Career", "Finance", "Relationships", "Personal Growth"], key="astro_focus"
            )
            special_question = st.text_input(
                "Optional question", key="astro_special_q",
                placeholder=f"e.g. What does my chart say about {focus_choice.lower()}?",
            )
            if st.button(f"Get my {focus_choice} reading", type="primary", key="astro_special_generate"):
                life_scores = astrology_scoring.theme_scores(chart)
                enriched_data = dict(chart)
                enriched_data["life_area_scores"] = life_scores
                enriched_data["yogas"] = astrology.detect_yogas(chart)
                with st.spinner(f"Reading your chart for {focus_choice.lower()}..."):
                    score, summary, details = gemini_client.generate_engine_reading(
                        "Astrology", enriched_data, "Life", special_question or None,
                        focus=focus_choice.lower(), language=p.get("language", "en"),
                    )
                save_reading_safely(f"Astrology · {focus_choice}", "Single Engine", "Astrology",
                                     f"{summary}\n\n{details}")
                st.session_state.astro_special_last = (focus_choice, score, summary, details)

            if st.session_state.get("astro_special_last"):
                f_choice, score, summary, details = st.session_state.astro_special_last
                styling.summary_card(f"{f_choice} · Astrology", summary)
                st.markdown(styling.score_gauge(score, "This reading"), unsafe_allow_html=True)
                with st.expander("Explore the full reading"):
                    styling.scroll_panel(f"Full {f_choice} reading", details)

        # -------------------------------------------------------------
        with tab_youtube:
            st.caption(
                "Aggregates what real astrology YouTube creators are currently saying about your "
                "sign — this reflects their opinions, not a validated prediction. Every claim links "
                "back to its actual source video."
            )
            sun_sign = chart["sun_sign"]
            yt_sign = st.selectbox(
                "Sign", astrology.SIGNS, index=astrology.SIGNS.index(sun_sign) if sun_sign in astrology.SIGNS else 0,
                key="yt_sign",
            )
            yt_period = st.selectbox("Time period", ["week", "month", "year"], key="yt_period")
            yt_question = st.text_input(
                "Optional question", key="yt_question",
                placeholder="e.g. What are astrologers saying about my career?",
            )
            btn_col, refresh_col = st.columns([2, 1])
            with btn_col:
                generate_clicked = st.button("Get YouTube Insights", type="primary", key="yt_generate")
            with refresh_col:
                force_refresh = st.checkbox("Force refresh", key="yt_force_refresh",
                                             help="Skip the 24-hour cache and fetch fresh results")

            if generate_clicked:
                with st.spinner("Researching astrology videos and extracting predictions..."):
                    result = run_youtube_insights_pipeline(
                        yt_sign, yt_period, yt_question or None, p.get("language", "en"), force_refresh
                    )
                st.session_state.yt_insights_last = result

            result = st.session_state.get("yt_insights_last")
            if result:
                if result.get("error"):
                    st.warning(result["error"])
                else:
                    evidence = result["evidence"]
                    styling.summary_card(f"{yt_sign} · YouTube Insights", result["summary"])
                    st.markdown(styling.score_gauge(result["score"], "This reading"), unsafe_allow_html=True)
                    st.caption(f"Based on {evidence['relevant_prediction_count']} recent video(s) "
                              f"mentioning {yt_sign}.")

                    with st.expander("Explore the full reading"):
                        styling.scroll_panel("Full YouTube Insights reading", result["details"])

                    if evidence["evidence_by_theme"]:
                        st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                        st.markdown("##### Sources — where each theme comes from")
                        st.markdown(styling.rule_based_badge(), unsafe_allow_html=True)
                        for theme in evidence["ranked_themes"]:
                            entries = evidence["evidence_by_theme"].get(theme, [])
                            if not entries:
                                continue
                            agreement = evidence["theme_agreement"].get(theme, len(entries))
                            with st.expander(f"{theme.title()} — {agreement} creator(s) agree"):
                                for e in entries:
                                    st.markdown(f"**[{e['source_title']}]({e['source_url']})** — "
                                              f"{e['source_channel']}")
                                    st.caption(e["prediction_summary"])

    # ========================================================================
    # PAGE: NUMEROLOGY
    # ========================================================================
    elif st.session_state.nav == "Numerology":
        st.subheader("Numerology")
        st.caption(
            "A belief-based self-reflection tradition, not a scientifically validated "
            "method of prediction — treat these numbers as a structured way to think "
            "about yourself, not a guarantee of what will happen."
        )
        system_used = p.get("numerology_system", "pythagorean").title()
        st.caption(f"Calculated with the {system_used} system · change this anytime in Profile → Edit Profile.")

        tab_overview, tab_cycles, tab_areas, tab_special, tab_names = st.tabs(
            ["Overview", "Life Cycles", "Life Areas", "Specialized Reading", "Name Comparison"]
        )

        # -------------------------------------------------------------
        with tab_overview:
            mulank = num["birthday"]["value"]
            bhagyank = num["life_path"]["value"]
            mulank_meaning = num["birthday"]["meaning"][0].lower() + num["birthday"]["meaning"][1:]
            bhagyank_meaning = num["life_path"]["meaning"][0].lower() + num["life_path"]["meaning"][1:]
            st.markdown(
                styling.vedic_number_cards(mulank, mulank_meaning, bhagyank, bhagyank_meaning),
                unsafe_allow_html=True,
            )

            profile_evidence = numerology_interpretation.build_profile_evidence(num)
            st.markdown(styling.rule_based_badge(), unsafe_allow_html=True)
            st.markdown("##### Key Findings")
            st.caption("Your core numbers and how they relate — straight from the rule engine, "
                      "visible instantly, no AI call needed to see this much.")
            for n in profile_evidence["numbers"][:3]:
                tier = 5 if n["is_master"] else (2 if n["karmic_debt"] else 3)
                tier_label = "Master Number" if n["is_master"] else \
                             (f"Karmic Debt {n['karmic_debt']}" if n["karmic_debt"] else f"Value {n['value']}")
                styling.key_finding_card(
                    n["context_label"], tier_label, tier, n["interpretation"], n["basis"],
                )
            for rel in profile_evidence["relationships"]:
                styling.key_finding_card(
                    f"{rel['label_a']} ↔ {rel['label_b']}", rel["strength_label"], rel["strength_tier"],
                    rel["interpretation"], rel["basis"],
                )
            with st.expander("See all core numbers"):
                for n in profile_evidence["numbers"][3:]:
                    tier = 5 if n["is_master"] else (2 if n["karmic_debt"] else 3)
                    tier_label = "Master Number" if n["is_master"] else \
                                 (f"Karmic Debt {n['karmic_debt']}" if n["karmic_debt"] else f"Value {n['value']}")
                    styling.key_finding_card(
                        n["context_label"], tier_label, tier, n["interpretation"], n["basis"],
                    )

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### AI-Personalized Reading")
            st.caption("Synthesizes the Key Findings above into an answer to your specific question.")
            reading_type, question = reading_type_and_question("num")

            if reading_type == "Question" and not question:
                st.info("Type your question above first.")
            elif st.button("Get my Numerology summary", type="primary", key="num_generate"):
                enriched_num = dict(num)
                enriched_num["rule_based_findings"] = profile_evidence
                with st.spinner("Reading your numbers..."):
                    score, summary, details = gemini_client.generate_engine_reading(
                        "Numerology", enriched_num, reading_type, question, language=p.get("language", "en")
                    )
                save_reading_safely(reading_type, "Single Engine", "Numerology", f"{summary}\n\n{details}")
                st.session_state.num_last = (score, summary, details)

            if st.session_state.num_last:
                score, summary, details = st.session_state.num_last
                styling.summary_card(f"{reading_type} · Numerology", summary)
                st.markdown(styling.score_gauge(score, "This reading"), unsafe_allow_html=True)
                with st.expander("Explore the full reading"):
                    styling.scroll_panel("Full Numerology reading", details)

            with st.expander("Explore your numbers"):
                items = [(v["value"], k.replace("_", " ").title()) for k, v in num.items()]
                st.markdown(styling.medallion_grid(items), unsafe_allow_html=True)

                karmic_flags = [(k, v) for k, v in num.items() if v.get("karmic_debt")]
                if karmic_flags:
                    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                    st.markdown("**Karmic Debt numbers found:**")
                    for key, entry in karmic_flags:
                        st.caption(f"⚠ {key.replace('_',' ').title()}: Karmic Debt "
                                  f"{entry['karmic_debt']} — {entry['karmic_debt_meaning']}")

                st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                st.markdown("**Calculation transparency** — see the exact arithmetic behind any number:")
                for key, entry in num.items():
                    styling.calculation_steps(key.replace("_", " ").title(), entry)

                with st.expander("What each number means"):
                    rows = [[k.replace("_", " ").title(), v["value"], v["meaning"]] for k, v in num.items()]
                    st.markdown(styling.table(["Number", "Value", "Meaning"], rows), unsafe_allow_html=True)

        # -------------------------------------------------------------
        with tab_cycles:
            st.caption("Four broad life periods, each with a Pinnacle (the opportunity) "
                      "and a Challenge (what it asks you to work through).")
            pc = numerology.pinnacle_and_challenge_cycles(p["dob"])
            current_idx = numerology.current_cycle_index(p["dob"])
            st.markdown(
                styling.timeline_cards(pc["pinnacles"], pc["challenges"], current_idx),
                unsafe_allow_html=True,
            )
            st.caption("Age ranges use the standard '36 minus Life Path' method — a widely used "
                      "convention, though some numerology schools calculate timing slightly differently.")

        # -------------------------------------------------------------
        with tab_areas:
            st.caption("Deterministic — every score below comes from a fixed weighting of your "
                      "actual numbers, never a random or AI-generated figure.")
            life_scores = numerology_scoring.theme_scores(num)
            for theme in numerology_scoring.THEMES:
                d = life_scores[theme]
                st.markdown(
                    styling.score_bar(theme.title(), d["score"], d["band"], d["reason"]),
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------------------
        with tab_special:
            st.caption("A reading focused on one area of life, still built only from your "
                      "calculated numbers above.")
            focus_choice = st.selectbox(
                "Focus area", ["Career", "Finance", "Relationships", "Personal Growth"], key="num_focus"
            )
            special_question = st.text_input(
                "Optional question", key="num_special_q",
                placeholder=f"e.g. What should I know about my {focus_choice.lower()} this year?",
            )
            if st.button(f"Get my {focus_choice} reading", type="primary", key="num_special_generate"):
                life_scores = numerology_scoring.theme_scores(num)
                enriched_data = dict(num)
                enriched_data["life_area_scores"] = life_scores
                with st.spinner(f"Reading your numbers for {focus_choice.lower()}..."):
                    score, summary, details = gemini_client.generate_engine_reading(
                        "Numerology", enriched_data, "Life", special_question or None,
                        focus=focus_choice.lower(), language=p.get("language", "en"),
                    )
                save_reading_safely(f"Numerology · {focus_choice}", "Single Engine", "Numerology",
                                     f"{summary}\n\n{details}")
                st.session_state.num_special_last = (focus_choice, score, summary, details)

            if st.session_state.get("num_special_last"):
                f_choice, score, summary, details = st.session_state.num_special_last
                styling.summary_card(f"{f_choice} · Numerology", summary)
                st.markdown(styling.score_gauge(score, "This reading"), unsafe_allow_html=True)
                with st.expander("Explore the full reading"):
                    styling.scroll_panel(f"Full {f_choice} reading", details)

        # -------------------------------------------------------------
        with tab_names:
            current_name = p.get("current_name", "").strip()
            if not current_name or current_name.lower() == p["name"].strip().lower():
                st.info(
                    "No separate current name on file, so there's nothing to compare — add one "
                    "in Profile → Edit Profile → Numerology settings if you have a married, "
                    "chosen, or otherwise different name in daily use."
                )
            else:
                cmp = numerology.compare_names(p["name"], current_name, p.get("numerology_system", "pythagorean"))
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Birth name** — {p['name']}")
                    for key in ("destiny", "soul_urge", "personality"):
                        st.metric(key.replace("_", " ").title(), cmp["birth_name"][key]["value"])
                with c2:
                    st.markdown(f"**Current name** — {current_name}")
                    for key in ("destiny", "soul_urge", "personality"):
                        st.metric(key.replace("_", " ").title(), cmp["current_name"][key]["value"])
                st.caption(
                    "Neither name is 'more correct' — some numerologists read the birth name as "
                    "your core blueprint and a current name as how you're expressing it day to day."
                )

    # ========================================================================
    # PAGE: PALMISTRY
    # ========================================================================
    elif st.session_state.nav == "Palmistry":
        st.subheader("Palmistry")
        st.caption(
            "Palmistry is a traditional, divinatory practice passed down over centuries — it is "
            "not scientifically validated. Every finding below is an AI-vision observation of "
            "YOUR actual photo, shown with its own confidence level, never a deterministic measurement."
        )

        hand = st.radio("Which hand?", ["Left", "Right"], horizontal=True, key="palm_hand")
        hand_key = hand.lower()
        stored = auth_store.get_palm_photo(st.session_state.user_id, hand_key)

        if stored is None:
            st.info(f"No {hand.lower()} palm photo saved yet.")
            if st.button(":material/add_a_photo: Add it in Profile", key="palm_goto_profile"):
                st.session_state.nav = "Profile"
                st.rerun()
        else:
            st.image(stored["image_bytes"], caption=f"{hand} palm (original)", width="stretch")
            st.caption("Want a different photo? Change it from Profile → Palm photos.")
            img_bytes, mime = stored["image_bytes"], stored["mime_type"]

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            palm_question = st.text_input(
                "What would you like to know?", key=f"palm_q_{hand_key}",
                placeholder="e.g. What does my life line say?",
            )
            if st.button("Get my Palmistry reading", type="primary", key=f"palm_generate_{hand_key}"):
                with st.spinner("Reading your palm — this looks closely, give it a moment..."):
                    result = gemini_client.palm_vision_reading_structured(
                        img_bytes, mime, hand_key, palm_question or None, language=p.get("language", "en")
                    )
                if "error" in result:
                    st.error(result["error"])
                else:
                    if result["image_quality_sufficient"] and result["summary"]:
                        save_reading_safely(f"Palmistry ({hand})", "AI-assisted", "Palmistry", result["summary"])
                    palm_last = dict(st.session_state.palm_last or {})
                    palm_last[hand_key] = result
                    st.session_state.palm_last = palm_last

        current = (st.session_state.palm_last or {}).get(hand_key)
        if current:
            if not current["image_quality_sufficient"]:
                st.warning(
                    f"Couldn't give a reliable reading from this photo: "
                    f"{current.get('quality_issue') or 'the image quality is too low.'} "
                    "Please retake it — see the tips in Profile → Palm photos — rather than get a "
                    "guessed reading from an unclear image."
                )
            else:
                score, summary = current["score"], current["summary"]
                styling.summary_card(f"Palmistry · {hand} hand", summary)
                st.markdown(styling.score_gauge(score, "This palm reading"), unsafe_allow_html=True)

                findings = current["findings"]
                if findings:
                    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                    st.markdown("##### Visual proof")
                    st.caption("Boxes mark where each Medium/High-confidence finding was detected on "
                              "your actual photo — Low-confidence guesses are listed below but not "
                              "boxed, so a box always means real visual confidence.")
                    annotated_bytes, drawn_count = palm_annotation.annotate_palm_image(img_bytes, findings)
                    if drawn_count:
                        st.image(annotated_bytes, caption=f"{hand} palm — annotated", width="stretch")
                    else:
                        st.caption("None of this reading's findings could be precisely localized on "
                                  "the photo — see the Palm Map below for what was found anyway.")

                    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                    st.markdown("##### Palm Map — tap any finding for its evidence")
                    for f in findings:
                        conf_icon = {"High": "🟢", "Medium": "🟡", "Low": "🟠"}.get(f["confidence"], "⚪")
                        with st.expander(f"{conf_icon} {f['feature']} — {f['confidence']} confidence"):
                            st.markdown(f"**Where AI sees it:** {f['location_description'] or 'general area, not precisely localized'}")
                            if f["markings"]:
                                st.markdown(f"**Markings noted:** {', '.join(f['markings'])}")
                            st.markdown(f"**Traditional meaning:** {f['traditional_meaning']}")
                            if f["note"]:
                                st.markdown(f"**On your hand specifically:** {f['note']}")

                    st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                    st.markdown("##### What this suggests, by life area")
                    st.caption("Each section only draws on the findings above — expand one to see exactly which.")
                    area_labels = {
                        "personality": "Personality", "career": "Career", "finance": "Finance",
                        "relationships": "Relationships", "strengths": "Strengths",
                        "challenges": "Challenges", "life_phases": "Life Phases",
                    }
                    for area_key, area_label in area_labels.items():
                        area = current["life_areas"].get(area_key, {})
                        narrative = area.get("narrative", "").strip()
                        if not narrative or narrative == "n/a":
                            continue
                        with st.expander(area_label):
                            st.write(narrative)
                            if area.get("linked_features"):
                                st.caption("Based on: " + ", ".join(area["linked_features"]))
                else:
                    st.caption("No specific features were confidently identified in this reading.")

        # Dominant vs non-dominant comparison, only when both hands have a usable reading
        both = st.session_state.palm_last or {}
        left_ok = both.get("left") and both["left"].get("image_quality_sufficient")
        right_ok = both.get("right") and both["right"].get("image_quality_sufficient")
        if left_ok and right_ok:
            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### Compare your hands")
            st.caption("Traditionally the dominant hand shows how you express yourself day to day; "
                      "the non-dominant hand shows more innate tendencies.")
            dominant_choice = st.radio("Which hand is dominant (the one you write with)?",
                                       ["Right", "Left"], horizontal=True, key="palm_dominant_choice")
            if st.button("Compare my hands", key="palm_compare_btn"):
                dominant_key = dominant_choice.lower()
                non_dominant_key = "left" if dominant_key == "right" else "right"
                with st.spinner("Comparing both hands..."):
                    comparison = gemini_client.palm_comparison_reading(
                        both[dominant_key], both[non_dominant_key], dominant_choice,
                        language=p.get("language", "en"),
                    )
                st.session_state.palm_comparison_last = comparison

            comp = st.session_state.get("palm_comparison_last")
            if comp and "error" not in comp:
                styling.scroll_panel("Hand comparison", comp.get("narrative", ""))
                for d in comp.get("differences", []):
                    with st.expander(d.get("feature", "Comparison")):
                        st.markdown(f"**{dominant_choice} (dominant):** {d.get('dominant_note','')}")
                        st.markdown(f"**Other hand:** {d.get('non_dominant_note','')}")
                        st.markdown(f"**What this suggests:** {d.get('interpretation','')}")
            elif comp and "error" in comp:
                st.error(comp["error"])

        st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
        with st.expander("Reference: where lines & mounts traditionally sit"):
            styling.palm_reference_card()

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

            spread_evidence = tarot_interpretation.build_spread_evidence(cards)
            st.markdown(styling.rule_based_badge(), unsafe_allow_html=True)
            st.markdown("##### Key Findings")
            st.caption("Each card's traditional meaning in its drawn position — visible instantly, "
                      "no AI call needed to see this much.")
            for c in spread_evidence["cards"]:
                styling.key_finding_card(
                    f"{c['name']} — {c['position']}", c["strength_label"], c["strength_tier"],
                    c["interpretation"], c["basis"],
                )
            if spread_evidence["patterns"]:
                st.caption("Patterns across the whole spread:")
                for pat in spread_evidence["patterns"]:
                    styling.key_finding_card(pat["pattern"], "Pattern", 4, pat["interpretation"], pat["basis"])

            st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
            st.markdown("##### AI-Personalized Reading")
            tarot_question = st.text_input(
                "What would you like to know?", key="tarot_q",
                placeholder="e.g. What should I focus on this week?",
            )
            if st.button("Get my Tarot summary", type="primary"):
                enriched_cards = {"cards": cards, "rule_based_findings": spread_evidence}
                with st.spinner("Reading the cards..."):
                    score, summary, details = gemini_client.generate_engine_reading(
                        "Tarot", enriched_cards, "Tarot spread", tarot_question or None,
                        language=p.get("language", "en"),
                    )
                save_reading_safely("Tarot spread", "Single Engine", "Tarot", f"{summary}\n\n{details}")
                st.session_state.tarot_last = (score, summary, details)

            if st.session_state.tarot_last:
                score, summary, details = st.session_state.tarot_last
                styling.summary_card("Tarot", summary)
                st.markdown(styling.score_gauge(score, "This spread"), unsafe_allow_html=True)
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
                f"{h.title()} hand: {r['summary']}" for h, r in palm_readings.items()
                if r and r.get("image_quality_sufficient") and r.get("summary")
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
                    score, summary, details = gemini_client.generate_unified_reading(
                        reading_type, u, question, language=p.get("language", "en")
                    )
                save_reading_safely(reading_type, "Combined", "all", f"{summary}\n\n{details}")
                # Snapshot each source engine's own rule-based evidence at the moment of
                # generation, so "open the individual reading" shows exactly what fed
                # into THIS synthesis, not whatever the chart/cards happen to be later.
                # Normalized to one shape (label, strength_label, strength_tier,
                # interpretation, basis) so the rendering loop below doesn't need to
                # know each engine's own internal dict shape.
                astro_placements = astrology_interpretation.build_chart_evidence(chart)["placements"][:2]
                num_numbers = numerology_interpretation.build_profile_evidence(num)["numbers"][:2]
                tarot_cards = tarot_interpretation.build_spread_evidence(cards)["cards"]
                source_evidence = {
                    "Astrology": [
                        {"label": f"{f['planet']} in {f['sign']} (House {f['house']})",
                         "strength_label": f["strength_label"], "strength_tier": f["strength_tier"],
                         "interpretation": f["interpretation"], "basis": f["basis"]}
                        for f in astro_placements
                    ],
                    "Numerology": [
                        {"label": f["context_label"],
                         "strength_label": "Master Number" if f["is_master"] else
                                           (f"Karmic Debt {f['karmic_debt']}" if f["karmic_debt"] else "Core Number"),
                         "strength_tier": 5 if f["is_master"] else (2 if f["karmic_debt"] else 3),
                         "interpretation": f["interpretation"], "basis": f["basis"]}
                        for f in num_numbers
                    ],
                    "Tarot": [
                        {"label": f"{f['name']} — {f['position']}",
                         "strength_label": f["strength_label"], "strength_tier": f["strength_tier"],
                         "interpretation": f["interpretation"], "basis": f["basis"]}
                        for f in tarot_cards
                    ],
                }
                st.session_state.anupt_last = (score, summary, details, u, source_evidence)

            if st.session_state.anupt_last:
                score, summary, details, u, source_evidence = st.session_state.anupt_last
                styling.summary_card(f"{reading_type} · ANUPT", summary)
                st.markdown(styling.score_gauge(score, "This reading"), unsafe_allow_html=True)

                st.markdown("##### Where the systems agree")
                cards_for_grid = [
                    (t.replace("_", " ").title(), u["theme_scores"][t]["strength"],
                     u["theme_scores"][t]["supporting_systems"])
                    for t in u["ranked_themes"][:4]
                ]
                styling.insight_grid(cards_for_grid)

                with st.expander("Explore the full reading"):
                    styling.scroll_panel("Full ANUPT reading", details)

                st.markdown("<div class='anupt-divider'></div>", unsafe_allow_html=True)
                st.markdown("##### Sources — how each system contributed")
                st.markdown(styling.rule_based_badge(), unsafe_allow_html=True)
                st.caption("A quick look at what each engine found on its own — open any of "
                          "them for the complete reading.")

                for source_name, findings in source_evidence.items():
                    if not findings:
                        continue
                    st.markdown(f"**{source_name}**")
                    for f in findings:
                        styling.key_finding_card(
                            f["label"], f["strength_label"], f["strength_tier"],
                            f["interpretation"], f["basis"],
                        )
                    if st.button(f"Open full {source_name} reading →", key=f"anupt_open_{source_name}"):
                        st.session_state.nav = source_name
                        st.rerun()

                if palm_summary_text:
                    st.markdown("**Palmistry**")
                    st.caption(palm_summary_text)
                    if st.button("Open full Palmistry reading →", key="anupt_open_palmistry"):
                        st.session_state.nav = "Palmistry"
                        st.rerun()

                with st.expander("Advanced Analysis — full evidence trail (raw)"):
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
