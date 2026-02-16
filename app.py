# ============================================================
# COSMIC AI – COMPLETE STREAMLIT APPLICATION
# Astrology + Numerology + Tarot + Palm + Chat
# Camera OR Upload supported
# ============================================================

import streamlit as st
import swisseph as swe
import datetime
import pytz
import random
import requests
import cv2
import numpy as np
from timezonefinder import TimezoneFinder
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================


OPENAI_KEY = st.secrets["OPENAI_KEY"]



swe.set_ephe_path(".")
swe.set_sid_mode(swe.SIDM_LAHIRI)

tf = TimezoneFinder()

# ============================================================
# AI NORMALIZER (fix messy DOB/TOB)
# ============================================================

def normalize_input(user_text, format_type):
    if not user_text:
        return ""

    client = OpenAI(api_key=OPENAI_KEY)

    if format_type == "date":
        instruction = "Convert to YYYY-MM-DD"
    else:
        instruction = "Convert to HH:MM in 24 hour format"

    prompt = f"""
Convert the following input into {instruction}.
If unclear, guess intelligently.
Output only final value.

Input: {user_text}
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip()


# ============================================================
# NUMEROLOGY
# ============================================================

LETTER_MAP = {
    **dict.fromkeys(list("AJS"), 1),
    **dict.fromkeys(list("BKT"), 2),
    **dict.fromkeys(list("CLU"), 3),
    **dict.fromkeys(list("DMV"), 4),
    **dict.fromkeys(list("ENW"), 5),
    **dict.fromkeys(list("FOX"), 6),
    **dict.fromkeys(list("GPY"), 7),
    **dict.fromkeys(list("HQZ"), 8),
    **dict.fromkeys(list("IR"), 9),
}

def reduce_number(n):
    while n > 9 and n not in (11, 22, 33):
        n = sum(int(d) for d in str(n))
    return n

def life_path(dob):
    digits = [int(x) for x in dob if x.isdigit()]
    return reduce_number(sum(digits))

def destiny_number(name):
    total = sum(LETTER_MAP.get(c, 0) for c in name.upper() if c.isalpha())
    return reduce_number(total)

def personal_year(dob):
    year = datetime.datetime.now().year
    day_month = sum(int(x) for x in dob[:7] if x.isdigit())
    return reduce_number(day_month + year)


# ============================================================
# TAROT
# ============================================================

TAROT_CARDS = [
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Lovers", "The Chariot", "Strength",
    "The Hermit", "Wheel of Fortune", "Justice", "The Hanged Man",
    "Death", "Temperance", "The Devil", "The Tower",
    "The Star", "The Moon", "The Sun", "Judgement", "The World"
]

def draw_tarot():
    deck = TAROT_CARDS.copy()
    random.shuffle(deck)
    return deck[:3]


# ============================================================
# ASTROLOGY
# ============================================================

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE
}

def get_sign(deg):
    return SIGNS[int(deg // 30)]


# ============================================================
# LOCATION
# ============================================================

def get_lat_lon_timezone(place):
    url = "https://photon.komoot.io/api/"
    params = {"q": place, "limit": 1}
    headers = {"User-Agent": "astro-app"}

    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()

    lon, lat = data["features"][0]["geometry"]["coordinates"]
    timezone_str = tf.timezone_at(lat=lat, lng=lon)

    return lat, lon, timezone_str


# ============================================================
# JULIAN + PLANETS
# ============================================================

def get_julian_day(dob, tob, timezone):
    local = pytz.timezone(timezone)
    naive = datetime.datetime.strptime(f"{dob} {tob}", "%Y-%m-%d %H:%M")
    local_dt = local.localize(naive)
    utc_dt = local_dt.astimezone(pytz.utc)

    jd = swe.julday(
        utc_dt.year, utc_dt.month, utc_dt.day,
        utc_dt.hour + utc_dt.minute / 60
    )
    return jd


def get_planet_positions(jd):
    positions = {}
    for name, planet in PLANETS.items():
        pos, _ = swe.calc_ut(jd, planet, swe.FLG_SIDEREAL)
        positions[name] = round(pos[0], 2)
    positions["Ketu"] = round((positions["Rahu"] + 180) % 360, 2)
    return positions


# ============================================================
# PALM ENGINE
# ============================================================

def analyze_palm(file):
    if file is None:
        return "Not provided"

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    strength = np.sum(edges) / 255

    if strength > 15000:
        return "strong lines"
    elif strength > 8000:
        return "moderate lines"
    else:
        return "faint lines"


# ============================================================
# PROFILE BUILDER
# ============================================================

def build_profile(name, dob, tob, pob, planets, lagna, tarot, left_palm, right_palm):
    asc_sign = get_sign(lagna)
    moon_sign = get_sign(planets["Moon"])
    planet_text = ", ".join([f"{p} in {get_sign(d)}" for p, d in planets.items()])

    lp = life_path(dob)
    destiny = destiny_number(name)
    pyear = personal_year(dob)

    return f"""
PERSON:
{name}, born {dob} {tob} at {pob}

ASTROLOGY:
Ascendant: {asc_sign}
Moon Sign: {moon_sign}
Planets: {planet_text}

NUMEROLOGY:
Life Path: {lp}
Destiny: {destiny}
Personal Year: {pyear}

TAROT:
Past: {tarot[0]}
Present: {tarot[1]}
Future: {tarot[2]}

PALM:
Left: {left_palm}
Right: {right_palm}
"""


# ============================================================
# AI CHAT
# ============================================================

def ask_ai(profile, question):
    client = OpenAI(api_key=OPENAI_KEY)

    prompt = f"""
You are a master life advisor.

Use:
- astrology for destiny
- numerology for nature
- tarot for current energy
- palm for strengths

Profile:
{profile}

Question:
{question}

Give confident, clear, practical advice.
Mention timing if visible.
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return res.choices[0].message.content


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("ANUPT")
st.subheader("Decode your destiny powered by astrology, numerology, palm reading, and tarot.") # Use st.subheader for main subtitles

name = st.text_input("Name")
dob = st.text_input("DOB")
tob = st.text_input("TOB")
pob = st.text_input("Place of Birth")

# ------------------------------------------------------------
# PALM INPUT – CAMERA OR UPLOAD
# ------------------------------------------------------------

st.subheader("✋ Left Palm")
left_camera = st.camera_input("Take photo (left)")
left_upload = st.file_uploader("Or upload", type=["jpg", "png"], key="l")

st.subheader("✋ Right Palm")
right_camera = st.camera_input("Take photo (right)")
right_upload = st.file_uploader("Or upload", type=["jpg", "png"], key="r")


def pick_image(camera, upload):
    if camera is not None:
        return camera
    if upload is not None:
        return upload
    return None


left_file = pick_image(left_camera, left_upload)
right_file = pick_image(right_camera, right_upload)


# ============================================================
# GENERATE PROFILE
# ============================================================

if st.button("Generate My Profile"):

    with st.spinner("Reading your destiny..."):

        dob = normalize_input(dob, "date")
        tob = normalize_input(tob, "time")

        lat, lon, timezone = get_lat_lon_timezone(pob)
        jd = get_julian_day(dob, tob, timezone)
        planets = get_planet_positions(jd)

        houses, ascmc = swe.houses_ex(jd, lat, lon, b'P', swe.FLG_SIDEREAL)
        lagna = ascmc[0]

        tarot = draw_tarot()

        left_palm = analyze_palm(left_file)
        right_palm = analyze_palm(right_file)

        profile = build_profile(
            name, dob, tob, pob,
            planets, lagna, tarot,
            left_palm, right_palm
        )

        st.session_state["profile"] = profile

    st.success("Profile ready! Ask your questions below.")


# ============================================================
# CHAT
# ============================================================

if "profile" in st.session_state:
    question = st.text_input("Ask anything about your life")

    if st.button("Ask"):
        with st.spinner("Consulting the universe..."):
            ans = ask_ai(st.session_state["profile"], question)
        st.write(ans)
