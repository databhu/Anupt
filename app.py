# ============================================================
# ANUPT – COSMIC AI (PREMIUM UX VERSION)
# ============================================================

import streamlit as st
import swisseph as swe
import datetime
import pytz
import random
import requests
import numpy as np
import cv2
from timezonefinder import TimezoneFinder
from openai import OpenAI

OPENAI_KEY = st.secrets["OPENAI_KEY"]

swe.set_ephe_path(".")
swe.set_sid_mode(swe.SIDM_LAHIRI)

tf = TimezoneFinder()


# ============================================================
# CITY AUTOCOMPLETE
# ============================================================

def search_city(query):
    if not query or len(query) < 3:
        return []

    try:
        url = "https://photon.komoot.io/api/"
        params = {"q": query, "limit": 5}
        headers = {"User-Agent": "astro-app"}

        r = requests.get(url, params=params, headers=headers, timeout=10)

        # check server response
        if r.status_code != 200:
            return []

        # sometimes API returns text/html instead of json
        if "application/json" not in r.headers.get("Content-Type", ""):
            return []

        data = r.json()

        cities = []
        for f in data.get("features", []):
            name = f["properties"].get("name", "")
            state = f["properties"].get("state", "")
            country = f["properties"].get("country", "")
            label = f"{name}, {state}, {country}"
            cities.append(label)

        return cities

    except Exception as e:
        return []



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
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n


def life_path(dob):
    return reduce_number(sum(int(x) for x in dob if x.isdigit()))


# ============================================================
# TAROT
# ============================================================

TAROT_CARDS = [
    "The Fool", "The Magician", "The Lovers", "The Tower",
    "The Star", "The Moon", "The Sun", "Judgement"
]

def draw_tarot():
    deck = TAROT_CARDS.copy()
    random.shuffle(deck)
    return deck[:3]


# ============================================================
# ASTROLOGY
# ============================================================

SIGNS = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS
}

def get_sign(deg):
    return SIGNS[int(deg // 30)]


# ============================================================
# PALM OVERLAY
# ============================================================

def draw_palm_overlay(file):

    if file is None:
        return None

    bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(bytes_data, 1)

    h, w, _ = img.shape

    # life line
    cv2.line(img, (int(w*0.2), int(h*0.8)), (int(w*0.5), int(h*0.4)), (0,0,255), 3)
    # head line
    cv2.line(img, (int(w*0.1), int(h*0.5)), (int(w*0.8), int(h*0.5)), (255,0,0), 3)
    # heart line
    cv2.line(img, (int(w*0.1), int(h*0.3)), (int(w*0.8), int(h*0.25)), (0,255,0), 3)

    return img


# ============================================================
# AI CHAT
# ============================================================

def ask_ai(profile, question):
    client = OpenAI(api_key=OPENAI_KEY)

    prompt = f"""
You are a master advisor.

Profile:
{profile}

Question:
{question}

Give confident practical life guidance.
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return res.choices[0].message.content


# ============================================================
# UI
# ============================================================

st.title("🌟 ANUPT")
st.caption("Decode destiny using Astrology • Numerology • Tarot • Palm")

# ------------------------------------------------------------
# SMART INPUTS
# ------------------------------------------------------------

name = st.text_input("Your Name")

dob = st.date_input(
    "Date of Birth",
    min_value=datetime.date(1900, 1, 1),
    max_value=datetime.date.today()
)# 📅 calendar

tob = st.time_input("Time of Birth")   # ⏰ clock

city_query = st.text_input("Type your birth city")
suggestions = search_city(city_query)

pob = st.selectbox("Select city", suggestions) if suggestions else ""


# ------------------------------------------------------------
# PALM
# ------------------------------------------------------------

st.subheader("Left Palm")
left_cam = st.camera_input("Take photo")
left_up = st.file_uploader("or Upload", type=["jpg","png"], key="lu")

st.subheader("Right Palm")
right_cam = st.camera_input("Take photo")
right_up = st.file_uploader("or Upload", type=["jpg","png"], key="ru")


def pick(cam, up):
    return cam if cam else up


left_file = pick(left_cam, left_up)
right_file = pick(right_cam, right_up)


# ------------------------------------------------------------
# CONFIRM DETAILS
# ------------------------------------------------------------

if name and pob:
    st.info(f"""
    **Please confirm your details**

    Name: {name}  
    DOB: {dob}  
    TOB: {tob}  
    POB: {pob}
    """)


# ============================================================
# GENERATE PROFILE
# ============================================================

if st.button("Generate My Destiny"):

    tarot = draw_tarot()

    profile = f"""
Name: {name}
DOB: {dob}
TOB: {tob}
POB: {pob}

Life Path: {life_path(str(dob))}

Tarot:
Past {tarot[0]}
Present {tarot[1]}
Future {tarot[2]}
"""

    st.session_state["profile"] = profile
    st.success("Profile Ready ✨")


# ============================================================
# SHOW PALM WITH LINES
# ============================================================

if left_file:
    img = draw_palm_overlay(left_file)
    st.image(img, caption="Life(red) Head(blue) Heart(green)")


# ============================================================
# CHAT
# ============================================================

if "profile" in st.session_state:
    q = st.text_input("Ask your question")

    if st.button("Ask Universe"):
        ans = ask_ai(st.session_state["profile"], q)
        st.write(ans)
