# ============================================================
# ANUPT – COSMIC AI (FULL PRODUCTION SINGLE FILE)
# Astrology + Numerology + Tarot + Palm + Structured Fusion
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

# ============================================================
# CONFIG
# ============================================================

OPENAI_KEY = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=OPENAI_KEY)

swe.set_ephe_path(".")
swe.set_sid_mode(swe.SIDM_LAHIRI)

tf = TimezoneFinder()

# ============================================================
# UTILITY
# ============================================================

SIGNS = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

def get_sign(deg: float) -> str:
    return SIGNS[int(deg // 30)]

# ============================================================
# CITY AUTOCOMPLETE (SAFE)
# ============================================================

def search_city(query: str):
    if not query or len(query) < 3:
        return []
    try:
        url = "https://photon.komoot.io/api/"
        params = {"q": query, "limit": 5}
        headers = {"User-Agent": "anupt-app"}
        r = requests.get(url, params=params, headers=headers, timeout=5)

        if r.status_code != 200:
            return []

        data = r.json()
        results = []
        for f in data.get("features", []):
            name = f["properties"].get("name", "")
            state = f["properties"].get("state", "")
            country = f["properties"].get("country", "")
            label = f"{name}, {state}, {country}"
            results.append(label)
        return results
    except:
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

def reduce_number(n: int) -> int:
    while n > 9 and n not in (11,22,33):
        n = sum(int(d) for d in str(n))
    return n

def numerology_engine(name: str, dob: datetime.date):
    dob_str = dob.strftime("%Y-%m-%d")
    life = reduce_number(sum(int(x) for x in dob_str if x.isdigit()))
    destiny = reduce_number(sum(LETTER_MAP.get(c,0) for c in name.upper() if c.isalpha()))
    current_year = datetime.datetime.now().year
    personal_year = reduce_number(
        sum(int(x) for x in dob_str[:7] if x.isdigit()) + current_year
    )

    return {
        "life_path": int(life),
        "destiny": int(destiny),
        "personal_year": int(personal_year)
    }

# ============================================================
# TAROT
# ============================================================

MAJOR_ARCANA = [
    "The Fool","The Magician","The High Priestess","The Empress",
    "The Emperor","The Lovers","The Chariot","Strength",
    "The Hermit","Wheel of Fortune","Justice","The Hanged Man",
    "Death","Temperance","The Devil","The Tower",
    "The Star","The Moon","The Sun","Judgement","The World"
]

def tarot_engine():
    deck = MAJOR_ARCANA.copy()
    random.shuffle(deck)

    spread = []
    positions = ["Past","Present","Future"]

    for i in range(3):
        spread.append({
            "position": positions[i],
            "card": deck[i],
            "reversed": random.choice([True, False])
        })

    return spread

# ============================================================
# ASTROLOGY
# ============================================================

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

def astrology_engine(dob: datetime.date, tob: datetime.time, pob: str):

    lat, lon, tz_name = get_location(pob)

    dt = datetime.datetime.combine(dob, tob)
    local = pytz.timezone(tz_name)
    dt_local = local.localize(dt)
    dt_utc = dt_local.astimezone(pytz.utc)

    jd = swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute/60
    )

    planets_data = {}
    for name, planet in PLANETS.items():
        pos, _ = swe.calc_ut(jd, planet, swe.FLG_SIDEREAL)
        deg = float(round(pos[0],2))
        planets_data[name] = {
            "degree": deg,
            "sign": get_sign(deg)
        }

    planets_data["Ketu"] = {
        "degree": float((planets_data["Rahu"]["degree"]+180)%360),
        "sign": get_sign((planets_data["Rahu"]["degree"]+180)%360)
    }

    houses, ascmc = swe.houses_ex(jd, lat, lon, b'P', swe.FLG_SIDEREAL)
    lagna_deg = float(round(ascmc[0],2))

    return {
        "ascendant": {
            "degree": lagna_deg,
            "sign": get_sign(lagna_deg)
        },
        "planets": planets_data
    }

def get_location(place):
    url = "https://photon.komoot.io/api/"
    params = {"q": place, "limit": 1}
    headers = {"User-Agent": "anupt"}
    r = requests.get(url, params=params, headers=headers, timeout=5)
    data = r.json()
    lon, lat = data["features"][0]["geometry"]["coordinates"]
    tz = tf.timezone_at(lat=lat, lng=lon)
    return float(lat), float(lon), tz

# ============================================================
# PALM ENGINE
# ============================================================

def palm_engine(file):
    if file is None:
        return None, None

    bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(bytes_data, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    strength = np.sum(edges) / 255

    if strength > 15000:
        level = "strong"
    elif strength > 8000:
        level = "moderate"
    else:
        level = "faint"

    overlay = img.copy()
    h,w,_ = overlay.shape

    cv2.line(overlay,(int(w*0.2),int(h*0.8)),(int(w*0.5),int(h*0.4)),(0,0,255),3)
    cv2.line(overlay,(int(w*0.1),int(h*0.5)),(int(w*0.8),int(h*0.5)),(255,0,0),3)
    cv2.line(overlay,(int(w*0.1),int(h*0.3)),(int(w*0.8),int(h*0.25)),(0,255,0),3)

    return {
        "life_line": level,
        "head_line": level,
        "heart_line": level
    }, overlay

# ============================================================
# AI FUSION
# ============================================================

def ask_ai(profile_data, question):

    prompt = f"""
You are a master advisor.

Structured Profile:
{profile_data}

Question:
{question}

Always give the detailed summary profile data of all the 4 engine segment
based on question match the summary of profile data and
Use astrology for destiny,
numerology for life direction,
tarot for situation,
palm for strengths.

Respond clearly and practically.

At the end summaries all the responses based on the question asked in 2 to 3 lines
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7
    )

    return res.choices[0].message.content

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🌟 ANUPT – Cosmic Intelligence")

name = st.text_input("Name")

dob = st.date_input(
    "Date of Birth",
    min_value=datetime.date(1900,1,1),
    max_value=datetime.date.today()
)

tob = st.time_input("Time of Birth")

city_query = st.text_input("Type birth city")
suggestions = search_city(city_query)
pob = st.selectbox("Select city", suggestions) if suggestions else ""

st.subheader("Left Palm")
left_cam = st.camera_input("Take photo (left)", key="left_cam")
left_up = st.file_uploader("Or upload", type=["jpg","png"], key="left_upload")

st.subheader("Right Palm")
right_cam = st.camera_input("Take photo (right)", key="right_cam")
right_up = st.file_uploader("Or upload", type=["jpg","png"], key="right_upload")

def pick(cam, up):
    return cam if cam else up

left_file = pick(left_cam, left_up)
right_file = pick(right_cam, right_up)

if name and pob:
    st.info(f"""
    Confirm Details:
    Name: {name}
    DOB: {dob}
    TOB: {tob}
    POB: {pob}
    """)

if st.button("Generate Full Profile"):

    astro = astrology_engine(dob, tob, pob)
    num = numerology_engine(name, dob)
    tarot = tarot_engine()
    left_palm, left_overlay = palm_engine(left_file)
    right_palm, right_overlay = palm_engine(right_file)

    profile = {
        "astrology": astro,
        "numerology": num,
        "tarot": tarot,
        "palm": {
            "left": left_palm,
            "right": right_palm
        }
    }

    st.session_state["profile"] = profile

    st.success("Profile Generated")

    if left_overlay is not None:
        st.image(left_overlay, caption="Left Palm Lines")

    if right_overlay is not None:
        st.image(right_overlay, caption="Right Palm Lines")

if "profile" in st.session_state:
    question = st.text_input("Ask your question")

    if st.button("Ask Universe"):
        answer = ask_ai(st.session_state["profile"], question)
        st.write(answer)
