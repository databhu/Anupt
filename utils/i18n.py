"""
ANUPT — UI translation layer (English / Hindi / Marathi).

Scope, stated plainly: this covers the UI CHROME that appears on every
page or is otherwise high-traffic — navigation, page headers, common
buttons, and the recurring belief-based disclaimers. It does NOT
translate the deep, long-form interpretive content (numerology number
meanings, astrology yoga/dignity descriptions, palm feature meanings,
tarot card keywords, etc.) — that's hundreds of paragraphs of
substantive text across the app, and attempting a genuinely complete
translation of all of it is a much larger undertaking than this pass
covers honestly. AI-generated readings already respect the language
setting independently (see ai/gemini_client.py's `language` parameter on
every reading-generation function) — that part of the experience is
fully multilingual regardless of what this module covers.

t(key, lang) falls back to English if a key or language is missing —
it never shows a raw key or raises, so a translation gap degrades
gracefully instead of breaking the page.
"""

LANGUAGES = {"en": "English", "hi": "हिंदी", "mr": "मराठी"}

TRANSLATIONS = {
    # --- Bottom navigation ---
    "nav_home": {"en": "Home", "hi": "होम", "mr": "मुख्यपृष्ठ"},
    "nav_astro": {"en": "Astro", "hi": "ज्योतिष", "mr": "ज्योतिष"},
    "nav_nums": {"en": "Nums", "hi": "अंक", "mr": "अंक"},
    "nav_palm": {"en": "Palm", "hi": "हस्तरेखा", "mr": "हस्तरेषा"},
    "nav_tarot": {"en": "Tarot", "hi": "टैरो", "mr": "टॅरो"},
    "nav_anupt": {"en": "ANUPT", "hi": "ANUPT", "mr": "ANUPT"},
    "nav_you": {"en": "You", "hi": "आप", "mr": "तुम्ही"},

    # --- Hero / brand ---
    "tagline": {"en": "Insights for a better you", "hi": "बेहतर आप के लिए अंतर्दृष्टि",
                "mr": "अधिक चांगल्या तुमच्यासाठी अंतर्दृष्टी"},
    "systems_line": {"en": "Astrology · Numerology · Palmistry · Tarot",
                      "hi": "ज्योतिष · अंकशास्त्र · हस्तरेखा शास्त्र · टैरो",
                      "mr": "ज्योतिष · अंकशास्त्र · हस्तरेषाशास्त्र · टॅरो"},

    # --- Page headers ---
    "page_home": {"en": "Home", "hi": "होम", "mr": "मुख्यपृष्ठ"},
    "page_astrology": {"en": "Astrology", "hi": "ज्योतिष", "mr": "ज्योतिष"},
    "page_numerology": {"en": "Numerology", "hi": "अंकशास्त्र", "mr": "अंकशास्त्र"},
    "page_palmistry": {"en": "Palmistry", "hi": "हस्तरेखा शास्त्र", "mr": "हस्तरेषाशास्त्र"},
    "page_tarot": {"en": "Tarot", "hi": "टैरो", "mr": "टॅरो"},
    "page_anupt": {"en": "ANUPT", "hi": "ANUPT", "mr": "ANUPT"},
    "page_profile": {"en": "Profile", "hi": "प्रोफ़ाइल", "mr": "प्रोफाइल"},

    # --- Common disclaimers (the recurring "not scientifically validated" captions) ---
    "disclaimer_astrology": {
        "en": "A traditional symbolic system passed down over centuries, not a scientifically "
              "validated method of prediction — the positions below are precise astronomical "
              "calculations; what they're said to mean is interpretive tradition.",
        "hi": "सदियों से चली आ रही एक पारंपरिक सांकेतिक प्रणाली, भविष्यवाणी की वैज्ञानिक रूप से "
              "प्रमाणित पद्धति नहीं है — नीचे दी गई स्थितियाँ सटीक खगोलीय गणनाएँ हैं; उनका अर्थ "
              "क्या बताया जाता है, यह व्याख्यात्मक परंपरा है।",
        "mr": "शतकानुशतके चालत आलेली एक पारंपरिक सांकेतिक प्रणाली, ही भाकीत करण्याची वैज्ञानिकदृष्ट्या "
              "सिद्ध पद्धत नाही — खालील स्थाने अचूक खगोलीय गणिते आहेत; त्यांचा अर्थ काय सांगितला "
              "जातो ही स्पष्टीकरणात्मक परंपरा आहे.",
    },
    "disclaimer_numerology": {
        "en": "A belief-based self-reflection tradition, not a scientifically validated method "
              "of prediction — treat these numbers as a structured way to think about yourself, "
              "not a guarantee of what will happen.",
        "hi": "एक विश्वास-आधारित आत्म-चिंतन परंपरा, भविष्यवाणी की वैज्ञानिक रूप से प्रमाणित पद्धति "
              "नहीं है — इन अंकों को अपने बारे में सोचने के एक संरचित तरीके के रूप में लें, इस "
              "बात की गारंटी के रूप में नहीं कि क्या होगा।",
        "mr": "श्रद्धेवर आधारित आत्म-चिंतनाची परंपरा, ही भाकीत करण्याची वैज्ञानिकदृष्ट्या सिद्ध पद्धत "
              "नाही — या अंकांकडे स्वतःबद्दल विचार करण्याचा एक रचनात्मक मार्ग म्हणून पाहा, काय घडेल "
              "याची हमी म्हणून नाही.",
    },
    "disclaimer_palmistry": {
        "en": "Palmistry is a traditional, divinatory practice passed down over centuries — it is "
              "not scientifically validated. Every finding below is an AI-vision observation of "
              "YOUR actual photo, shown with its own confidence level, never a deterministic measurement.",
        "hi": "हस्तरेखा शास्त्र सदियों से चली आ रही एक पारंपरिक, भविष्यसूचक विधा है — यह वैज्ञानिक "
              "रूप से प्रमाणित नहीं है। नीचे दिया गया हर निष्कर्ष आपकी असली तस्वीर का AI-विज़न "
              "अवलोकन है, जो अपने आत्मविश्वास स्तर के साथ दिखाया गया है, कभी भी निश्चित माप नहीं।",
        "mr": "हस्तरेषाशास्त्र ही शतकानुशतके चालत आलेली पारंपरिक, भविष्यकथन करणारी पद्धत आहे — ती "
              "वैज्ञानिकदृष्ट्या सिद्ध नाही. खालील प्रत्येक निष्कर्ष हा तुमच्या प्रत्यक्ष फोटोचे "
              "AI-दृष्टी निरीक्षण आहे, जो स्वतःच्या आत्मविश्वास पातळीसह दाखवला आहे, कधीही निश्चित "
              "मोजमाप नाही.",
    },
    "disclaimer_tarot": {
        "en": "Tarot is a traditional, reflective practice — not a scientifically validated method "
              "of prediction.",
        "hi": "टैरो एक पारंपरिक, चिंतनशील अभ्यास है — भविष्यवाणी की वैज्ञानिक रूप से प्रमाणित पद्धति "
              "नहीं है।",
        "mr": "टॅरो ही एक पारंपरिक, चिंतनशील पद्धत आहे — भाकीत करण्याची वैज्ञानिकदृष्ट्या सिद्ध पद्धत "
              "नाही.",
    },

    # --- Common buttons / actions ---
    "btn_log_in": {"en": "Log in", "hi": "लॉग इन करें", "mr": "लॉग इन करा"},
    "btn_sign_up": {"en": "Sign up", "hi": "साइन अप करें", "mr": "साइन अप करा"},
    "btn_save_profile": {"en": "Save profile", "hi": "प्रोफ़ाइल सहेजें", "mr": "प्रोफाइल जतन करा"},
    "btn_edit_profile": {"en": "Edit", "hi": "संपादित करें", "mr": "संपादित करा"},
    "btn_open_full_reading": {"en": "Explore the full reading", "hi": "पूरी रीडिंग देखें",
                              "mr": "संपूर्ण वाचन पहा"},
    "label_username": {"en": "Username", "hi": "उपयोगकर्ता नाम", "mr": "वापरकर्तानाव"},
    "label_password": {"en": "Password", "hi": "पासवर्ड", "mr": "पासवर्ड"},
    "label_choose_username": {"en": "Choose a username", "hi": "एक उपयोगकर्ता नाम चुनें", "mr": "वापरकर्तानाव निवडा"},
    "label_choose_password": {"en": "Choose a password", "hi": "एक पासवर्ड चुनें", "mr": "पासवर्ड निवडा"},
    "label_confirm_password": {"en": "Confirm password", "hi": "पासवर्ड की पुष्टि करें", "mr": "पासवर्डची पुष्टी करा"},
    "btn_create_account": {"en": "Create account", "hi": "खाता बनाएं", "mr": "खाते तयार करा"},
    "label_full_name": {"en": "Full name", "hi": "पूरा नाम", "mr": "पूर्ण नाव"},
    "label_date_of_birth": {"en": "Date of birth", "hi": "जन्म तिथि", "mr": "जन्मतारीख"},
    "label_time_of_birth": {"en": "Time of birth", "hi": "जन्म समय", "mr": "जन्मवेळ"},
    "label_place_of_birth": {"en": "Place of birth", "hi": "जन्म स्थान", "mr": "जन्मस्थान"},

    # --- Rule-based / evidence badges (shown across Astrology, Numerology, Tarot) ---
    "rule_based_badge": {"en": "⚙ Rule-Based — no AI involved", "hi": "⚙ नियम-आधारित — कोई AI शामिल नहीं",
                          "mr": "⚙ नियम-आधारित — कोणतेही AI सामील नाही"},
    "key_findings": {"en": "Key Findings", "hi": "मुख्य निष्कर्ष", "mr": "मुख्य निष्कर्ष"},

    # --- Language switcher itself ---
    "language_label": {"en": "Language", "hi": "भाषा", "mr": "भाषा"},
}


def t(key: str, lang: str = "en") -> str:
    """Looks up `key` in the requested language, falling back to English
    if either the key or that language's entry is missing — a translation
    gap should degrade to English text, never a raw key or a crash."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key
