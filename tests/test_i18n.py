"""
Tests for utils/i18n.py — the UI translation layer. The properties that
matter most: every key has complete en/hi/mr coverage (no silent gaps),
and lookup failures degrade gracefully (English fallback, or the raw key
as a last resort) rather than raising or showing something broken.
"""

from utils import i18n


class TestTranslationCompleteness:
    def test_every_key_has_all_three_languages(self):
        missing = []
        for key, translations in i18n.TRANSLATIONS.items():
            for lang in ["en", "hi", "mr"]:
                if lang not in translations or not translations[lang].strip():
                    missing.append((key, lang))
        assert missing == []

    def test_no_key_has_extra_unexpected_languages(self):
        for key, translations in i18n.TRANSLATIONS.items():
            assert set(translations.keys()) == {"en", "hi", "mr"}, key

    def test_languages_dict_matches_supported_set(self):
        assert set(i18n.LANGUAGES.keys()) == {"en", "hi", "mr"}


class TestLookup:
    def test_basic_lookup_returns_correct_language(self):
        assert i18n.t("nav_home", "en") == "Home"
        assert i18n.t("nav_home", "hi") == "होम"
        assert i18n.t("nav_home", "mr") == "मुख्यपृष्ठ"

    def test_defaults_to_english_when_lang_omitted(self):
        assert i18n.t("nav_home") == "Home"

    def test_unsupported_language_falls_back_to_english(self):
        assert i18n.t("nav_home", "fr") == "Home"
        assert i18n.t("nav_home", "de") == "Home"

    def test_missing_key_returns_the_key_itself_not_a_crash(self):
        assert i18n.t("this_key_does_not_exist", "hi") == "this_key_does_not_exist"

    def test_every_translation_key_lookup_works_in_all_languages(self):
        for key in i18n.TRANSLATIONS:
            for lang in ["en", "hi", "mr"]:
                result = i18n.t(key, lang)
                assert result.strip() != ""
                assert result != key  # a real translation, not a fallback-to-key miss
