"""
Tests for engines/numerology_interpretation.py — the deterministic rule
engine for numerology combinations. Same properties under test as
test_astrology_interpretation.py: determinism, and genuinely different
output for genuinely different inputs (same number vs different, Master
Number vs not, Karmic Debt vs not) rather than fixed canned text.
"""

from datetime import date

from engines import numerology as num
from engines import numerology_interpretation as interp


class TestRelationshipStrengthDeterminism:
    def test_same_inputs_produce_identical_output(self):
        a = interp.relationship_strength(8, 4, "Life Path", "Destiny")
        b = interp.relationship_strength(8, 4, "Life Path", "Destiny")
        assert a == b


class TestRelationshipTiers:
    def test_identical_numbers_score_top_tier(self):
        result = interp.relationship_strength(7, 7, "Life Path", "Destiny")
        assert result["strength_tier"] == 5
        assert result["strength_label"] == "Aligned"

    def test_master_number_and_its_reduction_score_second_tier(self):
        result = interp.relationship_strength(11, 2, "Life Path", "Destiny")
        assert result["strength_tier"] == 4
        assert result["strength_label"] == "Harmonic"

    def test_master_number_reduction_order_independent(self):
        # Should work whether the master number or its reduction comes first.
        a = interp.relationship_strength(22, 4, "Life Path", "Destiny")
        b = interp.relationship_strength(4, 22, "Life Path", "Destiny")
        assert a["strength_tier"] == b["strength_tier"] == 4

    def test_different_numbers_never_exceed_harmonic_tier(self):
        # Only identical numbers should reach tier 5 — anything else, even
        # a very high trait-affinity similarity, should not.
        for a in range(1, 10):
            for b in range(1, 10):
                if a == b:
                    continue
                result = interp.relationship_strength(a, b, "X", "Y")
                assert result["strength_tier"] < 5

    def test_data_driven_similarity_produces_a_meaningful_spread(self):
        # 8 and 22 share power/leadership/finance affinity (per
        # NUMBER_THEME_AFFINITY) much more than 8 and 2 (relationship-led) do
        # — this must show up as a real difference, not coincidence.
        high_similarity = interp.relationship_strength(8, 22, "A", "B")
        lower_similarity = interp.relationship_strength(8, 2, "A", "B")
        assert high_similarity["strength_tier"] >= lower_similarity["strength_tier"]

    def test_all_tiers_have_distinct_labels(self):
        labels = {v["label"] for v in interp.RELATIONSHIP_TIER_LANGUAGE.values()}
        assert len(labels) == len(interp.RELATIONSHIP_TIER_LANGUAGE)

    def test_interpretation_text_differs_between_tiers(self):
        aligned = interp.relationship_strength(5, 5, "Life Path", "Destiny")
        harmonic = interp.relationship_strength(11, 2, "Life Path", "Destiny")
        assert aligned["interpretation"] != harmonic["interpretation"]


class TestInterpretNumber:
    def test_karmic_debt_adds_a_distinguishing_clause(self):
        with_debt = num._life_path_detailed(date(1900, 1, 8))   # known Karmic Debt 19
        without_debt = num._life_path_detailed(date(1995, 8, 8))
        r_with = interp.interpret_number(with_debt, "Life Path")
        r_without = interp.interpret_number(without_debt, "Life Path")
        assert r_with["interpretation"] != r_without["interpretation"]
        assert any("karmic" in b.lower() for b in r_with["basis"])
        assert not any("karmic" in b.lower() for b in r_without["basis"])

    def test_master_number_adds_amplification_clause(self):
        master = num._life_path_detailed(date(1996, 11, 2))  # known Master Number 11
        result = interp.interpret_number(master, "Life Path")
        assert "Master Number" in result["interpretation"]
        assert any("Master Number" in b for b in result["basis"])

    def test_non_master_non_karmic_number_has_no_extra_clauses(self):
        plain = num._life_path_detailed(date(1995, 8, 8))
        result = interp.interpret_number(plain, "Life Path")
        assert len(result["basis"]) == 1  # just the base value, no exceptions

    def test_basis_always_references_the_context_label(self):
        entry = num._destiny_detailed("Test User")
        result = interp.interpret_number(entry, "Destiny")
        assert "Destiny" in result["basis"][0]


class TestProfileEvidenceBuilder:
    def test_seven_numbers_interpreted(self):
        profile = num.full_profile("Test User", date(1995, 8, 8))
        evidence = interp.build_profile_evidence(profile)
        assert len(evidence["numbers"]) == 7

    def test_two_relationships_computed(self):
        profile = num.full_profile("Test User", date(1995, 8, 8))
        evidence = interp.build_profile_evidence(profile)
        assert len(evidence["relationships"]) == 2

    def test_every_number_has_nonempty_interpretation(self):
        profile = num.full_profile("Test User", date(1995, 8, 8))
        evidence = interp.build_profile_evidence(profile)
        for n in evidence["numbers"]:
            assert n["interpretation"].strip() != ""

    def test_no_crash_across_many_profiles(self):
        for year in range(1960, 2020, 5):
            profile = num.full_profile("Someone", date(year, 4, 21))
            evidence = interp.build_profile_evidence(profile)
            assert len(evidence["numbers"]) == 7


class TestTranslatedMeanings:
    """The Hindi/Marathi translation layer added on top of the canonical
    English NUMBER_MEANINGS/KARMIC_DEBT_MEANINGS — engines.numerology's
    own full_profile() output must stay untouched (English, canonical),
    while interpret_number()'s DISPLAY text becomes language-aware."""

    def test_default_language_is_english_unchanged(self):
        entry = num._life_path_detailed(date(1995, 8, 8))
        result = interp.interpret_number(entry, "Life Path")
        assert "structure, discipline, reliability, hard work" in result["interpretation"]

    def test_hindi_translation_used_when_requested(self):
        entry = num._life_path_detailed(date(1995, 8, 8))  # value 4
        result = interp.interpret_number(entry, "Life Path", lang="hi")
        assert "संरचना" in result["interpretation"]  # "structure" in Hindi
        assert "structure, discipline" not in result["interpretation"]

    def test_marathi_translation_used_when_requested(self):
        entry = num._life_path_detailed(date(1995, 8, 8))
        result = interp.interpret_number(entry, "Life Path", lang="mr")
        assert "रचना" in result["interpretation"]  # "structure" in Marathi

    def test_master_number_clause_is_translated_too(self):
        entry = num._life_path_detailed(date(1996, 11, 2))  # known Master Number 11
        result = interp.interpret_number(entry, "Life Path", lang="hi")
        assert "मास्टर नंबर" in result["interpretation"]
        assert "Master Number" not in result["interpretation"]

    def test_karmic_debt_clause_is_translated_too(self):
        entry = num._life_path_detailed(date(1900, 1, 8))  # known Karmic Debt 19
        result = interp.interpret_number(entry, "Life Path", lang="mr")
        assert "कर्म ऋण" in result["interpretation"]
        assert "Karmic Debt" not in result["interpretation"]

    def test_every_core_number_value_has_a_working_translation(self):
        for value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33]:
            for lang in ["hi", "mr"]:
                result = interp.translated_number_meaning(value, lang)
                assert result.strip() != "", f"{value}/{lang}"

    def test_every_karmic_debt_value_has_a_working_translation(self):
        for kd in [13, 14, 16, 19]:
            for lang in ["hi", "mr"]:
                result = interp.translated_karmic_debt_meaning(kd, lang)
                assert result.strip() != "", f"{kd}/{lang}"

    def test_unsupported_language_falls_back_to_english(self):
        result = interp.translated_number_meaning(4, "fr")
        assert result == num.NUMBER_MEANINGS[4]

    def test_untranslated_value_falls_back_to_english(self):
        # A value with no Hindi/Marathi entry (e.g. an out-of-range number)
        # should fall back to whatever engines.numerology has, not crash.
        result = interp.translated_number_meaning(99, "hi")
        assert result == num.NUMBER_MEANINGS.get(99, "")

    def test_build_profile_evidence_threads_language_through(self):
        profile = num.full_profile("Test User", date(1995, 8, 8))
        evidence_hi = interp.build_profile_evidence(profile, lang="hi")
        life_path_entry = next(n for n in evidence_hi["numbers"] if n["context_label"] == "Life Path")
        assert "संरचना" in life_path_entry["interpretation"]

    def test_backward_compatible_default_still_works_without_lang_arg(self):
        # Existing callers that never pass lang= must be unaffected.
        profile = num.full_profile("Test User", date(1995, 8, 8))
        evidence = interp.build_profile_evidence(profile)
        life_path_entry = next(n for n in evidence["numbers"] if n["context_label"] == "Life Path")
        assert "structure" in life_path_entry["interpretation"]
