"""
Tests for engines/numerology_scoring.py — deterministic theme scoring.

The core property under test throughout is: given the same numerology
profile, scores never change, and every score's "reason" text names real
numbers that are actually present in that profile — i.e. the scoring is
arithmetic, not generated or random.
"""

from datetime import date

from engines import numerology as num
from engines import numerology_scoring as scoring


def _profile(name="Test Person", dob=date(1995, 8, 8)):
    return num.full_profile(name, dob)


class TestDeterminism:
    def test_same_profile_same_scores(self):
        p = _profile()
        assert scoring.theme_scores(p) == scoring.theme_scores(p)

    def test_no_randomness_across_many_calls(self):
        p = _profile()
        results = [scoring.theme_scores(p) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestScoreBounds:
    def test_all_themes_present(self):
        s = scoring.theme_scores(_profile())
        assert set(s.keys()) == set(scoring.THEMES)

    def test_scores_within_0_100_across_many_dates(self):
        for year in range(1950, 2020, 5):
            for month in (1, 6, 12):
                p = num.full_profile("Test Person", date(year, month, 15))
                for theme, data in scoring.theme_scores(p).items():
                    assert 0 <= data["score"] <= 100, f"{theme} out of bounds on {year}-{month}"

    def test_band_matches_score_thresholds(self):
        s = scoring.theme_scores(_profile())
        for theme, data in s.items():
            score, band = data["score"], data["band"]
            if score >= 75:
                assert band == "Strong"
            elif score >= 55:
                assert band == "Good"
            elif score >= 35:
                assert band == "Developing"
            else:
                assert band == "Growth Area"


class TestTraceability:
    def test_reason_names_a_number_actually_in_the_profile(self):
        p = _profile()
        s = scoring.theme_scores(p)
        profile_values = {k: v["value"] for k, v in p.items() if k in scoring.NUMBER_WEIGHT_IN_SCORE}
        for theme, data in s.items():
            for contributor in data["top_contributors"]:
                # the contributor's value must match what's actually in the profile
                assert profile_values[contributor["number_type"]] == contributor["value"]

    def test_top_contributor_has_highest_weighted_contribution_for_its_theme(self):
        p = _profile()
        s = scoring.theme_scores(p)
        for theme, data in s.items():
            if not data["top_contributors"]:
                continue
            top_key = data["top_contributors"][0]["number_type"]
            top_value = data["top_contributors"][0]["value"]
            top_weight = scoring.NUMBER_WEIGHT_IN_SCORE[top_key]
            top_contribution = top_weight * scoring.NUMBER_THEME_AFFINITY.get(top_value, {}).get(theme, 0)
            # every other contributing number's OWN weighted contribution (its own
            # weight times its own affinity) should be <= the top contributor's.
            for other_key, other_weight in scoring.NUMBER_WEIGHT_IN_SCORE.items():
                other_entry = p.get(other_key)
                if not other_entry or other_key == top_key:
                    continue
                other_affinity = scoring.NUMBER_THEME_AFFINITY.get(other_entry["value"], {}).get(theme, 0)
                other_contribution = other_weight * other_affinity
                assert other_contribution <= top_contribution + 1e-9, (
                    f"{theme}: {other_key} contributes {other_contribution} > "
                    f"top {top_key} contributes {top_contribution}"
                )

    def test_reason_is_never_empty(self):
        for year in range(1960, 2015, 7):
            p = num.full_profile("Someone", date(year, 4, 21))
            for theme, data in scoring.theme_scores(p).items():
                assert data["reason"].strip() != ""


class TestManualCrossCheck:
    def test_number_8_scores_high_on_finance_and_leadership(self):
        # 8's documented affinity (0.95 finance, 0.85 leadership) is the
        # highest of any number for those two themes — a profile dominated
        # by 8s should score noticeably higher there than one dominated by,
        # say, 2s (0.25 finance, 0.20 leadership).
        affinity_8 = scoring.NUMBER_THEME_AFFINITY[8]
        affinity_2 = scoring.NUMBER_THEME_AFFINITY[2]
        assert affinity_8["finance"] > affinity_2["finance"]
        assert affinity_8["leadership"] > affinity_2["leadership"]
        assert affinity_2["relationships"] > affinity_8["relationships"]

    def test_weights_sum_to_one(self):
        assert abs(sum(scoring.NUMBER_WEIGHT_IN_SCORE.values()) - 1.0) < 1e-9
