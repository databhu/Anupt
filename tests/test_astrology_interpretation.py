"""
Tests for engines/astrology_interpretation.py — the deterministic rule
engine that turns placements into genuine interpretation, not just raw
facts. The properties under test throughout: the SAME inputs always
produce the SAME output (determinism), and DIFFERENT inputs (dignity,
retrograde, combustion, orb tightness) produce genuinely DIFFERENT output
— not the same template with a noun swapped in.
"""

from datetime import date, time

from engines import astrology as astro
from engines import astrology_interpretation as interp

REF_DOB, REF_TIME = date(1995, 8, 8), time(14, 30)
REF_LAT, REF_LON, REF_UTC = 19.076, 72.8777, 5.5


class TestPlacementDeterminism:
    def test_same_inputs_produce_identical_output(self):
        a = interp.interpret_placement("Sun", "Leo", 10, "Rulership", False, "Angular")
        b = interp.interpret_placement("Sun", "Leo", 10, "Rulership", False, "Angular")
        assert a == b


class TestStrengthTierVariesByDignity:
    def test_exalted_scores_higher_than_debilitated(self):
        strong = interp.interpret_placement("Sun", "Aries", 10, "Exalted", False, "Angular")
        weak = interp.interpret_placement("Sun", "Libra", 10, "Fall", False, "Angular")
        assert strong["strength_tier"] > weak["strength_tier"]
        assert strong["strength_label"] != weak["strength_label"]
        assert strong["interpretation"] != weak["interpretation"]

    def test_five_tier_scale_covers_all_dignity_states(self):
        dignities = ["Exalted", "Own Sign", "Neutral", "Detriment", "Debilitated"]
        tiers = [interp.interpret_placement("Mars", "Aries", 1, d, False, "Angular")["strength_tier"]
                 for d in dignities]
        assert tiers == sorted(tiers, reverse=True)  # strictly decreasing as dignity worsens

    def test_all_five_strength_labels_are_distinct(self):
        labels = {v["label"] for v in interp.STRENGTH_TIER_LANGUAGE.values()}
        assert len(labels) == 5


class TestRetrogradeClause:
    def test_retrograde_adds_a_distinguishing_clause(self):
        direct = interp.interpret_placement("Mercury", "Gemini", 3, "Rulership", False, "Cadent")
        retro = interp.interpret_placement("Mercury", "Gemini", 3, "Rulership", True, "Cadent")
        assert direct["interpretation"] != retro["interpretation"]
        assert "basis" in retro and any("retrograde" in b for b in retro["basis"])

    def test_retrograde_clause_is_planet_specific(self):
        # Mercury retrograde and Mars retrograde must produce DIFFERENT
        # clauses, not the same "turns inward" sentence with the noun swapped.
        merc = interp.interpret_placement("Mercury", "Gemini", 3, "Neutral", True, "Cadent")
        mars = interp.interpret_placement("Mars", "Aries", 1, "Neutral", True, "Angular")
        merc_clause = interp._RETROGRADE_CLAUSE["Mercury"]
        mars_clause = interp._RETROGRADE_CLAUSE["Mars"]
        assert merc_clause != mars_clause
        assert merc_clause in merc["interpretation"]
        assert mars_clause in mars["interpretation"]

    def test_sun_and_moon_retrograde_is_a_no_op(self):
        # The Sun/Moon don't retrograde in standard practice; passing
        # retrograde=True for them shouldn't add a nonsensical clause.
        direct = interp.interpret_placement("Sun", "Leo", 1, "Rulership", False, "Angular")
        marked_retro = interp.interpret_placement("Sun", "Leo", 1, "Rulership", True, "Angular")
        assert direct["interpretation"] == marked_retro["interpretation"]


class TestCombustionException:
    """The clearest 'exception rule' in the system: proximity to the Sun
    overrides otherwise-favorable dignity, per real classical astrology."""

    def test_combust_planet_has_reduced_tier_vs_identical_non_combust(self):
        combust = interp.interpret_placement(
            "Mercury", "Virgo", 5, "Exalted", False, "Succedent",
            planet_longitude=155.0, sun_longitude=150.0,  # 5 deg apart, within Mercury's 14 deg orb
        )
        clear = interp.interpret_placement(
            "Mercury", "Virgo", 5, "Exalted", False, "Succedent",
            planet_longitude=155.0, sun_longitude=300.0,  # far apart
        )
        assert combust["strength_tier"] < clear["strength_tier"]
        assert combust["exceptions_applied"] and not clear["exceptions_applied"]

    def test_combustion_never_reduces_below_tier_1(self):
        result = interp.interpret_placement(
            "Saturn", "Aries", 6, "Debilitated", False, "Cadent",
            planet_longitude=100.0, sun_longitude=95.0,  # within Saturn's 15 deg orb
        )
        assert result["strength_tier"] >= 1

    def test_sun_is_never_combust(self):
        # The Sun trivially can't be combust relative to itself — it isn't
        # even in COMBUSTION_ORBS, since only OTHER planets are described as
        # combust by proximity to the Sun.
        assert interp.is_combust("Sun", 100.0, 100.0) is False
        assert "Sun" not in interp.COMBUSTION_ORBS

    def test_moon_can_be_combust_when_close_to_sun(self):
        # Unlike the Sun, the Moon genuinely can be combust in classical
        # Vedic astrology when close enough to the Sun.
        assert interp.is_combust("Moon", 100.0, 105.0) is True  # 5 deg, within Moon's 12 deg orb
        assert interp.is_combust("Moon", 100.0, 170.0) is False  # 70 deg, far outside

    def test_planet_far_from_sun_is_not_combust(self):
        assert interp.is_combust("Mercury", 10.0, 200.0) is False

    def test_planet_within_orb_of_sun_is_combust(self):
        assert interp.is_combust("Venus", 100.0, 105.0) is True  # 5 deg, within Venus's 10 deg orb

    def test_no_missing_api_leakage_when_longitudes_not_provided(self):
        # Callers that don't have longitude data (e.g. testing/demo code)
        # shouldn't crash — combustion just isn't checked.
        result = interp.interpret_placement("Mercury", "Virgo", 5, "Exalted", False, "Succedent")
        assert result["exceptions_applied"] == []


class TestAspectInterpretation:
    def test_tighter_orb_scores_higher_strength(self):
        tight = {"point_a": "Sun", "point_b": "Moon", "aspect": "Trine", "orb": 0.1}
        loose = {"point_a": "Sun", "point_b": "Moon", "aspect": "Trine", "orb": 7.5}
        r_tight = interp.interpret_aspect(tight)
        r_loose = interp.interpret_aspect(loose)
        assert r_tight["strength_tier"] > r_loose["strength_tier"]

    def test_different_aspect_types_produce_different_nature_text(self):
        square = {"point_a": "Mars", "point_b": "Saturn", "aspect": "Square", "orb": 1.0}
        trine = {"point_a": "Mars", "point_b": "Saturn", "aspect": "Trine", "orb": 1.0}
        r_square = interp.interpret_aspect(square)
        r_trine = interp.interpret_aspect(trine)
        assert r_square["interpretation"] != r_trine["interpretation"]

    def test_every_major_and_minor_aspect_type_has_nature_text(self):
        for name in list(astro.MAJOR_ASPECTS) + list(astro.MINOR_ASPECTS):
            assert name in interp._ASPECT_NATURE, name


class TestChartEvidenceBuilder:
    def test_produces_one_placement_per_planet(self):
        chart = astro.compute_chart("T", REF_DOB, REF_TIME, REF_LAT, REF_LON, REF_UTC)
        evidence = interp.build_chart_evidence(chart)
        assert len(evidence["placements"]) == len(chart["planets"])

    def test_placements_sorted_strongest_first(self):
        chart = astro.compute_chart("T", REF_DOB, REF_TIME, REF_LAT, REF_LON, REF_UTC)
        evidence = interp.build_chart_evidence(chart)
        tiers = [p["strength_tier"] for p in evidence["placements"]]
        assert tiers == sorted(tiers, reverse=True)

    def test_every_placement_has_full_evidence_chain(self):
        chart = astro.compute_chart("T", REF_DOB, REF_TIME, REF_LAT, REF_LON, REF_UTC)
        evidence = interp.build_chart_evidence(chart)
        for p in evidence["placements"]:
            assert p["interpretation"].strip() != ""
            assert len(p["basis"]) >= 1

    def test_no_crash_across_many_charts(self):
        for year in range(1960, 2020, 7):
            chart = astro.compute_chart("T", date(year, 3, 17), time(10, 0), REF_LAT, REF_LON, REF_UTC)
            evidence = interp.build_chart_evidence(chart)
            assert len(evidence["placements"]) > 0
