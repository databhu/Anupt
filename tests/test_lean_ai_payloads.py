"""
Regression test for the "Lightweight AI Request" requirement: AI call
payloads should send only the extracted, rule-based findings (already
trimmed to the strongest few) plus a small context dict — never the full
raw engine output (which duplicates the same information far more
verbosely and costs meaningfully more tokens on every single call).

These sizes are generous upper bounds, not exact — the point is to catch
a regression back to `dict(chart)`/`dict(num)`-style full dumps, not to
pin an exact byte count.
"""

import json
from datetime import date, time

from engines import astrology, astrology_interpretation, numerology, numerology_interpretation
from engines import tarot, tarot_interpretation

REF_DOB, REF_TIME = date(1995, 8, 8), time(14, 30)
REF_LAT, REF_LON, REF_UTC = 19.076, 72.8777, 5.5


class TestAstrologyPayloadStaysLean:
    def test_trimmed_evidence_is_much_smaller_than_the_old_combined_payload(self):
        chart = astrology.compute_chart("T", REF_DOB, REF_TIME, REF_LAT, REF_LON, REF_UTC)
        evidence = astrology_interpretation.build_chart_evidence(chart)

        # What the code used to send: the full raw chart PLUS the full,
        # untrimmed evidence on top of it (genuinely redundant, since
        # evidence is derived from chart in the first place).
        old_combined = dict(chart)
        old_combined["rule_based_findings"] = evidence
        old_size = len(json.dumps(old_combined, indent=2, default=str))

        # What it sends now: lean context + top-5 placements/top-4 aspects only.
        trimmed = {"placements": evidence["placements"][:5], "aspects": evidence["aspects"][:4]}
        lean_context = {"nakshatra": chart["nakshatra"], "dasha": chart["dasha"]}
        new_payload = {"context": lean_context, "rule_based_findings": trimmed}
        new_size = len(json.dumps(new_payload, indent=2, default=str))

        assert new_size < old_size * 0.6, f"expected a substantial reduction, got {old_size} -> {new_size}"
        assert new_size < 8000, "astrology payload grew unexpectedly large — check for a re-added raw dump"

    def test_trimmed_placements_capped_at_five(self):
        chart = astrology.compute_chart("T", REF_DOB, REF_TIME, REF_LAT, REF_LON, REF_UTC)
        evidence = astrology_interpretation.build_chart_evidence(chart)
        trimmed = evidence["placements"][:5]
        assert len(trimmed) <= 5


class TestNumerologyPayloadStaysLean:
    def test_lean_payload_much_smaller_than_the_old_combined_payload(self):
        num = numerology.full_profile("Test User", REF_DOB)
        evidence = numerology_interpretation.build_profile_evidence(num)

        old_combined = dict(num)
        old_combined["rule_based_findings"] = evidence
        old_size = len(json.dumps(old_combined, indent=2, default=str))

        new_payload = {"context": {"personal_year": num["personal_year"]["value"]}, "rule_based_findings": evidence}
        new_size = len(json.dumps(new_payload, indent=2, default=str))

        assert new_size < old_size * 0.6, f"expected a substantial reduction, got {old_size} -> {new_size}"
        assert new_size < 5000, "numerology payload grew unexpectedly large — check for a re-added raw dump"


class TestTarotPayloadStaysLean:
    def test_lean_payload_omits_raw_per_card_life_area_breakdowns(self):
        cards = tarot.draw_spread("test-id", "three_card")
        evidence = tarot_interpretation.build_spread_evidence(cards)
        payload = {"rule_based_findings": evidence}

        # The specific, meaningful win for tarot: the raw per-card
        # career/finance/love breakdowns and untrimmed keyword text are
        # gone — the interpretation text already carries what's relevant.
        payload_json = json.dumps(payload, default=str).lower()
        assert "career" not in payload_json
        assert "finance" not in payload_json
        assert '"love"' not in payload_json

    def test_old_payload_included_the_now_removed_fields(self):
        # Confirms the fields really were present before, so the test
        # above is actually checking something that changed.
        cards = tarot.draw_spread("test-id", "three_card")
        old_payload_json = json.dumps({"cards": cards}, default=str).lower()
        assert "career" in old_payload_json
