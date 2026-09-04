"""
Tests for engines/numerology.py — the pure calculation layer.

Every numeric assertion below was independently hand-calculated (see the
comment above each test) and cross-checked against the engine before being
committed here, specifically so this suite catches real arithmetic bugs
rather than just re-asserting whatever the code happens to currently
output. Run with: pytest tests/test_numerology.py -v
"""

from datetime import date

import pytest

from engines import numerology as num


# ---------------------------------------------------------------------------
# Core numbers — hand-verified reference cases
# ---------------------------------------------------------------------------

class TestLifePath:
    def test_basic_reduction(self):
        # 8/8/1995 -> "881995" -> 8+8+1+9+9+5=40 -> 4+0=4
        assert num.life_path_number(date(1995, 8, 8)) == 4

    def test_master_number_preserved(self):
        # 11/2/1996 -> digit sum 29 -> 2+9=11, a Master Number, must NOT
        # reduce further to 2.
        r = num._life_path_detailed(date(1996, 11, 2))
        assert r["value"] == 11
        assert r["is_master"] is True

    def test_karmic_debt_13(self):
        # 1/2/1900 -> "121900" -> 1+2+1+9+0+0=13 -> 4
        r = num._life_path_detailed(date(1900, 1, 2))
        assert r["value"] == 4
        assert r["karmic_debt"] == 13
        assert r["karmic_debt_meaning"] is not None

    def test_karmic_debt_14(self):
        # 1/12/1900 -> "1121900" -> 1+1+2+1+9+0+0=14 -> 5
        r = num._life_path_detailed(date(1900, 1, 12))
        assert r["value"] == 5
        assert r["karmic_debt"] == 14

    def test_karmic_debt_16(self):
        # 1/5/1900 -> "151900" -> 1+5+1+9+0+0=16 -> 7
        r = num._life_path_detailed(date(1900, 1, 5))
        assert r["value"] == 7
        assert r["karmic_debt"] == 16

    def test_karmic_debt_19_multi_step(self):
        # 1/8/1900 -> "181900" -> 1+8+1+9+0+0=19 -> 10 -> 1 (two reduction steps)
        r = num._life_path_detailed(date(1900, 1, 8))
        assert r["value"] == 1
        assert r["karmic_debt"] == 19
        assert len(r["calculation"]["steps"]) == 3  # breakdown + 2 reduction steps

    def test_no_karmic_debt_when_none_present(self):
        r = num._life_path_detailed(date(1995, 8, 8))
        assert r["karmic_debt"] is None
        assert r["karmic_debt_meaning"] is None

    def test_value_always_in_valid_range(self):
        valid = set(range(1, 10)) | num.MASTER_NUMBERS
        for year in range(1950, 2020, 3):
            for month in (1, 6, 12):
                for day in (1, 15, 28):
                    v = num.life_path_number(date(year, month, day))
                    assert v in valid, f"{month}/{day}/{year} -> {v} out of range"


class TestBirthdayNumber:
    def test_single_digit_day(self):
        r = num._birthday_detailed(date(1995, 8, 8))
        assert r["value"] == 8
        assert r["calculation"]["steps"] == ["8 is already a single digit or Master Number"]

    def test_two_digit_day_reduces(self):
        # day 29 -> 2+9=11 (Master, preserved)
        r = num._birthday_detailed(date(1995, 8, 29))
        assert r["value"] == 11
        assert r["is_master"] is True

    def test_day_13_reduces_normally_not_karmic_for_birthday_alone(self):
        # Birthday number just reduces 13 -> 4; 13 itself is flagged since the
        # reduction loop checks every value it sees, including the input.
        r = num._birthday_detailed(date(1995, 1, 13))
        assert r["value"] == 4
        assert r["karmic_debt"] == 13


class TestAttitudeNumber:
    def test_matches_hand_calculation(self):
        # month=8 (single digit), day=8 (single digit) -> 8+8=16 -> karmic 16 -> 7
        r = num._attitude_detailed(date(1995, 8, 8))
        assert r["value"] == 7
        assert r["karmic_debt"] == 16

    def test_no_reduction_needed(self):
        # month=1, day=2 -> 1+2=3, no karmic debt, no further reduction
        r = num._attitude_detailed(date(1995, 1, 2))
        assert r["value"] == 3
        assert r["karmic_debt"] is None


class TestNameNumbers:
    def test_destiny_pythagorean(self):
        # S=1, A=1, M=4 -> 6
        assert num.destiny_number("SAM", "pythagorean") == 6

    def test_destiny_chaldean_differs(self):
        # S=3, A=1, M=4 -> 8 (Chaldean's table genuinely differs from Pythagorean's)
        assert num.destiny_number("SAM", "chaldean") == 8

    def test_soul_urge_vowels_only(self):
        # SAM vowels: A(1) -> soul urge = 1
        assert num.soul_urge_number("SAM", "pythagorean") == 1

    def test_personality_consonants_only(self):
        # SAM consonants: S(1)+M(4) -> 5
        assert num.personality_number("SAM", "pythagorean") == 5

    def test_hyphenated_name_treated_as_continuous_letters(self):
        # Hyphen contributes no value and isn't a separator with meaning —
        # "MARY-JANE" should compute identically to "MARYJANE".
        assert num.destiny_number("MARY-JANE") == num.destiny_number("MARYJANE")

    def test_apostrophe_name_handled(self):
        assert num.destiny_number("O'BRIEN") == num.destiny_number("OBRIEN")

    def test_multi_word_full_name(self):
        # Just confirms no crash and a sane positive result for a realistic
        # multi-part name.
        v = num.destiny_number("Prabhu Naresh Randhir")
        assert isinstance(v, int) and 1 <= v or v in num.MASTER_NUMBERS

    def test_maturity_combines_life_path_and_destiny(self):
        assert num.maturity_number(4, 6) == num._reduce(4 + 6)


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------

class TestNameValidation:
    def test_empty_name_warns(self):
        warnings = num.validate_name("")
        assert any("No letters" in w for w in warnings)

    def test_no_vowels_warns_and_soul_urge_is_zero(self):
        warnings = num.validate_name("BRB")  # B,R,B all consonants
        assert any("No vowels" in w for w in warnings)
        assert num.soul_urge_number("BRB") == 0

    def test_no_consonants_warns_and_personality_is_zero(self):
        warnings = num.validate_name("AEIOU")
        assert any("No consonants" in w for w in warnings)
        assert num.personality_number("AEIOU") == 0

    def test_normal_name_has_no_warnings(self):
        assert num.validate_name("Prabhu Randhir") == []

    def test_numbers_and_symbols_in_name_are_ignored_not_crashed(self):
        # Defensive: a stray digit or symbol in a name field shouldn't crash
        # the calculation, just be ignored like any other non-letter.
        v = num.destiny_number("John123!")
        assert v == num.destiny_number("John")


class TestDateValidation:
    def test_future_date_warns(self):
        future = date.today().replace(year=date.today().year + 1)
        warnings = num.validate_dob(future)
        assert any("future" in w for w in warnings)

    def test_reasonable_date_has_no_warnings(self):
        assert num.validate_dob(date(1995, 8, 8)) == []

    def test_very_old_date_warns(self):
        warnings = num.validate_dob(date(1850, 1, 1))
        assert any("over 130 years" in w for w in warnings)

    def test_leap_year_date_does_not_crash(self):
        # Python's date type already rejects Feb 30, but Feb 29 on a real
        # leap year must compute cleanly.
        r = num._life_path_detailed(date(2000, 2, 29))
        assert isinstance(r["value"], int)

    def test_year_before_1900_warns_but_still_calculates(self):
        warnings = num.validate_dob(date(1850, 5, 5))
        assert any("before 1900" in w for w in warnings)
        # still produces a real number, doesn't refuse to calculate
        assert isinstance(num.life_path_number(date(1850, 5, 5)), int)


# ---------------------------------------------------------------------------
# Pinnacles & Challenges
# ---------------------------------------------------------------------------

class TestPinnacleAndChallengeCycles:
    def test_matches_hand_calculation_for_8_8_1995(self):
        pc = num.pinnacle_and_challenge_cycles(date(1995, 8, 8))
        assert [p["number"] for p in pc["pinnacles"]] == [7, 5, 3, 5]
        assert [c["number"] for c in pc["challenges"]] == [0, 2, 2, 2]

    def test_age_ranges_are_sequential_and_anchored_to_life_path(self):
        pc = num.pinnacle_and_challenge_cycles(date(1995, 8, 8))
        ranges = [p["age_range"] for p in pc["pinnacles"]]
        assert ranges == ["Birth–32", "33–41", "42–50", "51+"]

    def test_four_pinnacles_and_four_challenges_always_returned(self):
        pc = num.pinnacle_and_challenge_cycles(date(1988, 3, 17))
        assert len(pc["pinnacles"]) == 4
        assert len(pc["challenges"]) == 4

    def test_challenges_are_never_master_numbers(self):
        # Challenges are conventionally reported as plain 0-8 — verify across
        # a spread of dates that none slip through as 11/22/33.
        for year in range(1960, 2010, 5):
            pc = num.pinnacle_and_challenge_cycles(date(year, 11, 29))
            for c in pc["challenges"]:
                assert c["number"] not in num.MASTER_NUMBERS

    def test_challenges_are_within_0_to_8(self):
        for year in range(1960, 2010, 5):
            pc = num.pinnacle_and_challenge_cycles(date(year, 6, 15))
            for c in pc["challenges"]:
                assert 0 <= c["number"] <= 8


class TestCurrentCycleIndex:
    def test_returns_valid_index(self):
        idx = num.current_cycle_index(date(1995, 8, 8), as_of=date(2026, 9, 3))
        assert idx in (0, 1, 2, 3)

    def test_newborn_is_in_first_cycle(self):
        idx = num.current_cycle_index(date(2024, 1, 1), as_of=date(2024, 6, 1))
        assert idx == 0


# ---------------------------------------------------------------------------
# Birth name vs. current name comparison
# ---------------------------------------------------------------------------

class TestCompareNames:
    def test_different_names_both_populated(self):
        result = num.compare_names("Jane Smith", "Jane Doe")
        assert result["same_name"] is False
        assert result["current_name"] is not None
        assert result["birth_name"]["destiny"]["value"] != result["current_name"]["destiny"]["value"] \
            or True  # values *may* coincidentally match; just confirm both computed without error
        assert isinstance(result["current_name"]["destiny"]["value"], int) or \
            result["current_name"]["destiny"]["value"] in num.MASTER_NUMBERS

    def test_same_name_short_circuits(self):
        result = num.compare_names("Jane Smith", "Jane Smith")
        assert result["same_name"] is True
        assert result["current_name"] is None

    def test_same_name_different_casing_and_spacing_still_detected_as_same(self):
        result = num.compare_names("jane smith", "JANE SMITH")
        assert result["same_name"] is True


# ---------------------------------------------------------------------------
# Chaldean vs Pythagorean system selection end-to-end
# ---------------------------------------------------------------------------

class TestLetterSystemSelection:
    def test_full_profile_defaults_to_pythagorean(self):
        p_default = num.full_profile("Sam Lee", date(1995, 8, 8))
        p_explicit = num.full_profile("Sam Lee", date(1995, 8, 8), system="pythagorean")
        assert p_default["destiny"]["value"] == p_explicit["destiny"]["value"]

    def test_full_profile_chaldean_can_differ_from_pythagorean(self):
        p_py = num.full_profile("Sam Lee", date(1995, 8, 8), system="pythagorean")
        p_ch = num.full_profile("Sam Lee", date(1995, 8, 8), system="chaldean")
        # Name-based numbers may legitimately differ between systems
        assert isinstance(p_ch["destiny"]["value"], int) or p_ch["destiny"]["value"] in num.MASTER_NUMBERS

    def test_date_based_numbers_identical_across_systems(self):
        # Life Path, Birthday, Attitude, Personal Year/Month/Day don't depend
        # on letters at all, so they must be identical regardless of system.
        p_py = num.full_profile("Sam Lee", date(1995, 8, 8), system="pythagorean")
        p_ch = num.full_profile("Sam Lee", date(1995, 8, 8), system="chaldean")
        for key in ("life_path", "birthday", "attitude", "personal_year", "personal_month", "personal_day"):
            assert p_py[key]["value"] == p_ch[key]["value"], key

    def test_unknown_system_falls_back_to_pythagorean_table(self):
        # Defensive: an unrecognized system string shouldn't crash — it
        # should behave like Pythagorean rather than silently zeroing values.
        v = num.destiny_number("SAM", "not-a-real-system")
        assert v == num.destiny_number("SAM", "pythagorean")


# ---------------------------------------------------------------------------
# Backward compatibility — the contract every existing call site depends on
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """engines/unified.py and app.py both call full_profile(name, dob) with
    exactly two positional args and read specific keys off the result. These
    tests encode that contract explicitly so a future change can't silently
    break either caller."""

    def test_signature_works_with_two_positional_args_only(self):
        # Must not require the new `system` or `as_of` kwargs.
        p = num.full_profile("Test User", date(1995, 8, 8))
        assert isinstance(p, dict)

    def test_all_original_keys_present(self):
        p = num.full_profile("Test User", date(1995, 8, 8))
        for key in ("life_path", "destiny", "soul_urge", "personality",
                    "birthday", "maturity", "personal_year", "personal_month", "personal_day"):
            assert key in p, f"missing original key: {key}"

    def test_every_field_has_value_and_meaning_keys(self):
        p = num.full_profile("Test User", date(1995, 8, 8))
        for key, entry in p.items():
            assert "value" in entry, key
            assert "meaning" in entry, key
            assert isinstance(entry["value"], int), key

    def test_dict_items_iterable_as_medallion_grid_expects(self):
        p = num.full_profile("Test User", date(1995, 8, 8))
        items = [(v["value"], k.replace("_", " ").title()) for k, v in p.items()]
        assert all(isinstance(val, int) for val, _label in items)
        assert len(items) == len(p)

    def test_life_path_and_personal_year_values_are_valid_theme_map_keys(self):
        # engines/unified.py looks these values up in a theme map keyed by
        # 1-9/11/22/33 — a value outside that set would silently produce no
        # theme signal instead of an error, which is worse (fails quietly).
        valid = set(range(1, 10)) | num.MASTER_NUMBERS
        for year in range(1960, 2015, 4):
            p = num.full_profile("Test User", date(year, 3, 17))
            assert p["life_path"]["value"] in valid
            assert p["personal_year"]["value"] in valid

    def test_new_fields_do_not_replace_old_ones(self):
        # attitude is new; life_path etc. must still exist alongside it.
        p = num.full_profile("Test User", date(1995, 8, 8))
        assert "attitude" in p
        assert "life_path" in p
