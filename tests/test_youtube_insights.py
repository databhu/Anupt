"""
Tests for engines/youtube_insights.py — the deterministic Analysis,
Zodiac-organization, and Evidence stages of the YouTube Insights
pipeline. The core properties under test: hallucinated signs/themes from
the AI extraction step are filtered out (never trusted structurally),
and the theme-agreement ranking genuinely reflects how many DISTINCT
sources support each theme, not just how many times a theme is mentioned.
"""

from engines import youtube_insights as yi


class TestValidateExtractedPrediction:
    def test_valid_entry_passes_through(self):
        result = yi.validate_extracted_prediction({
            "video_id": "abc", "signs_mentioned": ["Aries"], "themes": ["career"],
            "prediction_summary": "Career growth ahead.",
        })
        assert result["video_id"] == "abc"
        assert result["signs_mentioned"] == ["Aries"]
        assert result["themes"] == ["career"]

    def test_hallucinated_sign_is_dropped_valid_ones_kept(self):
        result = yi.validate_extracted_prediction({
            "video_id": "abc", "signs_mentioned": ["Aries", "NotARealSign"], "themes": [],
        })
        assert result["signs_mentioned"] == ["Aries"]

    def test_hallucinated_theme_is_dropped_valid_ones_kept(self):
        result = yi.validate_extracted_prediction({
            "video_id": "abc", "signs_mentioned": ["Leo"], "themes": ["career", "madeUpTheme"],
        })
        assert result["themes"] == ["career"]

    def test_entry_with_zero_valid_signs_is_rejected_entirely(self):
        assert yi.validate_extracted_prediction({
            "video_id": "abc", "signs_mentioned": ["NotASign", "AlsoNotASign"], "themes": ["career"],
        }) is None

    def test_missing_video_id_rejected(self):
        assert yi.validate_extracted_prediction({"signs_mentioned": ["Aries"]}) is None

    def test_no_themes_at_all_defaults_to_general(self):
        result = yi.validate_extracted_prediction({"video_id": "abc", "signs_mentioned": ["Aries"], "themes": []})
        assert result["themes"] == ["general"]

    def test_malformed_input_types_do_not_crash(self):
        assert yi.validate_extracted_prediction("not a dict") is None
        assert yi.validate_extracted_prediction(None) is None
        assert yi.validate_extracted_prediction([]) is None
        assert yi.validate_extracted_prediction({"video_id": "x", "signs_mentioned": "not a list"}) is None
        assert yi.validate_extracted_prediction({"video_id": "x", "signs_mentioned": ["Aries"], "themes": "nope"}) is not None

    def test_every_zodiac_sign_is_a_valid_target(self):
        for sign in yi.ZODIAC_SIGNS:
            result = yi.validate_extracted_prediction({"video_id": "x", "signs_mentioned": [sign], "themes": []})
            assert result is not None, sign


class TestAnalyzePredictions:
    def _sample(self):
        return [
            {"video_id": "v1", "signs_mentioned": ["Aries"], "themes": ["career", "love"]},
            {"video_id": "v2", "signs_mentioned": ["Aries", "Taurus"], "themes": ["career"]},
            {"video_id": "v3", "signs_mentioned": ["Aries"], "themes": ["career", "finance"]},
            {"video_id": "v4", "signs_mentioned": ["Taurus"], "themes": ["career"]},  # not Aries — must be excluded
        ]

    def test_filters_to_only_the_target_sign(self):
        result = yi.analyze_predictions(self._sample(), "Aries")
        assert result["relevant_prediction_count"] == 3  # v1, v2, v3 — not v4

    def test_theme_agreement_counts_distinct_videos_not_raw_mentions(self):
        result = yi.analyze_predictions(self._sample(), "Aries")
        assert result["theme_agreement"]["career"] == 3  # v1, v2, v3 all mention career
        assert result["theme_agreement"]["love"] == 1
        assert result["theme_agreement"]["finance"] == 1

    def test_ranked_themes_puts_highest_agreement_first(self):
        result = yi.analyze_predictions(self._sample(), "Aries")
        assert result["ranked_themes"][0] == "career"

    def test_no_matching_videos_returns_empty_not_crash(self):
        result = yi.analyze_predictions(self._sample(), "Gemini")
        assert result["relevant_prediction_count"] == 0
        assert result["ranked_themes"] == []

    def test_multi_sign_video_counts_for_each_sign_independently(self):
        # v2 mentions both Aries and Taurus — it should count toward BOTH
        # sign's analysis independently, not "used up" by one.
        aries_result = yi.analyze_predictions(self._sample(), "Aries")
        taurus_result = yi.analyze_predictions(self._sample(), "Taurus")
        assert any(p["video_id"] == "v2" for p in aries_result["relevant_predictions"])
        assert any(p["video_id"] == "v2" for p in taurus_result["relevant_predictions"])


class TestOrganizeByTheme:
    def test_groups_predictions_under_each_theme_they_raise(self):
        analysis = yi.analyze_predictions([
            {"video_id": "v1", "signs_mentioned": ["Leo"], "themes": ["career", "love"]},
        ], "Leo")
        organized = yi.organize_by_theme(analysis)
        assert "career" in organized and "love" in organized
        assert organized["career"][0]["video_id"] == "v1"
        assert organized["love"][0]["video_id"] == "v1"

    def test_themes_with_no_predictions_are_absent_not_empty_list(self):
        analysis = yi.analyze_predictions([
            {"video_id": "v1", "signs_mentioned": ["Leo"], "themes": ["career"]},
        ], "Leo")
        organized = yi.organize_by_theme(analysis)
        assert "health" not in organized  # never mentioned, shouldn't appear at all


class TestAttachEvidence:
    def test_every_prediction_linked_to_its_real_source_video(self):
        videos = [{"video_id": "v1", "title": "T1", "channel_title": "C1",
                   "url": "https://youtube.com/v1", "published_at": "2026-01-01"}]
        organized = {"career": [{"video_id": "v1", "prediction_summary": "Growth ahead."}]}
        evidence = yi.attach_evidence(organized, videos)
        assert evidence["career"][0]["source_title"] == "T1"
        assert evidence["career"][0]["source_url"] == "https://youtube.com/v1"

    def test_prediction_referencing_unknown_video_is_dropped_not_crashed(self):
        # Defensive: a prediction whose video isn't in the source list
        # (shouldn't normally happen, but must not crash if it does).
        organized = {"career": [{"video_id": "ghost", "prediction_summary": "x"}]}
        evidence = yi.attach_evidence(organized, [])
        assert "career" not in evidence  # no valid entries -> theme omitted entirely

    def test_duplicate_video_within_same_theme_only_counted_once(self):
        videos = [{"video_id": "v1", "title": "T1", "channel_title": "C1",
                   "url": "https://youtube.com/v1", "published_at": "2026-01-01"}]
        organized = {"career": [
            {"video_id": "v1", "prediction_summary": "First mention."},
            {"video_id": "v1", "prediction_summary": "Duplicate mention."},
        ]}
        evidence = yi.attach_evidence(organized, videos)
        assert len(evidence["career"]) == 1


class TestFullPipeline:
    def test_end_to_end_realistic_scenario(self):
        videos = [
            {"video_id": "v1", "title": "Aries Weekly", "channel_title": "AstroChannel",
             "url": "https://youtube.com/v1", "published_at": "2026-09-01"},
            {"video_id": "v2", "title": "All Signs", "channel_title": "StarGuide",
             "url": "https://youtube.com/v2", "published_at": "2026-09-02"},
            {"video_id": "v3", "title": "Aries Career", "channel_title": "ZodiacDaily",
             "url": "https://youtube.com/v3", "published_at": "2026-09-03"},
        ]
        extracted = [
            {"video_id": "v1", "signs_mentioned": ["Aries"], "themes": ["career", "love"],
             "prediction_summary": "Career growth this week."},
            {"video_id": "v2", "signs_mentioned": ["Aries", "Taurus"], "themes": ["career"],
             "prediction_summary": "Career shifts likely."},
            {"video_id": "v3", "signs_mentioned": ["Aries"], "themes": ["career", "finance"],
             "prediction_summary": "Financial opportunity tied to career."},
        ]
        evidence = yi.build_insights_evidence(extracted, videos, "Aries")

        assert evidence["relevant_prediction_count"] == 3
        assert evidence["ranked_themes"][0] == "career"
        assert evidence["theme_agreement"]["career"] == 3
        assert len(evidence["evidence_by_theme"]["career"]) == 3
        source_urls = {e["source_url"] for e in evidence["evidence_by_theme"]["career"]}
        assert source_urls == {"https://youtube.com/v1", "https://youtube.com/v2", "https://youtube.com/v3"}

    def test_no_crash_with_empty_extraction(self):
        evidence = yi.build_insights_evidence([], [], "Leo")
        assert evidence["relevant_prediction_count"] == 0
        assert evidence["evidence_by_theme"] == {}
