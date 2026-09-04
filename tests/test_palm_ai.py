"""
Tests for ai/gemini_client.py's structured palm reading functions.

The core property under test throughout: a response that violates the
closed feature vocabulary, has out-of-range coordinates, or isn't valid
JSON at all must never be silently trusted — it gets filtered or rejected,
never displayed as if it were a legitimate finding. This is the "never
fabricate a line or feature that is not clearly visible" requirement
enforced in code, not just in a prompt instruction the model might ignore.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

logging.disable(logging.CRITICAL)

import os
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")

from ai import gemini_client as gc
from engines import palmistry


def _gemini_response(text: str):
    r = MagicMock()
    r.status_code = 200
    r.json = lambda: {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return r


def _valid_payload(**overrides) -> str:
    base = {
        "image_quality_sufficient": True,
        "quality_issue": None,
        "findings": [
            {"feature": "Life Line", "confidence": "High", "location_description": "curves around thumb base",
             "bbox": [0.1, 0.3, 0.35, 0.8], "markings": [], "note": "deep and continuous"},
        ],
        "life_areas": {area: {"narrative": "n/a", "linked_features": []} for area in palmistry.LIFE_AREAS},
        "score": 7,
        "summary": "A grounded, steady reading overall.",
    }
    base.update(overrides)
    return json.dumps(base)


class TestStructuredPalmReadingParsing:
    def test_well_formed_response_parses_correctly(self):
        with patch("requests.post", return_value=_gemini_response(_valid_payload())):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["image_quality_sufficient"] is True
        assert len(result["findings"]) == 1
        assert result["findings"][0]["feature"] == "Life Line"
        assert result["findings"][0]["category"] == "line"  # enriched from PALM_FEATURES, not the model
        assert result["score"] == 7

    def test_markdown_code_fence_is_stripped(self):
        fenced = f"```json\n{_valid_payload()}\n```"
        with patch("requests.post", return_value=_gemini_response(fenced)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert "error" not in result
        assert result["score"] == 7

    def test_stray_text_around_json_is_handled(self):
        wrapped = f"Here is my analysis:\n{_valid_payload()}\nLet me know if you have questions!"
        with patch("requests.post", return_value=_gemini_response(wrapped)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert "error" not in result
        assert result["score"] == 7

    def test_totally_non_json_response_returns_error_not_crash(self):
        with patch("requests.post", return_value=_gemini_response("I cannot help with that.")):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert "error" in result

    def test_malformed_json_returns_error_not_crash(self):
        with patch("requests.post", return_value=_gemini_response("{not: valid json,,,}")):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert "error" in result

    def test_missing_api_key_returns_error_without_calling_api(self):
        with patch.dict(os.environ, {}, clear=True), patch("requests.post") as mock_post:
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert "error" in result
        mock_post.assert_not_called()


class TestHallucinationFiltering:
    """The single most important property: a feature name outside the
    closed vocabulary must never reach the UI, no matter how confidently
    or plausibly the model presents it."""

    def test_hallucinated_feature_name_is_dropped(self):
        payload = _valid_payload(findings=[
            {"feature": "Life Line", "confidence": "High", "location_description": "x",
             "bbox": None, "markings": [], "note": "real one"},
            {"feature": "Line of Cosmic Destiny", "confidence": "High", "location_description": "made up",
             "bbox": None, "markings": [], "note": "fabricated"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        names = [f["feature"] for f in result["findings"]]
        assert "Line of Cosmic Destiny" not in names
        assert "Life Line" in names
        assert len(result["findings"]) == 1

    def test_all_hallucinated_findings_yields_empty_list_not_error(self):
        payload = _valid_payload(findings=[
            {"feature": "Not A Real Feature", "confidence": "High", "location_description": "x",
             "bbox": None, "markings": [], "note": "x"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["findings"] == []
        assert "error" not in result

    def test_hallucinated_marking_type_is_dropped_but_feature_kept(self):
        payload = _valid_payload(findings=[
            {"feature": "Heart Line", "confidence": "Medium", "location_description": "x",
             "bbox": None, "markings": ["Island", "Made-up Squiggle"], "note": "x"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["findings"][0]["markings"] == ["Island"]

    def test_hallucinated_linked_feature_in_life_area_is_dropped(self):
        payload = _valid_payload()
        data = json.loads(payload)
        data["life_areas"]["career"]["linked_features"] = ["Head Line", "Nonexistent Line"]
        with patch("requests.post", return_value=_gemini_response(json.dumps(data))):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["life_areas"]["career"]["linked_features"] == ["Head Line"]

    def test_invalid_confidence_value_defaults_to_low_not_dropped(self):
        # An out-of-vocabulary confidence shouldn't silently become "High" —
        # defaulting to the most conservative value is the safe direction.
        payload = _valid_payload(findings=[
            {"feature": "Life Line", "confidence": "Extremely Certain", "location_description": "x",
             "bbox": None, "markings": [], "note": "x"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["findings"][0]["confidence"] == "Low"


class TestBoundingBoxValidation:
    def test_valid_bbox_is_kept(self):
        payload = _valid_payload(findings=[
            {"feature": "Life Line", "confidence": "High", "location_description": "x",
             "bbox": [0.1, 0.2, 0.5, 0.9], "markings": [], "note": "x"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["findings"][0]["bbox"] == [0.1, 0.2, 0.5, 0.9]

    def test_out_of_range_bbox_is_nulled(self):
        payload = _valid_payload(findings=[
            {"feature": "Life Line", "confidence": "High", "location_description": "x",
             "bbox": [0.1, 0.2, 1.5, 0.9], "markings": [], "note": "x"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["findings"][0]["bbox"] is None

    def test_malformed_bbox_shape_is_nulled(self):
        payload = _valid_payload(findings=[
            {"feature": "Life Line", "confidence": "High", "location_description": "x",
             "bbox": [0.1, 0.2], "markings": [], "note": "x"},  # only 2 values, not 4
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["findings"][0]["bbox"] is None

    def test_null_bbox_stays_null(self):
        payload = _valid_payload(findings=[
            {"feature": "Life Line", "confidence": "Low", "location_description": "x",
             "bbox": None, "markings": [], "note": "faint, hard to localize precisely"},
        ])
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["findings"][0]["bbox"] is None


class TestPoorImageQualityHandling:
    def test_model_flagged_poor_quality_is_respected(self):
        payload = json.dumps({
            "image_quality_sufficient": False,
            "quality_issue": "The photo is too blurry to trace individual lines.",
            "findings": [], "life_areas": {}, "score": None,
            "summary": "Cannot give a reliable reading from this photo.",
        })
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["image_quality_sufficient"] is False
        assert result["score"] is None
        assert result["findings"] == []
        assert "blurry" in result["quality_issue"]

    def test_score_is_clamped_to_1_10(self):
        payload = _valid_payload(score=15)
        with patch("requests.post", return_value=_gemini_response(payload)):
            result = gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left")
        assert result["score"] == 10


class TestPromptConstruction:
    def test_vocabulary_block_includes_every_feature_name(self):
        block = gc._palm_vocabulary_prompt_block()
        for name in palmistry.PALM_FEATURES:
            assert name in block

    def test_marathi_instruction_included_when_requested(self):
        with patch("requests.post", return_value=_gemini_response(_valid_payload())) as mock_post:
            gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left", language="mr")
        sent_prompt = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        assert "Marathi" in sent_prompt

    def test_question_included_when_provided(self):
        with patch("requests.post", return_value=_gemini_response(_valid_payload())) as mock_post:
            gc.palm_vision_reading_structured(b"fake", "image/jpeg", "left", question="Will I be wealthy?")
        sent_prompt = mock_post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        assert "Will I be wealthy?" in sent_prompt


class TestComparisonReading:
    def test_well_formed_comparison_parses(self):
        comp = json.dumps({
            "narrative": "The dominant hand shows a more defined Head Line.",
            "differences": [{"feature": "Head Line", "dominant_note": "deep", "non_dominant_note": "lighter",
                             "interpretation": "Conscious thinking has sharpened."}],
        })
        with patch("requests.post", return_value=_gemini_response(comp)):
            result = gc.palm_comparison_reading(
                {"findings": [{"feature": "Head Line"}]}, {"findings": [{"feature": "Head Line"}]}, "Right"
            )
        assert "narrative" in result
        assert len(result["differences"]) == 1

    def test_malformed_comparison_returns_error(self):
        with patch("requests.post", return_value=_gemini_response("not json")):
            result = gc.palm_comparison_reading({"findings": []}, {"findings": []}, "Right")
        assert "error" in result

    def test_missing_key_returns_error_without_api_call(self):
        with patch.dict(os.environ, {}, clear=True), patch("requests.post") as mock_post:
            result = gc.palm_comparison_reading({"findings": []}, {"findings": []}, "Right")
        assert "error" in result
        mock_post.assert_not_called()
