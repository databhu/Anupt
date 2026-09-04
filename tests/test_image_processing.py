"""
Tests for utils/image_processing.py.

The preserve_tone tests below encode a real bug found during manual visual
QA: plain per-channel autocontrast could shift a photo's color balance
enough to break the skin-tone heuristic in engines.palmistry's quality
gate — including for photos taken in the exact even, flat lighting this
app's own guidance recommends, not just unusual edge cases. A photo that
passes the quality gate BEFORE optimization must still pass it AFTER,
since optimization is meant to improve a photo, not accidentally cause a
good one to be rejected.
"""

from io import BytesIO

import numpy as np
from PIL import Image

from engines import palmistry
from utils import image_processing


def _flat_lit_skin_tone_image(width=700, height=900, base_rgb=(200, 160, 140)):
    """A synthetic image with the narrow per-channel dynamic range that
    genuinely even, flat lighting produces on a real photo — the exact
    condition that exposed the autocontrast color-shift bug. Includes thin
    darker stripes purely to give the blur detector real edge content to
    measure (a perfectly flat color has none at all, which would fail the
    blur check for an unrelated, correct reason and isn't what this test
    is checking)."""
    arr = np.full((height, width, 3), base_rgb, dtype="uint8")
    darker = tuple(max(0, c - 50) for c in base_rgb)
    arr[::12, :, :] = darker
    arr[:, ::12, :] = darker
    return Image.fromarray(arr)


class TestOptimizeImage:
    def test_upscales_small_images(self):
        small = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype("uint8"))
        out = image_processing.optimize_image(small)
        assert min(out.size) >= image_processing.MIN_SIDE

    def test_downscales_oversized_images(self):
        huge = Image.fromarray((np.random.rand(3000, 4000, 3) * 255).astype("uint8"))
        out = image_processing.optimize_image(huge)
        assert max(out.size) <= image_processing.MAX_SIDE

    def test_converts_rgba_to_rgb(self):
        rgba = Image.fromarray((np.random.rand(1000, 1000, 4) * 255).astype("uint8"), "RGBA")
        out = image_processing.optimize_image(rgba)
        assert out.mode == "RGB"

    def test_never_mutates_input_image(self):
        img = _flat_lit_skin_tone_image()
        original_bytes = img.tobytes()
        image_processing.optimize_image(img)
        assert img.tobytes() == original_bytes


class TestPreserveToneRegression:
    """A photo that passes the palmistry quality gate before optimization
    must still pass it afterward — optimization must not itself cause a
    rejection."""

    def test_flat_lit_skin_tone_image_still_passes_quality_gate_after_optimization(self):
        img = _flat_lit_skin_tone_image()
        assert palmistry.assess_image(img)["passed"] is True  # sanity: passes before optimization

        optimized = image_processing.optimize_image(img)
        result = palmistry.assess_image(optimized)
        assert result["passed"] is True, f"optimization caused a false rejection: {result['issues']}"

    def test_skin_tone_ratio_stays_above_the_quality_gates_own_threshold(self):
        # The specific failure mode found: preserve_tone=False could drop
        # skin_tone_ratio all the way to 0.0 even when the pre-optimization
        # ratio was comfortably passing. The bar that actually matters is
        # the quality gate's own pass threshold, not an arbitrary number.
        img = _flat_lit_skin_tone_image()
        before = palmistry.assess_image(img)["skin_tone_ratio"]
        after = palmistry.assess_image(image_processing.optimize_image(img))["skin_tone_ratio"]
        assert before > palmistry.MIN_SKIN_TONE_RATIO
        assert after > palmistry.MIN_SKIN_TONE_RATIO

    def test_several_realistic_skin_tone_variants_all_survive_optimization(self):
        # A spread of plausible skin-tone base colors under flat lighting —
        # not just one specific RGB triple.
        for base_rgb in [(200, 160, 140), (180, 140, 115), (150, 110, 90), (220, 185, 165)]:
            img = _flat_lit_skin_tone_image(base_rgb=base_rgb)
            optimized = image_processing.optimize_image(img)
            result = palmistry.assess_image(optimized)
            assert result["passed"] is True, f"{base_rgb} failed after optimization: {result['issues']}"


class TestJpegEncoding:
    def test_to_jpeg_bytes_produces_valid_jpeg(self):
        img = _flat_lit_skin_tone_image()
        data = image_processing.to_jpeg_bytes(img)
        reloaded = Image.open(BytesIO(data))
        assert reloaded.format == "JPEG"

    def test_optimize_to_jpeg_bytes_end_to_end(self):
        img = _flat_lit_skin_tone_image()
        data = image_processing.optimize_to_jpeg_bytes(img)
        reloaded = Image.open(BytesIO(data))
        assert reloaded.format == "JPEG"
        assert min(reloaded.size) >= image_processing.MIN_SIDE
