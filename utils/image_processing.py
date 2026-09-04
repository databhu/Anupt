"""
ANUPT — image optimization for palm photos.

Honesty note, matching the rest of this app's stance on palmistry: this is
classical image processing (Lanczos resampling, unsharp mask, autocontrast)
— it makes a photo more consistent and pleasant to view, and it's genuinely
true that Lanczos upscaling produces smoother edges than a phone's own
nearest-neighbor scaling in some viewers. It is NOT AI super-resolution and
does not invent detail that wasn't in the original photo. A blurry photo
upscaled this way is still a blurry photo — which is exactly why the
deterministic quality gate (engines/palmistry.py) runs on the OPTIMIZED
image and still rejects it if the underlying capture was genuinely bad.
"""

from io import BytesIO

from PIL import Image, ImageFilter, ImageOps

MIN_SIDE = 900     # upscale if the shorter side is below this
MAX_SIDE = 2200    # downscale if the longer side exceeds this (keeps storage/bandwidth sane)
JPEG_QUALITY = 90


def optimize_image(pil_image: Image.Image) -> Image.Image:
    """Normalize orientation, upscale small photos, downscale oversized ones,
    and apply a mild sharpen + autocontrast pass. Always returns an RGB image."""
    img = ImageOps.exif_transpose(pil_image)  # respect phone camera orientation metadata
    img = img.convert("RGB")

    w, h = img.size
    short_side, long_side = min(w, h), max(w, h)

    if short_side < MIN_SIDE:
        scale = MIN_SIDE / short_side
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    elif long_side > MAX_SIDE:
        scale = MAX_SIDE / long_side
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    # Mild, not aggressive — enough to counter typical phone-camera softness
    # without introducing halos or fake-looking crunch on skin/line detail.
    img = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=60, threshold=3))
    # preserve_tone=True keeps the relative balance between R/G/B channels
    # while still expanding contrast — plain per-channel autocontrast can
    # shift color balance enough to break skin-tone detection downstream
    # (engines.palmistry's quality gate runs on this optimized output),
    # including for genuinely good photos: even, flat lighting — the exact
    # condition this app recommends — produces the narrow per-channel
    # dynamic range most likely to trigger that shift.
    img = ImageOps.autocontrast(img, cutoff=1, preserve_tone=True)
    return img


def optimize_to_jpeg_bytes(pil_image: Image.Image) -> bytes:
    """optimize_image() then re-encode as a reasonably-sized JPEG, for storage."""
    return to_jpeg_bytes(optimize_image(pil_image))


def to_jpeg_bytes(pil_image: Image.Image) -> bytes:
    """Encode an already-processed image to JPEG bytes — split out from
    optimize_to_jpeg_bytes() so a caller that already has the optimized PIL
    image in hand (e.g. to show a preview) doesn't have to run optimize_image()
    a second time just to get storage-ready bytes."""
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()
