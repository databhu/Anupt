"""
ANUPT — Palm reading image annotation.

Draws the "VISUAL PROOF" step of IMAGE -> DETECTION -> VISUAL PROOF ->
TRADITIONAL INTERPRETATION -> PERSONALIZED READING directly onto a copy
of the user's own photo: a box and label at each finding's approximate
location, color-coded by feature category.

Deliberately conservative about what gets drawn:
- Only findings with a usable bbox (already validated as in-range 0-1
  coordinates by ai/gemini_client.py's parsing) are drawn at all.
- Low-confidence findings are skipped by default — a box on the photo
  reads as "the AI is confident this is here," so a Low-confidence guess
  gets a place in the written findings list, not a confident-looking box
  on the image itself.
Never modifies the original photo — always returns new bytes.
"""

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

_CATEGORY_COLORS = {
    "line": (194, 21, 126),    # magenta — matches the app's brand line-color usage
    "mount": (76, 31, 147),    # indigo
    "shape": (140, 101, 48),   # gold-text
}
_DEFAULT_COLOR = (194, 21, 126)


def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older Pillow: load_default() takes no size argument.
        return ImageFont.load_default()


def annotate_palm_image(image_bytes: bytes, findings: list, min_confidence: str = "Medium") -> tuple[bytes, int]:
    """Returns (new JPEG bytes, count of boxes actually drawn). Draws a box +
    label for every finding that has a usable bbox and meets `min_confidence`
    ("Low" draws everything with a bbox; "Medium" — the default — skips
    Low-confidence guesses; "High" draws only the most certain findings).
    The count matters to the caller: zero boxes drawn (e.g. every finding
    lacked a localizable bbox) should show a plain "no boxes could be
    placed" message rather than an annotated-looking image with nothing on it."""
    confidence_rank = {"Low": 0, "Medium": 1, "High": 2}
    min_rank = confidence_rank.get(min_confidence, 1)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font_size = max(14, round(min(w, h) * 0.028))
    font = _load_font(font_size)

    drawn_count = 0
    for f in findings:
        bbox = f.get("bbox")
        if not bbox:
            continue
        if confidence_rank.get(f.get("confidence", "Low"), 0) < min_rank:
            continue

        x0, y0, x1, y1 = bbox
        box = (x0 * w, y0 * h, x1 * w, y1 * h)
        color = _CATEGORY_COLORS.get(f.get("category"), _DEFAULT_COLOR)

        draw.rectangle(box, outline=color, width=max(2, round(min(w, h) * 0.004)))

        label = f.get("feature", "")
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        label_y = box[1] - text_h - 6 if box[1] - text_h - 6 > 0 else box[1] + 4
        draw.rectangle(
            (box[0], label_y, box[0] + text_w + 10, label_y + text_h + 6),
            fill=color,
        )
        draw.text((box[0] + 5, label_y + 2), label, fill=(255, 255, 255), font=font)
        drawn_count += 1

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), drawn_count
