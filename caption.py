from __future__ import annotations

from io import BytesIO
from typing import List

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    """Return a truetype font or fall back to the default."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def add_caption(image_bytes: bytes, text: str) -> bytes:
    """Return image bytes with a caption added above the image."""
    if not text:
        return image_bytes

    with Image.open(BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        width = im.width
        draw = ImageDraw.Draw(im)
        max_font = 40
        min_font = 12
        padding = 10
        for size in range(max_font, min_font - 1, -2):
            font = _load_font(size)
            lines = _wrap_text(draw, text, font, int(width * 0.95))
            bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font)
            if bbox[2] <= width:
                break
        else:
            font = _load_font(min_font)
            lines = _wrap_text(draw, text, font, int(width * 0.95))
            bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font)
        text_height = bbox[3] - bbox[1]
        caption_height = text_height + 2 * padding

        caption_img = Image.new("RGB", (width, caption_height), "white")
        caption_draw = ImageDraw.Draw(caption_img)
        caption_draw.multiline_text(
            (width // 2, padding),
            "\n".join(lines),
            font=font,
            fill="black",
            align="center",
            anchor="ma",
        )

        new_img = Image.new("RGB", (width, caption_height + im.height), "white")
        new_img.paste(caption_img, (0, 0))
        new_img.paste(im, (0, caption_height))
        out = BytesIO()
        new_img.save(out, format="PNG")
        return out.getvalue()
