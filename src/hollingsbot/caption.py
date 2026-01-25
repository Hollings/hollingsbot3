from __future__ import annotations

import textwrap
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    """Return a truetype font or fall back to the default."""
    font_candidates = (
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def calculate_font_size(caption: str) -> int:
    """Return a font size based on caption length."""
    if len(caption) < 100:
        return 40
    if len(caption) < 200:
        return 30
    return 20


def _add_caption(img: Image.Image, caption: str) -> Image.Image:
    """Return a new image with the caption above the original image."""
    font_size = calculate_font_size(caption)
    font = _load_font(font_size)

    margin = 20
    max_width = img.width - 2 * margin

    temp_img = Image.new("RGB", (img.width, img.height), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_img)

    wrap_count = 100
    wrapped = textwrap.fill(caption, width=wrap_count)
    bbox = temp_draw.textbbox((0, 0), wrapped, font=font)
    while bbox[2] > max_width and wrap_count > 1:
        wrap_count -= 1
        wrapped = textwrap.fill(caption, width=wrap_count)
        bbox = temp_draw.textbbox((0, 0), wrapped, font=font)

    caption_height = bbox[3] + 2 * margin
    new_img = Image.new("RGB", (img.width, img.height + caption_height), (255, 255, 255))
    new_img.paste(img, (0, caption_height))

    draw = ImageDraw.Draw(new_img)
    text_width = bbox[2] - bbox[0]
    x = (new_img.width - text_width) // 2
    y = margin
    draw.text((x, y), wrapped, fill="black", font=font)

    return new_img


def add_caption(image_bytes: bytes, text: str) -> bytes:
    """Return image bytes with a caption rendered above the original."""
    if not text:
        return image_bytes

    buf = BytesIO(image_bytes)
    img = Image.open(buf)
    img.load()  # <-- make pixel data independent of `buf`
    img = img.convert("RGB")  # afterwards you can safely close the buffer
    buf.close()

    new_img = _add_caption(img, text)

    out = BytesIO()
    new_img.save(out, format="PNG")
    out.seek(0)  # Discord will read from the start
    return out.read()
