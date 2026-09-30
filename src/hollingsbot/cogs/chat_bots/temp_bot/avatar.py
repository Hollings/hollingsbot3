"""Small local avatar used while an AI-generated temp-bot portrait is unavailable."""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageDraw, ImageFont


def fallback_avatar(name: str) -> bytes:
    """Make a deterministic PNG monogram so every temp bot has an avatar."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    background = tuple(40 + value // 4 for value in digest[:3])
    initials = "".join(word[0].upper() for word in name.split()[:2]) or "?"

    image = Image.new("RGB", (256, 256), background)
    draw = ImageDraw.Draw(image)
    draw.ellipse((18, 18, 238, 238), outline="white", width=5)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
    except OSError:
        font = ImageFont.load_default()
    draw.text((128, 128), initials, fill="white", font=font, anchor="mm")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
