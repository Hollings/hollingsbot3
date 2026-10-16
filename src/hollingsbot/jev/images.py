"""Bounded Jev image input; image tokens are paid again on every word request."""

from __future__ import annotations

import base64
import binascii
import io

from PIL import Image, ImageOps

# Stop between words after this total reply spend (text and any suggester included).
# A word's in-flight requests can take the reply slightly over this threshold.
IMAGE_REPLY_BUDGET = 0.01


def prepare_image(data_url: str) -> str | None:
    """A single upright, baseline JPEG on a 512px canvas, or None if unreadable.

    Padding preserves the aspect ratio and meets Jev's minimum side/area limits,
    including for small emoji and tall screenshots. No remote URLs are fetched.
    """
    if not isinstance(data_url, str) or len(data_url) > 24 * 1024 * 1024:
        return None
    if not data_url.startswith("data:image/") or ";base64," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(";base64,", 1)[1], validate=True)
        if len(raw) > 16 * 1024 * 1024:
            return None
        with Image.open(io.BytesIO(raw)) as original:
            if original.width * original.height > 20_000_000:
                return None
            image = ImageOps.exif_transpose(original).convert("RGBA")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (512, 512), "white")
            canvas.paste(image, ((512 - image.width) // 2, (512 - image.height) // 2), image)
            output = io.BytesIO()
            canvas.save(output, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
        return None
