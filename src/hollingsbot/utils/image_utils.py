from __future__ import annotations

from io import BytesIO
from typing import Tuple

from PIL import Image


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB, flattening any alpha channel against white."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "LA":
            rgb = img.convert("RGBA")
            bg.paste(rgb, mask=rgb.split()[-1])
        else:
            bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def _resize(im: Image.Image, scale: float) -> Image.Image:
    if scale >= 0.999:
        return im
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    return im.resize((nw, nh), Image.LANCZOS)


def compress_image_to_fit(
    image_bytes: bytes,
    limit_bytes: int,
    *,
    target_margin: float = 0.98,
    prefer_jpeg: bool = True,
) -> Tuple[bytes, str]:
    """
    Compress image bytes to fit under Discord's per-file size limit.

    Strategy:
    - Decode with PIL and convert to RGB (flatten transparency to white).
    - Try a grid of (scale, quality) combinations saving as JPEG.
    - Return the first candidate under `limit_bytes * target_margin`.
    - If all candidates exceed the limit, return the smallest tried.

    Returns (data, ext) where `ext` is the suggested file extension ("jpg" or "png").
    """
    if limit_bytes <= 0:
        # Nothing to do, return original
        return image_bytes, "png"

    target = int(limit_bytes * max(0.5, min(1.0, target_margin)))

    try:
        base = Image.open(BytesIO(image_bytes))
        base.load()
    except Exception:
        # If decoding fails, just return original
        return image_bytes, "png"

    # Prefer JPEG for strong size reductions
    if prefer_jpeg:
        rgb = _flatten_to_rgb(base)
        # Start near original size, then progressively reduce
        scales = [1.0, 0.95, 0.9, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15]
        qualities = [85, 75, 65, 55, 50, 45, 40, 35, 30, 25, 20]

        best_data = None
        best_size = None

        for s in scales:
            im = _resize(rgb, s)
            for q in qualities:
                buf = BytesIO()
                try:
                    im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
                except Exception:
                    buf.seek(0); buf.truncate(0)
                    im.save(buf, format="JPEG", quality=q)
                data = buf.getvalue()
                if best_size is None or len(data) < best_size:
                    best_data, best_size = data, len(data)
                if len(data) <= target:
                    return data, "jpg"

        # If nothing fit under target, return smallest attempt
        if best_data is not None:
            return best_data, "jpg"

    # Fallback: try PNG optimize + palette quantization (last resort)
    try:
        pal = base.convert("P", palette=Image.ADAPTIVE, colors=256)
        buf = BytesIO()
        pal.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= target:
            return data, "png"
        # Return this attempt if it's smaller than original
        if len(data) < len(image_bytes):
            return data, "png"
    except Exception:
        pass

    # Give up; return original
    return image_bytes, "png"

