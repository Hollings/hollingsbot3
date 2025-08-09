from __future__ import annotations
import io
import re
import logging
from typing import List, Tuple

# Try to enable SVG rendering; degrade gracefully if native Cairo is missing.
try:
    import cairosvg  # type: ignore
    _SVG_RENDERING_AVAILABLE = True
except Exception as e:  # pragma: no cover
    cairosvg = None  # type: ignore
    _SVG_RENDERING_AVAILABLE = False
    logging.getLogger(__name__).warning("CairoSVG unavailable; will attach .svg instead of PNG. %s", e)

SVG_BLOCK_RE = re.compile(r"```svg\s*([\s\S]*?)```", re.IGNORECASE)


def svg_rendering_available() -> bool:
    return _SVG_RENDERING_AVAILABLE


def _render_svg_to_png_bytes(svg_xml: str) -> bytes:
    if not _SVG_RENDERING_AVAILABLE:
        raise RuntimeError("CairoSVG not available")
    return cairosvg.svg2png(bytestring=svg_xml.encode("utf-8"))  # type: ignore[attr-defined]


def extract_render_and_strip_svgs(text: str) -> Tuple[str, List[Tuple[str, io.BytesIO]]]:
    """
    Find all ```svg ... ``` code blocks in `text`, render each to a PNG when possible,
    and replace each block with a short note placeholder BEFORE any other code-block/file extraction.

    Returns:
      cleaned_text: original text with svg blocks replaced by notes
      svg_files: list of (filename, BytesIO) for sending as attachments (PNG if possible, otherwise raw .svg)
    """
    svg_files: List[Tuple[str, io.BytesIO]] = []
    idx = 0

    def _sub(match: re.Match) -> str:
        nonlocal idx
        idx += 1
        svg_xml = match.group(1).strip()

        if _SVG_RENDERING_AVAILABLE:
            filename = f"svg_render_{idx}.png"
            try:
                png_bytes = _render_svg_to_png_bytes(svg_xml)
                buf = io.BytesIO(png_bytes)
                buf.seek(0)
                svg_files.append((filename, buf))
                return f"[SVG rendered and attached: {filename}]"
            except Exception as e:
                fallback = f"svg_render_{idx}.svg"
                b = io.BytesIO(svg_xml.encode("utf-8"))
                b.seek(0)
                svg_files.append((fallback, b))
                return f"[SVG render failed, attached original SVG: {fallback} — {e}]"
        else:
            filename = f"svg_render_{idx}.svg"
            b = io.BytesIO(svg_xml.encode("utf-8"))
            b.seek(0)
            svg_files.append((filename, b))
            return f"[SVG attached (SVG format) because rendering is unavailable: {filename}]"

    cleaned_text = SVG_BLOCK_RE.sub(_sub, text)
    return cleaned_text, svg_files
