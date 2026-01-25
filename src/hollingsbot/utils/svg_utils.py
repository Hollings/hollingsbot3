# /mnt/data/svg_utils.py
from __future__ import annotations
import io
import re
import logging
from typing import List, Tuple
import html

# Try to enable SVG rendering; degrade gracefully if native Cairo is missing.
try:
    import cairosvg  # type: ignore
    _SVG_RENDERING_AVAILABLE = True
except Exception as e:  # pragma: no cover
    cairosvg = None  # type: ignore
    _SVG_RENDERING_AVAILABLE = False
    logging.getLogger(__name__).warning("CairoSVG unavailable; will attach .svg instead of PNG. %s", e)

# Old behavior: fenced code blocks ```svg ... ```
SVG_BLOCK_RE = re.compile(r"```svg\s*([\s\S]*?)```", re.IGNORECASE)

# New behavior: find any <svg ...> ... </svg> anywhere, even inside non-labeled code blocks
INLINE_SVG_RE = re.compile(r"(?is)<svg\b[^>]*>[\s\S]*?</svg>")

# Control chars that XML parsers reject
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# Closing tag finder
_SVG_OPEN_TAG_RE = re.compile(r"(?is)<svg\b([^>]*)>")
_SVG_CLOSE_TAG_RE = re.compile(r"(?is)</svg\s*>")


def svg_rendering_available() -> bool:
    return _SVG_RENDERING_AVAILABLE


def _render_svg_to_png_bytes(svg_xml: str) -> bytes:
    if not _SVG_RENDERING_AVAILABLE:
        raise RuntimeError("CairoSVG not available")
    # CairoSVG expects valid UTF-8 bytes
    return cairosvg.svg2png(bytestring=svg_xml.encode("utf-8"))  # type: ignore[attr-defined]


def _ensure_svg_root_has_namespaces(svg_xml: str) -> str:
    """
    Make sure the root <svg> tag has at least the SVG namespace, plus xlink if needed.
    """
    m = _SVG_OPEN_TAG_RE.search(svg_xml)
    if not m:
        return svg_xml  # no root tag found; caller handles wrapping/closing recovery

    attrs = m.group(1) or ""

    needs_xmlns = "xmlns=" not in attrs
    uses_xlink = "xlink:" in svg_xml
    has_xlink_ns = "xmlns:xlink=" in attrs

    extras = []
    if needs_xmlns:
        extras.append('xmlns="http://www.w3.org/2000/svg"')
    if uses_xlink and not has_xlink_ns:
        extras.append('xmlns:xlink="http://www.w3.org/1999/xlink"')

    if not extras:
        return svg_xml

    # Insert the missing namespace attrs before the closing ">"
    new_open = f"<svg{attrs} {' '.join(extras)}>"
    return svg_xml[:m.start()] + new_open + svg_xml[m.end():]


def _strip_doctype_and_scripts(svg_xml: str) -> str:
    # Remove DOCTYPE and any internal subset to avoid entity expansion issues
    svg_xml = re.sub(r"(?is)<!DOCTYPE[\s\S]*?>", "", svg_xml)
    # Remove XML entity declarations if present
    svg_xml = re.sub(r"(?is)<!ENTITY[\s\S]*?>", "", svg_xml)
    # Drop <script> blocks for safety and to avoid parser quirks
    svg_xml = re.sub(r"(?is)<script[\s\S]*?</script>", "", svg_xml)
    return svg_xml


def _fix_common_entities(svg_xml: str) -> str:
    """
    Fixes the most common reasons for:
    'Invalid character in entity name'
    1) Named HTML entities that are not valid in XML (like &nbsp;).
    2) Ampersands without a terminating semicolon.
    3) Bare ampersands in text.
    Strategy:
      - First, add missing semicolons for a small set of common entities.
      - Convert HTML5 named entities to their unicode characters with html.unescape.
      - Re-escape any bare ampersands that do not start a numeric or named XML entity.
    """
    # Add semicolons when commonly omitted
    common = ("amp", "lt", "gt", "quot", "apos", "nbsp", "copy", "reg", "times", "euro", "mdash", "ndash")
    svg_xml = re.sub(r"&(" + "|".join(common) + r")(?!;)", r"&\1;", svg_xml, flags=re.IGNORECASE)

    # Convert HTML entities to characters
    svg_xml = html.unescape(svg_xml)

    # Re-escape stray ampersands that are not entities like &#123; or &#xAF; or &name;
    svg_xml = re.sub(r"&(?!(?:#\d+|#x[0-9a-fA-F]+|\w+);)", "&amp;", svg_xml)

    return svg_xml


def _sanitize_svg(svg_xml: str) -> str:
    """
    Sanitize and toughen SVG markup so that CairoSVG has a better chance to parse it.
    """
    # Strip BOM
    svg_xml = svg_xml.lstrip("\ufeff")

    # Remove control chars
    svg_xml = _CONTROL_CHARS_RE.sub("", svg_xml)

    # Normalize newlines
    svg_xml = svg_xml.replace("\r\n", "\n").replace("\r", "\n")

    # If there is an opening <svg> without a closing tag, append one
    if _SVG_OPEN_TAG_RE.search(svg_xml) and not _SVG_CLOSE_TAG_RE.search(svg_xml):
        svg_xml = svg_xml + "\n</svg>"

    # Remove doctype and scripts
    svg_xml = _strip_doctype_and_scripts(svg_xml)

    # Fix entities and ampersands
    svg_xml = _fix_common_entities(svg_xml)

    # Ensure required namespaces on the root tag
    svg_xml = _ensure_svg_root_has_namespaces(svg_xml)

    # Ensure XML declaration for parsers that prefer it
    stripped = svg_xml.lstrip()
    if not stripped.startswith("<?xml"):
        svg_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_xml

    return svg_xml


def _render_with_recovery(svg_xml: str) -> Tuple[bool, bytes | None, str | None]:
    """
    Try to render as-is, then try a sanitized version, then give up.
    Returns (ok, png_bytes_or_none, error_message_or_none).
    """
    # Try raw first
    try:
        png = _render_svg_to_png_bytes(svg_xml)
        return True, png, None
    except Exception as e1:
        # Try sanitized
        try:
            cleaned = _sanitize_svg(svg_xml)
            png = _render_svg_to_png_bytes(cleaned)
            return True, png, None
        except Exception as e2:
            # Final attempt: sanitize again after forcing a minimal wrapper if root is missing
            try:
                if "<svg" not in svg_xml.lower():
                    # Wrap standalone fragments in a minimal svg shell
                    wrapped = '<svg xmlns="http://www.w3.org/2000/svg">%s</svg>' % svg_xml
                else:
                    wrapped = svg_xml
                wrapped = _sanitize_svg(wrapped)
                png = _render_svg_to_png_bytes(wrapped)
                return True, png, None
            except Exception as e3:
                # Report the last error, keep earlier for logs
                logging.getLogger(__name__).debug("SVG render failed. First error: %r; Second: %r; Third: %r", e1, e2, e3)
                return False, None, str(e3)


def _attach_bytes(name: str, data: bytes) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


def extract_render_and_strip_svgs(text: str) -> Tuple[str, List[Tuple[str, io.BytesIO]]]:
    """
    Find any SVG fragments and render each to a PNG when possible.
    Strategy:
      1) Replace ```svg ...``` fenced blocks first
      2) Then find any inline <svg>...</svg> fragments anywhere else in the text
      3) For each fragment, try rendering raw, then sanitized. If all fail, attach the raw .svg
    Returns:
      cleaned_text: original text with svg fragments replaced by short notes
      svg_files: list of (filename, BytesIO) for sending as attachments (PNG if possible, otherwise raw .svg)
    """
    svg_files: List[Tuple[str, io.BytesIO]] = []
    idx = 0

    def _process_svg_fragment(svg_xml: str) -> str:
        nonlocal idx
        idx += 1

        if _SVG_RENDERING_AVAILABLE:
            ok, png_bytes, err = _render_with_recovery(svg_xml)
            if ok and png_bytes is not None:
                filename = f"svg_render_{idx}.png"
                svg_files.append((filename, _attach_bytes(filename, png_bytes)))
                return f"[SVG rendered and attached: {filename}]"
            else:
                # Attach sanitized original SVG so the user still gets something
                fallback = f"svg_render_{idx}.svg"
                # Even for fallback, store a sanitized svg so it opens cleanly
                safe_svg = _sanitize_svg(svg_xml)
                svg_files.append((fallback, _attach_bytes(fallback, safe_svg.encode("utf-8"))))
                return f"[SVG render failed, attached original SVG: {fallback}. Error: {err}]"
        else:
            filename = f"svg_render_{idx}.svg"
            safe_svg = _sanitize_svg(svg_xml)
            svg_files.append((filename, _attach_bytes(filename, safe_svg.encode("utf-8"))))
            return f"[SVG attached (SVG format) because rendering is unavailable: {filename}]"

    # Pass 1: fenced code blocks ```svg ... ```
    def _sub_fenced(match: re.Match) -> str:
        svg_xml = match.group(1).strip()
        return _process_svg_fragment(svg_xml)

    cleaned_text = SVG_BLOCK_RE.sub(_sub_fenced, text)

    # Pass 2: any inline <svg>...</svg> that remain in the text
    def _sub_inline(match: re.Match) -> str:
        svg_xml = match.group(0)
        return _process_svg_fragment(svg_xml)

    cleaned_text = INLINE_SVG_RE.sub(_sub_inline, cleaned_text)

    return cleaned_text, svg_files
