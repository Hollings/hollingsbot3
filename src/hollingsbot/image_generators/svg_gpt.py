from __future__ import annotations

import os
import re
import io
import time
import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# --- Cairo / rasterization availability ---------------------------------------------------------
try:
    import cairosvg  # type: ignore

    _HAS_CAIRO = True
except Exception as e:  # pragma: no cover - optional dependency
    _HAS_CAIRO = False
    logger.warning(
        "CairoSVG unavailable; PNG rasterization will fall back to a placeholder. %s",
        e,
    )

from .base import ImageGeneratorAPI

# --- Client singleton ---------------------------------------------------------------------------
_CLIENT: Optional[AsyncOpenAI] = None


def _client() -> AsyncOpenAI:
    """Create or return a shared AsyncOpenAI client with reasonable defaults."""
    global _CLIENT
    if _CLIENT is None:
        http_timeout = float(os.getenv("OPENAI_HTTP_TIMEOUT", "30"))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        logger.debug(
            "Initializing AsyncOpenAI client (timeout=%ss, max_retries=%s)",
            http_timeout,
            max_retries,
        )
        _CLIENT = AsyncOpenAI(timeout=http_timeout, max_retries=max_retries)
    return _CLIENT


# --- Helpers ------------------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:svg)?\s*([\s\S]*?)```", re.IGNORECASE)
_SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.IGNORECASE | re.MULTILINE)
_SVG_BLOCK_RE = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)

# Remove clearly-problematic constructs but keep the rest of the SVG intact.
def _strip_disallowed(svg: str) -> str:
    svg = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"<foreignObject\b[^>]*>[\s\S]*?</foreignObject>", "", svg, flags=re.IGNORECASE)
    # Drop base64-embedded rasters (keep the rest of the SVG)
    svg = re.sub(
        r"<image\b[^>]*(?:href|xlink:href)\s*=\s*['\"]data:[^'\"]*['\"][^>]*\/?>",
        "",
        svg,
        flags=re.IGNORECASE,
    )
    # Remove event handlers
    svg = re.sub(r"\son[a-z]+\s*=\s*['\"][^'\"]*['\"]", "", svg, flags=re.IGNORECASE)
    return svg


def _simplify_svg(svg: str) -> str:
    """Aggressively simplify features that often break rasterizers (filters, masks, etc.)."""
    # Remove filter references and definitions
    svg = re.sub(r"\sfilter\s*=\s*['\"]url\([^'\"]*\)['\"]", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"<filter\b[^>]*>[\s\S]*?</filter>", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"<fe[A-Za-z]+\b[^>]*\/?>", "", svg, flags=re.IGNORECASE)
    # Remove masks/clipPaths and their uses
    svg = re.sub(r"<mask\b[^>]*>[\s\S]*?</mask>", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"<clipPath\b[^>]*>[\s\S]*?</clipPath>", "", svg, flags=re.IGNORECASE)
    svg = re.sub(r"\sclip-path\s*=\s*['\"][^'\"]*['\"]", "", svg, flags=re.IGNORECASE)
    # Remove animations
    svg = re.sub(r"<animate(?:Transform)?\b[^>]*>[\s\S]*?</animate(?:Transform)?>", "", svg, flags=re.IGNORECASE)
    return svg


def _extract_svg(text: str, *, prompt: str | None = None) -> str:
    """
    Extract (or salvage) a single standalone <svg>...</svg> from model text.
    Be generous: accept fenced blocks, auto-close if needed, or wrap stray markup.
    As a last resort, synthesize a minimal SVG that shows the prompt text.
    """
    original = text or ""
    # Handle code fences
    m = _CODE_BLOCK_RE.search(original)
    if m:
        logger.debug("SVG found inside code fence")
        text = m.group(1)
    else:
        text = original

    # Clean stray backticks/newlines/unicode separators
    text = (text or "").strip().strip("`").replace("\u2028", "\n").replace("\u2029", "\n")

    # Happy path: exact <svg>...</svg> block
    m = _SVG_BLOCK_RE.search(text)
    if m:
        svg = m.group(0).strip()
        logger.debug("Extracted full <svg> block (len=%d)", len(svg))
        return svg

    # If we have an <svg ...> start but no close tag, auto-close.
    lower = text.lower()
    start = lower.find("<svg")
    if start != -1:
        logger.warning("Model output contained an unterminated <svg>; auto-closing.")
        body = text[start:].rstrip()
        if not body.lower().endswith("</svg>"):
            body = body + "</svg>"
        return body

    # If there is *some* markup, wrap it so we still show something.
    if "<" in text and ">" in text:
        logger.warning("No <svg> root found; wrapping returned markup inside an SVG root.")
        inner = re.sub(r"^(\s*<!DOCTYPE[^>]*>)", "", text, flags=re.IGNORECASE).strip()
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" '
            f'viewBox="0 0 1024 1024"><g>{inner}</g></svg>'
        )

    # Last resort: synthesize a minimally-informative SVG
    logger.error("Model did not return usable markup; synthesizing fallback SVG.")
    prompt_label = (prompt or "Generated image").strip()
    prompt_label = (prompt_label[:120] + "…") if len(prompt_label) > 120 else prompt_label
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">'
        '<defs><linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">'
        '<stop offset="0" stop-color="#f7f7fb"/><stop offset="1" stop-color="#e8f0ff"/></linearGradient></defs>'
        '<rect width="1024" height="1024" fill="url(#bg)"/>'
        '<circle cx="512" cy="512" r="220" fill="#0b0d15" opacity=".08"/>'
        f'<text x="48" y="96" font-family="DejaVu Sans, Arial, sans-serif" font-size="28" fill="#111">{prompt_label}</text>'
        "</svg>"
    )


def _enforce_canvas(svg: str, size: int = 1024) -> str:
    """Ensure width/height/viewBox are set to a square canvas."""

    def _repl(m: re.Match) -> str:
        attrs = m.group(1)
        # Remove existing conflicting attrs
        attrs = re.sub(r'\b(width|height)\s*=\s*"[^"]*"', "", attrs, flags=re.IGNORECASE)
        attrs = re.sub(r'\bviewBox\s*=\s*"[^"]*"', "", attrs, flags=re.IGNORECASE)
        attrs = " ".join(attrs.split())
        core = f' width="{size}" height="{size}" viewBox="0 0 {size} {size}"'
        if attrs:
            core = f"{core} {attrs}"
        return f"<svg{core}>"

    out = _SVG_OPEN_RE.sub(_repl, svg, count=1)
    if out != svg:
        logger.debug("Applied canvas normalization to SVG (size=%d)", size)
    return out


def _placeholder_png(message: str) -> bytes:
    """Create a simple PNG with an error message (used when rasterization fails)."""
    w, h = 1024, 1024
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    message = f"{message}\n\n(This is a placeholder image.)"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=28)
    except Exception:
        font = ImageFont.load_default()
    d.multiline_text((40, 40), message, fill="black", font=font, spacing=8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _render_svg(svg: str) -> bytes:
    """Render SVG to PNG bytes, or create a placeholder PNG if Cairo is missing."""
    if _HAS_CAIRO:
        logger.debug("Rendering SVG with CairoSVG (len=%d)", len(svg))
        return cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    logger.warning("CairoSVG not installed; returning placeholder PNG")
    return _placeholder_png("SVG rasterization not available on this system.")


# --- Main implementation ------------------------------------------------------------------------
class SvgGPTImageGenerator(ImageGeneratorAPI):
    """
    Generates an SVG with the model, normalizes it to a 1024x1024 canvas, and returns PNG bytes.
    Uses the OpenAI Responses API (not Chat Completions). Avoids unsupported parameters.
    """

    def __init__(self, model: str):
        self.model = model
        # Tunables via env for easy ops
        self.temperature = float(os.getenv("SVG_GPT_TEMPERATURE", "0.2"))
        self.max_output_tokens = int(os.getenv("SVG_GPT_MAX_TOKENS", "5000"))
        self.request_timeout = int(os.getenv("SVG_GPT_TIMEOUT", "30"))

    async def generate(self, prompt: str, *, seed: int | None = None) -> bytes:  # type: ignore[override]
        """
        Ask the model to produce SVG markup for the given prompt and rasterize it to PNG.
        If the model request times out, returns a placeholder PNG instead of throwing.
        """
        prompt_preview = " ".join(prompt.split())
        if len(prompt_preview) > 200:
            prompt_preview = prompt_preview[:200] + "…"
        logger.info(
            "SVG generation start model=%s seed=%s prompt_preview=%s",
            self.model,
            seed,
            prompt_preview,
        )

        sys = (
            "You produce ONLY valid, standalone SVG markup. "
            "Do NOT wrap in code fences or any prose. "
            "Use a 1024x1024 canvas. "
            "Output must be concise (< 1000 characters). "
            "Absolutely DO NOT embed base64 data URIs or external images; no <image href='data:'>. "
            "Avoid <foreignObject>. No scripts. "
            "Prefer simple shapes and paths. Provide a single <svg> root element. "
            "End your output with </svg> and do not output anything after it."
        )

        user = (
            "Create an illustration as SVG for the following description. "
            "Your response MUST be only the <svg>…</svg> markup, nothing else.\n\n"
            f"Description: {prompt}\n"
            "Canvas: 1024x1024 square."
        )

        # --- request helper (with retry) --------------------------------------------------------
        async def _call_openai(max_tokens: int) -> object:
            return await _client().responses.create(
                model=self.model,
                instructions=sys,
                input=user,
                max_output_tokens=max_tokens,
                reasoning={"effort": "medium"},
                timeout=self.request_timeout,  # per-request client timeout (OpenAI SDK)
            )

        t0 = time.time()
        try:
            coro = _call_openai(self.max_output_tokens)
            rsp = await asyncio.wait_for(coro, timeout=self.request_timeout + 5)  # outer guard
        except asyncio.TimeoutError:
            api_ms = int((time.time() - t0) * 1000)
            logger.warning(
                "OpenAI request timed out after %ss (latency_ms=%d)",
                self.request_timeout,
                api_ms,
            )
            return _placeholder_png(f"SVG generation timed out after {self.request_timeout}s.")
        except Exception as e:
            api_ms = int((time.time() - t0) * 1000)
            logger.exception("OpenAI request failed (latency_ms=%d): %s", api_ms, e)
            return _placeholder_png("SVG generation failed.")

        api_ms = int((time.time() - t0) * 1000)

        # Extract plain text from Responses API
        def _gather_text(r: object) -> str:
            # preferred (SDK helper)
            t = getattr(r, "output_text", None)
            if t:
                return t
            # aggregate from .output[].content[].text
            parts: list[str] = []
            output = getattr(r, "output", None)
            if output:
                for item in output:
                    for c in getattr(item, "content", []) or []:
                        tt = getattr(c, "text", None)
                        if tt:
                            parts.append(tt)
            return "\n".join(parts)

        text: str = _gather_text(rsp) or ""
        status = getattr(rsp, "status", "")
        incomplete_reason = getattr(getattr(rsp, "incomplete_details", None), "reason", None)

        logger.debug(
            "Responses status=%s incomplete_reason=%s text_len=%d preview=%r",
            status,
            incomplete_reason,
            len(text),
            text[:120],
        )

        # If incomplete due to token cap (or empty text), retry once with a bigger cap & stop-seq.
        if (status != "completed" and incomplete_reason == "max_output_tokens") or not text.strip():
            try:
                logger.info("Retrying with higher max_output_tokens and low reasoning.")
                bigger = max(self.max_output_tokens + 2000, 5000)
                coro = _call_openai(bigger)
                rsp = await asyncio.wait_for(coro, timeout=self.request_timeout + 5)
                text = _gather_text(rsp) or text
            except Exception as e:
                logger.warning("Retry failed: %s", e)

        # Extract and normalize SVG (with salvage & sanitization)
        try:
            svg = _extract_svg(text, prompt=prompt)
            svg = _strip_disallowed(svg)
            svg = _enforce_canvas(svg, 1024)
        except Exception as e:
            logger.exception("Failed to extract/normalize SVG from model output: %s", e)
            # As a last resort, synthesize a minimal SVG to avoid hard failure.
            svg = _extract_svg("", prompt=prompt)
            svg = _enforce_canvas(svg, 1024)

        # Render; on failure, simplify then retry once.
        t1 = time.time()
        try:
            png = _render_svg(svg)
        except Exception as e:
            logger.warning("Render failed (%s). Retrying with simplified SVG.", e)
            try:
                simpler = _simplify_svg(svg)
                png = _render_svg(simpler)
            except Exception as e2:
                logger.exception("Failed to render even after simplification: %s", e2)
                return _placeholder_png("SVG rasterization failed.")

        render_ms = int((time.time() - t1) * 1000)
        logger.info(
            "SVG generation done bytes=%d timings_ms api=%d render=%d",
            len(png),
            api_ms,
            render_ms,
        )
        return png
