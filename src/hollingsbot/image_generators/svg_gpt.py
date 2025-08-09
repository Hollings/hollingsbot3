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


def _extract_svg(text: str) -> str:
    """Extract a single standalone <svg>...</svg> from model text (optionally inside a code fence)."""
    code_block = _CODE_BLOCK_RE.search(text)
    if code_block:
        logger.debug("SVG found inside code fence")
        text = code_block.group(1)

    # Strip stray code fences or language hints
    text = text.strip().strip("`").strip()

    if text.lower().startswith("xml"):
        text = text[3:].lstrip()

    # Ensure we have exactly one <svg> ... </svg>
    if "<svg" not in text.lower():
        logger.error("Model output did not contain an <svg> root element")
        raise ValueError("Model did not return an <svg> document.")
    start = text.lower().find("<svg")
    end = text.lower().rfind("</svg>")
    if start == -1 or end == -1:
        logger.error("Model output contained incomplete SVG content (missing start or end tag)")
        raise ValueError("Incomplete SVG content.")
    svg = text[start : end + len("</svg>")].strip()
    logger.debug("Extracted SVG length=%d", len(svg))
    return svg


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
    if out == svg:
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
            "Prefer simple shapes and paths. Provide a single <svg> root element."
            "The token limit for this response is 5000"
        )

        user = (
            "Create an illustration as SVG for the following description. "
            "Your response MUST be only the <svg>…</svg> markup, nothing else.\n\n"
            f"Description: {prompt}\n"
            "Canvas: 1024x1024 square."
        )

        t0 = time.time()
        try:
            # OpenAI Responses API (correct parameter: max_output_tokens)
            coro = _client().responses.create(
                model=self.model,
                instructions=sys,
                input=user,
                max_output_tokens=self.max_output_tokens,
                timeout=self.request_timeout,  # per-request client timeout (OpenAI SDK)
            )
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
        text: str
        logger.info("rsp:", rsp)
        try:
            text = getattr(rsp, "output_text", None)  # preferred
            logger.info(f"SVG generation result: {text}")
            if not text:
                # Fallback: aggregate from .output[].content[].text
                parts: list[str] = []
                output = getattr(rsp, "output", None)
                if output:
                    for item in output:
                        for c in getattr(item, "content", []) or []:
                            t = getattr(c, "text", None)
                            if t:
                                parts.append(t)
                text = "\n".join(parts) if parts else str(rsp)
        except Exception:
            text = str(rsp)

        # Extract and normalize SVG
        try:
            svg = _extract_svg(text)
            # Refuse embedded data URIs which explode output size/time
            if "data:image" in svg.lower():
                logger.warning("Model attempted to embed base64 image data; returning placeholder")
                return _placeholder_png("SVG contained embedded raster data; not allowed.")
            svg = _enforce_canvas(svg, 1024)
        except Exception as e:
            logger.exception("Failed to extract/normalize SVG from model output: %s", e)
            return _placeholder_png("Invalid SVG returned by the model.")

        t1 = time.time()
        try:
            png = _render_svg(svg)
        except Exception as e:
            logger.exception("Failed to render SVG to PNG: %s", e)
            return _placeholder_png("SVG rasterization failed.")
        render_ms = int((time.time() - t1) * 1000)

        logger.info(
            "SVG generation done bytes=%d timings_ms api=%d render=%d",
            len(png),
            api_ms,
            render_ms,
        )
        return png
