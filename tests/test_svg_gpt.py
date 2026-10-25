"""Tests for SVG-GPT image generator helpers (pure text/markup transforms)."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from PIL import Image

from hollingsbot.image_generators import svg_gpt
from hollingsbot.image_generators.svg_gpt import (
    _enforce_canvas,
    _extract_svg,
    _placeholder_png,
    _simplify_svg,
    _strip_disallowed,
)


class TestExtractSvg:
    def test_plain_svg_block(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        assert _extract_svg(svg) == svg

    def test_inside_code_fence(self):
        out = _extract_svg("```svg\n<svg><circle/></svg>\n```")
        assert out.startswith("<svg>")
        assert out.endswith("</svg>")

    def test_autocloses_unterminated(self):
        out = _extract_svg("<svg><rect/>")
        assert out.endswith("</svg>")

    def test_wraps_stray_markup(self):
        out = _extract_svg("<rect width='10'/>")
        assert out.startswith("<svg")
        assert "<rect width='10'/>" in out
        assert out.endswith("</svg>")

    def test_fallback_synthesizes_with_prompt(self):
        out = _extract_svg("", prompt="a sunset")
        assert out.startswith("<svg")
        assert "a sunset" in out

    def test_fallback_truncates_long_prompt(self):
        out = _extract_svg("not markup at all", prompt="x" * 200)
        assert "…" in out


class TestStripDisallowed:
    def test_removes_script(self):
        out = _strip_disallowed("<svg><script>evil()</script><rect/></svg>")
        assert "script" not in out.lower()
        assert "<rect/>" in out

    def test_removes_foreign_object(self):
        out = _strip_disallowed("<svg><foreignObject>x</foreignObject></svg>")
        assert "foreignobject" not in out.lower()

    def test_removes_event_handlers(self):
        out = _strip_disallowed('<svg><rect onclick="x()"/></svg>')
        assert "onclick" not in out.lower()

    def test_removes_base64_image(self):
        out = _strip_disallowed('<svg><image href="data:image/png;base64,AAA"/></svg>')
        assert "data:" not in out


class TestSimplifySvg:
    def test_removes_filter_definitions(self):
        out = _simplify_svg("<svg><filter id='f'><feBlur/></filter><rect/></svg>")
        assert "<filter" not in out
        assert "<rect/>" in out

    def test_removes_mask_and_clippath(self):
        out = _simplify_svg("<svg><mask id='m'></mask><clipPath id='c'></clipPath></svg>")
        assert "<mask" not in out
        assert "<clipPath" not in out

    def test_removes_filter_attribute(self):
        out = _simplify_svg('<svg><rect filter="url(#f)"/></svg>')
        assert "filter=" not in out


class TestEnforceCanvas:
    def test_sets_canvas_dimensions(self):
        out = _enforce_canvas("<svg><rect/></svg>", size=512)
        assert 'width="512"' in out
        assert 'height="512"' in out
        assert 'viewBox="0 0 512 512"' in out

    def test_replaces_existing_dimensions(self):
        out = _enforce_canvas('<svg width="64" height="64" viewBox="0 0 64 64"><rect/></svg>', size=1024)
        assert 'width="1024"' in out
        assert 'width="64"' not in out

    def test_preserves_other_attributes(self):
        out = _enforce_canvas('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>', size=256)
        assert "xmlns" in out


class TestPlaceholderPng:
    def test_returns_valid_png(self):
        data = _placeholder_png("oops")
        img = Image.open(io.BytesIO(data))
        assert img.format == "PNG"
        assert img.size == (1024, 1024)


@pytest.mark.parametrize("truncated", [False, True])
async def test_claude_svg_uses_messages_api(monkeypatch, fake_anthropic, truncated):
    original_handle = fake_anthropic._handle

    def respond(request):
        payload = original_handle(request).json()
        payload["content"] = [
            {"type": "thinking", "thinking": "private reasoning", "signature": "test"},
            {"type": "text", "text": "<svg><circle/></svg>"},
        ]
        payload["stop_reason"] = "max_tokens" if truncated and len(fake_anthropic.requests) == 1 else "end_turn"
        return httpx2.Response(200, json=payload)

    monkeypatch.setattr(fake_anthropic, "_handle", respond)
    monkeypatch.setattr(svg_gpt, "get_client", lambda _name, _factory: fake_anthropic.client())
    rendered = []
    monkeypatch.setattr(svg_gpt, "_render_svg", lambda svg: rendered.append(svg) or b"png")
    try:
        assert await svg_gpt.SvgGPTImageGenerator("claude-opus-5-5").generate("a circle") == b"png"
        assert len(fake_anthropic.requests) == (2 if truncated else 1)
        request = fake_anthropic.requests[0]
        assert request["model"] == "claude-opus-5-5"
        assert request["output_config"] == {"effort": "medium"}
        assert "a circle" in request["messages"][0]["content"]
        assert "temperature" not in request
        assert "private reasoning" not in rendered[0]
        assert "<circle/>" in rendered[0]
        assert 'width="1024"' in rendered[0]
        if truncated:
            assert fake_anthropic.requests[1]["max_tokens"] > request["max_tokens"]
    finally:
        await fake_anthropic.client().close()


async def test_gpt_svg_keeps_responses_api(monkeypatch):
    create = AsyncMock(return_value=SimpleNamespace(output_text="<svg><rect/></svg>", status="completed"))
    monkeypatch.setattr(svg_gpt, "_client", lambda: SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr(svg_gpt, "_render_svg", lambda _svg: b"png")
    assert await svg_gpt.SvgGPTImageGenerator("gpt-5").generate("a square") == b"png"
    assert create.call_args.kwargs["model"] == "gpt-5"


@pytest.mark.parametrize("reason", ["max_tokens", "end_turn"])
async def test_never_renders_incomplete_svg(monkeypatch, reason):
    # Cover both provider-reported truncation and an unfinished SVG even when
    # the provider claims the response ended normally.
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="<svg><path d='M0 0")], stop_reason=reason)
    create = AsyncMock(return_value=response)
    monkeypatch.setattr(svg_gpt, "get_client", lambda *_args: SimpleNamespace(messages=SimpleNamespace(create=create)))
    render = AsyncMock()
    monkeypatch.setattr(svg_gpt, "_render_svg", render)
    monkeypatch.setattr(svg_gpt, "_placeholder_png", lambda _message: b"incomplete")
    monkeypatch.delenv("SVG_GPT_MAX_TOKENS", raising=False)
    assert await svg_gpt.SvgGPTImageGenerator("claude-opus-5-5").generate("a detailed scene") == b"incomplete"
    assert [call.kwargs["max_tokens"] for call in create.call_args_list] == [32000, 128000]
    render.assert_not_called()
