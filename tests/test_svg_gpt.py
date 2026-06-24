"""Tests for SVG-GPT image generator helpers (pure text/markup transforms)."""

from __future__ import annotations

import io

from PIL import Image

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
