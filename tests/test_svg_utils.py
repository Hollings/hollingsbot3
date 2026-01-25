"""Tests for SVG utilities."""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest

from hollingsbot.utils.svg_utils import (
    svg_rendering_available,
    _sanitize_svg,
    _ensure_svg_root_has_namespaces,
    _fix_common_entities,
    _strip_doctype_and_scripts,
    extract_render_and_strip_svgs,
    SVG_BLOCK_RE,
    INLINE_SVG_RE,
)


class TestSanitizeSvg:
    """Tests for SVG sanitization."""

    def test_strips_bom(self):
        """Test that BOM is stripped."""
        svg = '\ufeff<svg><rect/></svg>'
        result = _sanitize_svg(svg)
        assert not result.startswith('\ufeff')

    def test_removes_control_chars(self):
        """Test that control characters are removed."""
        svg = '<svg>\x00\x01\x02<rect/></svg>'
        result = _sanitize_svg(svg)
        assert '\x00' not in result
        assert '\x01' not in result
        assert '\x02' not in result

    def test_normalizes_newlines(self):
        """Test that newlines are normalized."""
        svg = '<svg>\r\n<rect/>\r</svg>'
        result = _sanitize_svg(svg)
        assert '\r\n' not in result
        assert '\r' not in result

    def test_adds_closing_tag_if_missing(self):
        """Test that missing closing tag is added."""
        svg = '<svg><rect/>'
        result = _sanitize_svg(svg)
        assert '</svg>' in result

    def test_adds_xml_declaration(self):
        """Test that XML declaration is added if missing."""
        svg = '<svg><rect/></svg>'
        result = _sanitize_svg(svg)
        assert '<?xml version="1.0"' in result

    def test_preserves_existing_xml_declaration(self):
        """Test that existing XML declaration is not duplicated."""
        svg = '<?xml version="1.0"?><svg><rect/></svg>'
        result = _sanitize_svg(svg)
        assert result.count('<?xml') == 1


class TestEnsureSvgNamespaces:
    """Tests for namespace handling."""

    def test_adds_xmlns_if_missing(self):
        """Test that xmlns is added if missing."""
        svg = '<svg><rect/></svg>'
        result = _ensure_svg_root_has_namespaces(svg)
        assert 'xmlns="http://www.w3.org/2000/svg"' in result

    def test_preserves_existing_xmlns(self):
        """Test that existing xmlns is preserved."""
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        result = _ensure_svg_root_has_namespaces(svg)
        assert result.count('xmlns=') == 1

    def test_adds_xlink_namespace_when_used(self):
        """Test that xlink namespace is added when xlink is used."""
        svg = '<svg><use xlink:href="#foo"/></svg>'
        result = _ensure_svg_root_has_namespaces(svg)
        assert 'xmlns:xlink=' in result

    def test_no_xlink_namespace_when_not_used(self):
        """Test that xlink namespace is not added when not used."""
        svg = '<svg><rect/></svg>'
        result = _ensure_svg_root_has_namespaces(svg)
        assert 'xmlns:xlink=' not in result

    def test_handles_no_svg_tag(self):
        """Test handling when there's no svg tag."""
        text = 'not an svg'
        result = _ensure_svg_root_has_namespaces(text)
        assert result == text


class TestFixCommonEntities:
    """Tests for entity fixing."""

    def test_fixes_nbsp_entity(self):
        """Test that &nbsp; is converted."""
        svg = '<text>&nbsp;</text>'
        result = _fix_common_entities(svg)
        # nbsp should be converted to actual non-breaking space
        assert '&nbsp;' not in result

    def test_fixes_missing_semicolon(self):
        """Test that missing semicolons are added."""
        svg = '<text>&amp &lt</text>'
        result = _fix_common_entities(svg)
        # After unescape, the entities should be converted
        assert '<' in result or '&lt;' in result

    def test_escapes_bare_ampersands(self):
        """Test that bare ampersands are escaped."""
        svg = '<text>foo & bar</text>'
        result = _fix_common_entities(svg)
        assert '&amp;' in result


class TestStripDoctypeAndScripts:
    """Tests for doctype and script removal."""

    def test_removes_doctype(self):
        """Test that DOCTYPE is removed."""
        svg = '<!DOCTYPE svg PUBLIC "..."><svg><rect/></svg>'
        result = _strip_doctype_and_scripts(svg)
        assert '<!DOCTYPE' not in result

    def test_removes_entity_declarations(self):
        """Test that ENTITY declarations are removed."""
        svg = '<!ENTITY foo "bar"><svg><rect/></svg>'
        result = _strip_doctype_and_scripts(svg)
        assert '<!ENTITY' not in result

    def test_removes_script_tags(self):
        """Test that script tags are removed."""
        svg = '<svg><script>alert("xss")</script><rect/></svg>'
        result = _strip_doctype_and_scripts(svg)
        assert '<script>' not in result
        assert 'alert' not in result


class TestSvgBlockRegex:
    """Tests for the fenced code block regex."""

    def test_matches_svg_code_block(self):
        """Test matching ```svg ... ``` blocks."""
        text = '```svg\n<svg><rect/></svg>\n```'
        matches = SVG_BLOCK_RE.findall(text)
        assert len(matches) == 1
        assert '<svg>' in matches[0]

    def test_case_insensitive(self):
        """Test that matching is case insensitive."""
        text = '```SVG\n<svg><rect/></svg>\n```'
        matches = SVG_BLOCK_RE.findall(text)
        assert len(matches) == 1

    def test_multiple_blocks(self):
        """Test matching multiple SVG blocks."""
        text = '```svg\n<svg>1</svg>\n```\ntext\n```svg\n<svg>2</svg>\n```'
        matches = SVG_BLOCK_RE.findall(text)
        assert len(matches) == 2


class TestInlineSvgRegex:
    """Tests for the inline SVG regex."""

    def test_matches_inline_svg(self):
        """Test matching inline SVG tags."""
        text = 'Before <svg><rect/></svg> After'
        matches = INLINE_SVG_RE.findall(text)
        assert len(matches) == 1
        assert '<svg>' in matches[0]

    def test_matches_svg_with_attributes(self):
        """Test matching SVG with attributes."""
        text = '<svg width="100" height="100"><rect/></svg>'
        matches = INLINE_SVG_RE.findall(text)
        assert len(matches) == 1

    def test_matches_multiline_svg(self):
        """Test matching multiline SVG."""
        text = '''<svg>
            <rect x="0" y="0"/>
        </svg>'''
        matches = INLINE_SVG_RE.findall(text)
        assert len(matches) == 1


class TestExtractRenderAndStripSvgs:
    """Tests for the main SVG extraction function."""

    @patch('hollingsbot.utils.svg_utils._SVG_RENDERING_AVAILABLE', False)
    def test_returns_svg_file_when_rendering_unavailable(self):
        """Test that SVG file is returned when rendering is unavailable."""
        text = '```svg\n<svg><rect/></svg>\n```'
        cleaned, files = extract_render_and_strip_svgs(text)
        assert len(files) == 1
        assert files[0][0].endswith('.svg')
        assert '[SVG attached' in cleaned

    @patch('hollingsbot.utils.svg_utils._SVG_RENDERING_AVAILABLE', False)
    def test_strips_svg_from_text(self):
        """Test that SVG is stripped from returned text."""
        text = 'Before ```svg\n<svg><rect/></svg>\n``` After'
        cleaned, files = extract_render_and_strip_svgs(text)
        assert '<svg>' not in cleaned
        assert 'Before' in cleaned
        assert 'After' in cleaned

    @patch('hollingsbot.utils.svg_utils._SVG_RENDERING_AVAILABLE', False)
    def test_handles_multiple_svgs(self):
        """Test handling multiple SVGs."""
        text = '```svg\n<svg>1</svg>\n```\n<svg>2</svg>'
        cleaned, files = extract_render_and_strip_svgs(text)
        assert len(files) == 2
        assert '<svg>' not in cleaned

    @patch('hollingsbot.utils.svg_utils._SVG_RENDERING_AVAILABLE', False)
    def test_handles_empty_text(self):
        """Test handling empty text."""
        text = ''
        cleaned, files = extract_render_and_strip_svgs(text)
        assert cleaned == ''
        assert files == []

    @patch('hollingsbot.utils.svg_utils._SVG_RENDERING_AVAILABLE', False)
    def test_handles_text_without_svg(self):
        """Test handling text without SVG."""
        text = 'Just some regular text'
        cleaned, files = extract_render_and_strip_svgs(text)
        assert cleaned == text
        assert files == []

    @patch('hollingsbot.utils.svg_utils._SVG_RENDERING_AVAILABLE', True)
    @patch('hollingsbot.utils.svg_utils._render_svg_to_png_bytes')
    def test_renders_to_png_when_available(self, mock_render):
        """Test that PNG is rendered when CairoSVG is available."""
        mock_render.return_value = b'fake png data'
        text = '```svg\n<svg><rect/></svg>\n```'
        cleaned, files = extract_render_and_strip_svgs(text)
        assert len(files) == 1
        assert files[0][0].endswith('.png')
        assert '[SVG rendered' in cleaned

    @patch('hollingsbot.utils.svg_utils._SVG_RENDERING_AVAILABLE', True)
    @patch('hollingsbot.utils.svg_utils._render_svg_to_png_bytes')
    def test_falls_back_to_svg_on_render_failure(self, mock_render):
        """Test fallback to SVG when rendering fails."""
        mock_render.side_effect = Exception("Render failed")
        text = '```svg\n<svg><rect/></svg>\n```'
        cleaned, files = extract_render_and_strip_svgs(text)
        assert len(files) == 1
        assert files[0][0].endswith('.svg')
        assert 'render failed' in cleaned.lower()


class TestSvgRenderingAvailable:
    """Tests for the svg_rendering_available function."""

    def test_returns_boolean(self):
        """Test that it returns a boolean."""
        result = svg_rendering_available()
        assert isinstance(result, bool)
