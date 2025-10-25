"""Tests for URL metadata extraction."""

from __future__ import annotations

import pytest

from hollingsbot.url_metadata import (
    extract_urls,
    format_metadata_for_llm,
    URLMetadata,
    _parse_metadata,
)


@pytest.mark.asyncio
async def test_extract_urls():
    """Test URL extraction from text."""
    text = "Check out https://example.com and www.test.com for more info"
    urls = await extract_urls(text)
    assert len(urls) == 2
    assert "https://example.com" in urls
    assert "https://www.test.com" in urls


@pytest.mark.asyncio
async def test_extract_urls_no_urls():
    """Test URL extraction with no URLs."""
    text = "This text has no URLs"
    urls = await extract_urls(text)
    assert len(urls) == 0


@pytest.mark.asyncio
async def test_extract_urls_multiple():
    """Test URL extraction with multiple URLs."""
    text = "Visit https://example.com, http://test.org, and www.sample.net!"
    urls = await extract_urls(text)
    assert len(urls) == 3


@pytest.mark.asyncio
async def test_extract_urls_with_query_params():
    """Test URL extraction with query parameters."""
    text = "Check this out: https://x.com/tarngerine/status/1981835235332698465?s=46"
    urls = await extract_urls(text)
    assert len(urls) == 1
    assert urls[0] == "https://x.com/tarngerine/status/1981835235332698465?s=46"


@pytest.mark.asyncio
async def test_extract_urls_with_trailing_punctuation():
    """Test URL extraction excludes trailing punctuation."""
    text = "Check this out: https://example.com/page?id=123!"
    urls = await extract_urls(text)
    assert len(urls) == 1
    # Should not include the trailing !
    assert urls[0] == "https://example.com/page?id=123"


def test_parse_metadata_open_graph():
    """Test parsing Open Graph metadata."""
    html = """
    <html>
    <head>
        <meta property="og:title" content="Test Title">
        <meta property="og:description" content="Test Description">
        <meta property="og:image" content="https://example.com/image.jpg">
        <meta property="og:site_name" content="Example Site">
    </head>
    <body></body>
    </html>
    """
    metadata = _parse_metadata("https://example.com", html)
    assert metadata.url == "https://example.com"
    assert metadata.title == "Test Title"
    assert metadata.description == "Test Description"
    assert metadata.image_urls == ["https://example.com/image.jpg"]
    assert metadata.site_name == "Example Site"


def test_parse_metadata_twitter_cards():
    """Test parsing Twitter Card metadata."""
    html = """
    <html>
    <head>
        <meta name="twitter:title" content="Twitter Title">
        <meta name="twitter:description" content="Twitter Description">
        <meta name="twitter:image" content="https://example.com/twitter.jpg">
    </head>
    <body></body>
    </html>
    """
    metadata = _parse_metadata("https://example.com", html)
    assert metadata.title == "Twitter Title"
    assert metadata.description == "Twitter Description"
    assert metadata.image_urls == ["https://example.com/twitter.jpg"]


def test_parse_metadata_fallback_to_html():
    """Test fallback to standard HTML tags."""
    html = """
    <html>
    <head>
        <title>HTML Title</title>
        <meta name="description" content="HTML Description">
    </head>
    <body></body>
    </html>
    """
    metadata = _parse_metadata("https://example.com", html)
    assert metadata.title == "HTML Title"
    assert metadata.description == "HTML Description"


def test_format_metadata_for_llm():
    """Test formatting metadata for LLM context."""
    metadata = URLMetadata(
        url="https://example.com",
        title="Example Title",
        description="Example Description",
        image_urls=["https://example.com/image.jpg"],
        site_name="Example Site",
    )
    formatted = format_metadata_for_llm(metadata)
    assert "https://example.com" in formatted
    assert "Example Title" in formatted
    assert "Example Description" in formatted
    assert "https://example.com/image.jpg" in formatted
    assert "Example Site" in formatted


def test_format_metadata_truncates_long_description():
    """Test that long descriptions are truncated."""
    long_desc = "a" * 400
    metadata = URLMetadata(
        url="https://example.com",
        title="Title",
        description=long_desc,
    )
    formatted = format_metadata_for_llm(metadata)
    assert "..." in formatted
    assert len(formatted) < len(long_desc) + 100  # Should be much shorter


def test_format_metadata_minimal():
    """Test formatting with minimal metadata."""
    metadata = URLMetadata(
        url="https://example.com",
        title="Just a Title",
    )
    formatted = format_metadata_for_llm(metadata)
    assert "https://example.com" in formatted
    assert "Just a Title" in formatted


def test_parse_metadata_multiple_images():
    """Test parsing multiple Open Graph images (e.g., from Twitter/X)."""
    html = """
    <html>
    <head>
        <meta property="og:title" content="Multi-Image Post">
        <meta property="og:image" content="https://example.com/image1.jpg">
        <meta property="og:image" content="https://example.com/image2.jpg">
        <meta property="og:image" content="https://example.com/image3.jpg">
    </head>
    <body></body>
    </html>
    """
    metadata = _parse_metadata("https://example.com", html)
    assert metadata.title == "Multi-Image Post"
    assert len(metadata.image_urls) == 3
    assert "https://example.com/image1.jpg" in metadata.image_urls
    assert "https://example.com/image2.jpg" in metadata.image_urls
    assert "https://example.com/image3.jpg" in metadata.image_urls


def test_format_metadata_multiple_images():
    """Test formatting metadata with multiple images."""
    metadata = URLMetadata(
        url="https://example.com",
        title="Multi-Image Post",
        description="A post with multiple images",
        image_urls=[
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg",
        ],
    )
    formatted = format_metadata_for_llm(metadata)
    assert "Multi-Image Post" in formatted
    assert "Images (3):" in formatted
    assert "1. https://example.com/image1.jpg" in formatted
    assert "2. https://example.com/image2.jpg" in formatted
    assert "3. https://example.com/image3.jpg" in formatted


def test_format_metadata_exclude_images():
    """Test that include_images=False excludes image URLs."""
    metadata = URLMetadata(
        url="https://example.com",
        title="Test Title",
        description="Test Description",
        image_urls=["https://example.com/image.jpg"],
    )
    formatted = format_metadata_for_llm(metadata, include_images=False)
    assert "Test Title" in formatted
    assert "Test Description" in formatted
    assert "https://example.com/image.jpg" not in formatted
