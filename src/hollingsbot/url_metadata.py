"""Utilities for extracting Open Graph and Twitter Card metadata from URLs."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import aiohttp
from bs4 import BeautifulSoup

_LOG = logging.getLogger(__name__)

# Regex pattern for detecting URLs in text
# Matches http(s):// URLs or www. URLs, excluding common trailing punctuation
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"]+[^\s<>\".,!?;:)\]}>]|www\.[^\s<>\"]+[^\s<>\".,!?;:)\]}>]",
    re.IGNORECASE
)

_MAX_CONTENT_LENGTH = 5_000_000  # 5MB limit for fetched HTML
_REQUEST_TIMEOUT = 10  # seconds


@dataclass
class URLMetadata:
    """Metadata extracted from a URL."""
    url: str
    title: str | None = None
    description: str | None = None
    image_urls: list[str] = None  # type: ignore[assignment]
    site_name: str | None = None

    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []


async def extract_urls(text: str) -> list[str]:
    """Extract all URLs from the given text.

    Args:
        text: The text to search for URLs

    Returns:
        List of URLs found in the text
    """
    urls = _URL_PATTERN.findall(text)
    # Ensure URLs start with http:// or https://
    normalized_urls = []
    for url in urls:
        if url.startswith("www."):
            normalized_urls.append(f"https://{url}")
        else:
            normalized_urls.append(url)
    return normalized_urls


async def fetch_url_metadata(url: str) -> URLMetadata | None:
    """Fetch Open Graph/Twitter Card metadata from a URL.

    Args:
        url: The URL to fetch metadata from

    Returns:
        URLMetadata object if successful, None otherwise
    """
    try:
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        # Increase header size limits for sites like X/Twitter that send large cookies
        connector = aiohttp.TCPConnector(limit_per_host=5)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            max_line_size=16384,
            max_field_size=16384
        ) as session:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; DiscordBot/1.0; +https://discord.com)"
            }
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                # Check content length
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > _MAX_CONTENT_LENGTH:
                    _LOG.warning("Skipping URL %s: content too large (%s bytes)", url, content_length)
                    return None

                # Only process HTML content
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    _LOG.debug("Skipping URL %s: not HTML (Content-Type: %s)", url, content_type)
                    return None

                html = await response.text()

        metadata = _parse_metadata(url, html)

        # Log extracted metadata for debugging
        if metadata.image_urls:
            _LOG.info(
                "Extracted %d image(s) from %s: %s",
                len(metadata.image_urls),
                url,
                ", ".join(metadata.image_urls)
            )

        return metadata

    except asyncio.TimeoutError:
        _LOG.warning("Timeout fetching URL: %s", url)
        return None
    except aiohttp.ClientError as exc:
        _LOG.warning("Failed to fetch URL %s: %s", url, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("Unexpected error fetching URL %s: %s", url, exc)
        return None


def _parse_metadata(url: str, html: str) -> URLMetadata:
    """Parse Open Graph and Twitter Card metadata from HTML.

    Args:
        url: The URL being parsed (for the metadata object)
        html: The HTML content to parse

    Returns:
        URLMetadata object with extracted information
    """
    soup = BeautifulSoup(html, "html.parser")
    metadata = URLMetadata(url=url)

    # Try Open Graph tags first
    og_title = soup.find("meta", property="og:title")
    if og_title:
        metadata.title = og_title.get("content")

    og_description = soup.find("meta", property="og:description")
    if og_description:
        metadata.description = og_description.get("content")

    # Extract ALL og:image tags (Twitter/X posts often have multiple images)
    og_images = soup.find_all("meta", property="og:image")
    for og_image in og_images:
        img_url = og_image.get("content")
        if img_url and img_url not in metadata.image_urls:
            metadata.image_urls.append(img_url)

    og_site_name = soup.find("meta", property="og:site_name")
    if og_site_name:
        metadata.site_name = og_site_name.get("content")

    # Fall back to Twitter Card tags if Open Graph tags are not available
    if not metadata.title:
        twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
        if twitter_title:
            metadata.title = twitter_title.get("content")

    if not metadata.description:
        twitter_description = soup.find("meta", attrs={"name": "twitter:description"})
        if twitter_description:
            metadata.description = twitter_description.get("content")

    # Also extract Twitter Card images (in addition to OG images)
    if not metadata.image_urls:
        twitter_images = soup.find_all("meta", attrs={"name": "twitter:image"})
        for twitter_image in twitter_images:
            img_url = twitter_image.get("content")
            if img_url and img_url not in metadata.image_urls:
                metadata.image_urls.append(img_url)

    # Fall back to standard HTML tags if still missing
    if not metadata.title:
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata.title = title_tag.string.strip()

    if not metadata.description:
        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description:
            metadata.description = meta_description.get("content")

    return metadata


def format_metadata_for_llm(metadata: URLMetadata, include_images: bool = True) -> str:
    """Format URL metadata for inclusion in LLM conversation.

    Args:
        metadata: The URLMetadata object to format
        include_images: Whether to include image URLs (default: True)

    Returns:
        Formatted string for LLM context
    """
    parts = [f"[Link metadata for {metadata.url}]"]

    if metadata.site_name:
        parts.append(f"Site: {metadata.site_name}")

    if metadata.title:
        parts.append(f"Title: {metadata.title}")

    if metadata.description:
        # Truncate long descriptions
        desc = metadata.description
        if len(desc) > 300:
            desc = desc[:297] + "..."
        parts.append(f"Description: {desc}")

    if include_images and metadata.image_urls:
        if len(metadata.image_urls) == 1:
            parts.append(f"Image: {metadata.image_urls[0]}")
        else:
            parts.append(f"Images ({len(metadata.image_urls)}):")
            for i, img_url in enumerate(metadata.image_urls, 1):
                parts.append(f"  {i}. {img_url}")

    return "\n".join(parts)


async def extract_url_metadata(text: str, max_urls: int = 3) -> list[URLMetadata]:
    """Extract URLs from text and fetch their metadata.

    Args:
        text: The text to search for URLs
        max_urls: Maximum number of URLs to process (default: 3)

    Returns:
        List of URLMetadata objects (only those with title or description)
    """
    urls = await extract_urls(text)
    if not urls:
        return []

    # Limit to avoid processing too many URLs
    urls = urls[:max_urls]

    metadata_list = []
    for url in urls:
        metadata = await fetch_url_metadata(url)
        if metadata and (metadata.title or metadata.description):
            metadata_list.append(metadata)

    return metadata_list


async def extract_and_format_url_metadata(text: str, max_urls: int = 3, include_images: bool = True) -> str:
    """Extract URLs from text and fetch their metadata.

    Args:
        text: The text to search for URLs
        max_urls: Maximum number of URLs to process (default: 3)
        include_images: Whether to include image URLs in formatted output (default: True)

    Returns:
        Formatted metadata string, or empty string if no URLs found
    """
    urls = await extract_urls(text)
    if not urls:
        return ""

    # Limit to avoid processing too many URLs
    urls = urls[:max_urls]

    formatted_parts = []
    for url in urls:
        metadata = await fetch_url_metadata(url)
        if metadata and (metadata.title or metadata.description):
            formatted_parts.append(format_metadata_for_llm(metadata, include_images=include_images))

    return "\n\n".join(formatted_parts)
