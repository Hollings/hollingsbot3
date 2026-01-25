"""Utilities for extracting Open Graph and Twitter Card metadata from URLs."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from dataclasses import dataclass

import aiohttp
from bs4 import BeautifulSoup
from PIL import Image

_LOG = logging.getLogger(__name__)

# Regex pattern for detecting URLs in text
# Matches http(s):// URLs or www. URLs, excluding common trailing punctuation
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"]+[^\s<>\".,!?;:)\]}>]|www\.[^\s<>\"]+[^\s<>\".,!?;:)\]}>]",
    re.IGNORECASE
)

_MAX_CONTENT_LENGTH = 5_000_000  # 5MB limit for fetched HTML
_REQUEST_TIMEOUT = 10  # seconds
_IMAGE_MAX_EDGE = 2048  # Max dimension for images sent to LLM
_IMAGE_MAX_BYTES = 9_500_000  # Max size for processed images


@dataclass
class ImageAttachment:
    """Image attachment for LLM conversation."""
    name: str
    url: str | None
    data_url: str | None
    width: int | None = None
    height: int | None = None
    size: int | None = None


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


def _extract_tweet_id(url: str) -> str | None:
    """Extract tweet ID from a Twitter/X URL.

    Args:
        url: The Twitter/X URL

    Returns:
        Tweet ID if found, None otherwise
    """
    # Match patterns like:
    # https://twitter.com/user/status/1234567890
    # https://x.com/user/status/1234567890
    match = re.search(r"/status/(\d+)", url)
    return match.group(1) if match else None


def _generate_syndication_token(tweet_id: str) -> str:
    """Generate token for Twitter syndication API.

    Args:
        tweet_id: The numeric tweet ID

    Returns:
        Generated token string
    """
    import math
    tweet_id_num = int(tweet_id)
    value = (tweet_id_num / 1e15) * math.pi
    token = ""
    # Convert to base 36
    base36_str = ""
    int_part = int(value)
    while int_part > 0:
        digit = int_part % 36
        if digit < 10:
            base36_str = str(digit) + base36_str
        else:
            base36_str = chr(ord('a') + digit - 10) + base36_str
        int_part //= 36

    # Remove leading zeros and dots
    token = re.sub(r'(0+|\.)', '', base36_str)
    return token if token else "0"


async def fetch_twitter_syndication_data(url: str) -> URLMetadata | None:
    """Fetch tweet data from Twitter's syndication endpoint.

    This endpoint provides full tweet data including all images,
    unlike Open Graph metadata which only includes the first image.

    Args:
        url: The Twitter/X URL

    Returns:
        URLMetadata object if successful, None otherwise
    """
    tweet_id = _extract_tweet_id(url)
    if not tweet_id:
        _LOG.debug("Could not extract tweet ID from URL: %s", url)
        return None

    token = _generate_syndication_token(tweet_id)
    syndication_url = f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token={token}"

    try:
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.get(syndication_url) as response:
            if response.status != 200:
                _LOG.warning("Syndication API returned HTTP %d for tweet %s", response.status, tweet_id)
                return None

            data = await response.json()

        # Extract data from syndication response
        metadata = URLMetadata(url=url)

        # Get text
        metadata.title = data.get("user", {}).get("name", "") + " on X"
        metadata.description = data.get("text", "")
        metadata.site_name = "X (formerly Twitter)"

        # Extract all image URLs from photos
        photos = data.get("photos", [])
        for photo in photos:
            img_url = photo.get("url")
            if img_url and img_url not in metadata.image_urls:
                metadata.image_urls.append(img_url)

        # Also check for video thumbnail
        video = data.get("video")
        if video:
            thumb_url = video.get("poster")
            if thumb_url and thumb_url not in metadata.image_urls:
                metadata.image_urls.append(thumb_url)

        _LOG.info(
            "Syndication API: Extracted %d image(s) from tweet %s",
            len(metadata.image_urls),
            tweet_id
        )

        return metadata

    except Exception as exc:
        _LOG.warning("Failed to fetch Twitter syndication data for %s: %s", url, exc)
        return None


async def fetch_url_metadata(url: str) -> URLMetadata | None:
    """Fetch Open Graph/Twitter Card metadata from a URL.

    For Twitter/X URLs, uses the syndication endpoint to get all images.
    For other URLs, falls back to Open Graph/Twitter Card metadata.

    Args:
        url: The URL to fetch metadata from

    Returns:
        URLMetadata object if successful, None otherwise
    """
    # Try Twitter syndication endpoint first for Twitter/X URLs
    if "twitter.com" in url.lower() or "x.com" in url.lower():
        twitter_metadata = await fetch_twitter_syndication_data(url)
        if twitter_metadata:
            return twitter_metadata
        _LOG.debug("Twitter syndication failed, falling back to Open Graph for %s", url)

    # Fall back to Open Graph metadata for non-Twitter URLs or if syndication fails
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
    except Exception as exc:
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


def _encode_jpeg(image: Image.Image) -> bytes:
    """Encode an image as JPEG with compression to meet size limits."""
    for quality in (90, 85, 80, 75, 70, 60, 50):
        out = io.BytesIO()
        image.save(out, format="JPEG", optimize=True, quality=quality)
        if out.tell() <= _IMAGE_MAX_BYTES:
            return out.getvalue()
    return out.getvalue()


async def download_and_process_image(image_url: str, index: int = 1) -> ImageAttachment | None:
    """Download and process an image URL for LLM consumption.

    Args:
        image_url: The URL of the image to download
        index: Index number for naming (default: 1)

    Returns:
        ImageAttachment object with processed image data, or None if failed
    """
    try:
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; DiscordBot/1.0; +https://discord.com)"
            }
            async with session.get(image_url, headers=headers) as response:
                if response.status != 200:
                    _LOG.warning("Failed to download image from %s: HTTP %d", image_url, response.status)
                    return None

                image_data = await response.read()

        # Process the image
        with Image.open(io.BytesIO(image_data)) as img:
            img = img.convert("RGB")
            width, height = img.size
            longest = max(width, height)

            # Resize if needed
            if longest > _IMAGE_MAX_EDGE:
                scale = _IMAGE_MAX_EDGE / float(longest)
                resized = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                img = img.resize(resized, Image.LANCZOS)
                width, height = img.size

            jpeg_bytes = _encode_jpeg(img)

        # Create data URL
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")

        return ImageAttachment(
            name=f"url_image_{index}.jpg",
            url=image_url,
            data_url=data_url,
            width=width,
            height=height,
            size=len(jpeg_bytes),
        )

    except Exception as exc:
        _LOG.exception("Failed to download/process image from %s: %s", image_url, exc)
        return None


async def download_images_from_metadata(metadata: URLMetadata, max_images: int = 4) -> list[ImageAttachment]:
    """Download and process images from URL metadata.

    Args:
        metadata: URLMetadata object containing image URLs
        max_images: Maximum number of images to download (default: 4)

    Returns:
        List of ImageAttachment objects with processed image data
    """
    if not metadata.image_urls:
        return []

    # Limit number of images to avoid using too many tokens
    image_urls = metadata.image_urls[:max_images]

    # Download images in parallel
    tasks = [
        download_and_process_image(img_url, index=i+1)
        for i, img_url in enumerate(image_urls)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failures and exceptions
    images = []
    for result in results:
        if isinstance(result, ImageAttachment):
            images.append(result)
        elif isinstance(result, Exception):
            _LOG.warning("Image download failed with exception: %s", result)

    return images


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
