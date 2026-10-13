"""Bounded HTTP downloads for links supplied in chat messages."""

import asyncio
import ipaddress
import socket
from dataclasses import dataclass

import aiohttp
from aiohttp.resolver import ThreadedResolver
from yarl import URL


def _public_address(host: str) -> bool:
    address = ipaddress.ip_address(host)
    return address.is_global and not address.is_multicast


def validate_public_url(url: str) -> URL:
    parsed = URL(url)
    if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.user is not None:
        raise ValueError("Expected a public HTTP(S) URL without credentials")
    try:
        ipaddress.ip_address(parsed.host)
    except ValueError:
        pass  # DNS names are checked by the resolver at connection time.
    else:
        if not _public_address(parsed.host):
            raise ValueError("Links to non-public addresses are not supported")
    return parsed


class PublicResolver(ThreadedResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        addresses = await super().resolve(host, port, family)
        if not addresses or any(not _public_address(item["host"]) for item in addresses):
            raise OSError("Refusing a non-public destination")
        return addresses


async def read_limited(response, max_bytes: int) -> bytes:
    """Cap actual decoded bytes, including chunked or compressed responses."""
    data = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        if len(data) + len(chunk) > max_bytes:
            raise ValueError(f"Response exceeds {max_bytes} bytes")
        data.extend(chunk)
    return bytes(data)


@dataclass(frozen=True)
class Download:
    data: bytes
    url: str
    content_type: str
    charset: str


async def download_public_url(url: str, *, max_bytes: int, timeout: float = 10, html_only: bool = False) -> Download:
    """Check every redirect and resolved address before connecting; never fetch LAN services."""

    async def download():
        connector = aiohttp.TCPConnector(resolver=PublicResolver(), limit_per_host=5)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout),
            max_line_size=16384,
            max_field_size=16384,
        ) as session:
            current = url
            for _ in range(6):
                destination = validate_public_url(current)
                async with session.get(
                    destination,
                    allow_redirects=False,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; DiscordBot/1.0; +https://discord.com)"},
                ) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location:
                            raise ValueError("Redirect has no destination")
                        current = str(destination.join(URL(location)))
                        continue
                    response.raise_for_status()
                    if html_only and response.content_type != "text/html":
                        raise ValueError("Response is not HTML")
                    data = await read_limited(response, max_bytes)
                    return Download(data, str(response.url), response.content_type, response.charset or "utf-8")
            raise ValueError("Too many redirects")

    # Keep the total redirect chain within one deadline.
    return await asyncio.wait_for(download(), timeout)
