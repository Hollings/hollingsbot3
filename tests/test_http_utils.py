from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp.resolver import ThreadedResolver
from yarl import URL

from hollingsbot.utils import http_utils


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://10.0.0.1",
        "http://169.254.169.254/latest",
        "http://[::1]",
        "http://[::ffff:127.0.0.1]",
        "http://100.64.0.1",
        "http://224.0.0.1",
        "file:///etc/passwd",
        "http://user:password@example.com",
    ],
)
def test_private_or_non_http_destinations_rejected(url):
    with pytest.raises(ValueError):
        http_utils.validate_public_url(url)


async def test_private_dns_answers_rejected(monkeypatch):
    monkeypatch.setattr(ThreadedResolver, "resolve", AsyncMock(return_value=[{"host": "127.0.0.1"}]))
    with pytest.raises(OSError, match="non-public"):
        await http_utils.PublicResolver().resolve("looks-public.example")


async def test_public_dns_answers_preserved(monkeypatch):
    answers = [{"host": "8.8.8.8"}]
    monkeypatch.setattr(ThreadedResolver, "resolve", AsyncMock(return_value=answers))
    assert await http_utils.PublicResolver().resolve("example.com") == answers


async def test_streamed_limit_without_content_length():
    async def chunks(size):
        yield b"abc"
        yield b"def"

    response = SimpleNamespace(content=SimpleNamespace(iter_chunked=chunks))
    with pytest.raises(ValueError, match="exceeds"):
        await http_utils.read_limited(response, 5)
    assert await http_utils.read_limited(response, 6) == b"abcdef"


async def test_redirect_to_private_host_is_blocked_before_second_request(monkeypatch):
    response = SimpleNamespace(status=302, headers={"Location": "http://127.0.0.1/admin"})
    response_context = AsyncMock()
    response_context.__aenter__.return_value = response
    session = Mock()
    session.get.return_value = response_context
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    monkeypatch.setattr(http_utils.aiohttp, "ClientSession", Mock(return_value=session_context))
    monkeypatch.setattr(http_utils.aiohttp, "TCPConnector", Mock())
    with pytest.raises(ValueError, match="non-public"):
        await http_utils.download_public_url("https://example.com", max_bytes=100)
    session.get.assert_called_once()
    assert session.get.call_args.args == (URL("https://example.com"),)
