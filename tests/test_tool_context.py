import asyncio

from hollingsbot.tools.parser import get_current_context, set_current_context


async def test_concurrent_tool_responses_keep_their_channel_context():
    ready, release = asyncio.Event(), asyncio.Event()

    async def first():
        set_current_context({"channel_id": 1})
        ready.set()
        await release.wait()
        return get_current_context()

    async def second():
        await ready.wait()
        set_current_context({"channel_id": 2})
        release.set()
        return get_current_context()

    assert await asyncio.gather(first(), second()) == [{"channel_id": 1}, {"channel_id": 2}]
