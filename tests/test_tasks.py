"""Worker failures must close resources and leave a terminal prompt status."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from hollingsbot import prompt_db, tasks
from hollingsbot.text_generators import client_cache
from hollingsbot.utils.celery_utils import RemoteOutcomeUnknown


@pytest.fixture
def worker(monkeypatch, tmp_path):
    status = Mock()
    monkeypatch.setattr(tasks, "begin_image_attempt", Mock(return_value=7))
    monkeypatch.setattr(tasks, "finish_image_attempt", status)
    monkeypatch.setattr(tasks, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(tasks, "log_llm_api_call", Mock())
    return status


@pytest.mark.parametrize("outcome", [b"image", RuntimeError("provider failed"), "timeout"])
def test_image_cleanup_runs_on_generation_loop(worker, monkeypatch, outcome):
    loops = []

    class Generator:
        async def generate(self, prompt):
            loops.append(asyncio.get_running_loop())
            if isinstance(outcome, Exception):
                raise outcome
            if outcome == "timeout":
                await asyncio.sleep(10)
            return outcome

        async def aclose(self):
            loops.append(asyncio.get_running_loop())

    monkeypatch.setattr(tasks, "get_image_generator", lambda *a, **kw: Generator())
    if outcome == b"image":
        path = tasks.generate_image.run(1, "fake", "model", "hi")
        assert Path(path).read_bytes() == b"image"
        assert worker.call_args.args == (7, "generated")
    else:
        with pytest.raises((RuntimeError, RemoteOutcomeUnknown)):
            tasks.generate_image.run(1, "fake", "model", "hi", timeout=0.01)
        assert worker.call_args.args[1] == ("outcome_unknown" if outcome == "timeout" else "failed")
    assert len(loops) == 2 and loops[0] is loops[1]


@pytest.mark.parametrize("failure", ["factory", "empty", "disk"])
def test_image_failures_always_update_status(worker, monkeypatch, failure):
    class Generator:
        async def generate_many(self, prompt):
            return [] if failure == "empty" else [b"image"]

        async def aclose(self):
            pass

    generator = Generator()
    factory = Mock(return_value=generator)
    if failure == "factory":
        factory.side_effect = ValueError("bad model")
    if failure == "disk":
        monkeypatch.setattr(tasks, "_ensure_output_dir", Mock(side_effect=OSError("disk full")))
    monkeypatch.setattr(tasks, "get_image_generator", factory)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        tasks.generate_image.run(1, "fake", "model", "hi")
    assert worker.call_args.args == (7, "failed")


def test_replayed_worker_task_does_not_call_provider_and_records_actual_model(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_db, "DB_PATH", tmp_path / "prompts.db")
    monkeypatch.setattr(tasks, "OUTPUT_DIR", tmp_path)
    provider = Mock()

    class Generator:
        async def generate(self, prompt):
            provider(prompt)
            return b"image"

        async def aclose(self):
            pass

    monkeypatch.setattr(tasks, "get_image_generator", lambda *_args, **_kwargs: Generator())
    prompt_id = prompt_db.bulk_add_prompts(["hi"], "1", "fake", "primary", request_message_id=1)[0]
    tasks.generate_image.push_request(id="original-task")
    try:
        tasks.generate_image.run(prompt_id, "fake", "fallback", "hi")
        with pytest.raises(RemoteOutcomeUnknown, match="already ran"):
            tasks.generate_image.run(prompt_id, "fake", "fallback", "hi")
    finally:
        tasks.generate_image.pop_request()
    provider.assert_called_once_with("hi")
    with prompt_db._connect() as conn:
        assert conn.execute("SELECT model, status FROM image_attempts").fetchall() == [("fallback", "generated")]
        assert conn.execute("SELECT status FROM prompts").fetchone() == ("queued",)


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_text_workers_close_clients_without_retrying_type_errors(worker, monkeypatch, chat, fail):
    class Client:
        def __init__(self):
            self.loop = asyncio.get_running_loop()
            self.closed = False

        async def close(self):
            assert asyncio.get_running_loop() is self.loop
            self.closed = True

    clients = []

    async def generate(prompt, temperature=1.0):
        client = client_cache.get_client("review-test", Client)
        clients.append(client)
        if fail:
            raise TypeError("bug inside provider")
        return "hello"

    monkeypatch.setattr(tasks, "get_text_generator", lambda *a: Mock(generate=generate))
    call = tasks.generate_llm_chat_response.run if chat else tasks.generate_text.run
    payload = [{"role": "user", "text": "hi"}] if chat else "hi"
    if fail:
        with pytest.raises(TypeError, match="bug inside provider"):
            call("openai", "model", payload)
    else:
        call("openai", "model", payload)
    assert len(clients) == 1
    assert clients[0].closed
    assert "review-test" not in client_cache._cache
    if chat:
        assert tasks.log_llm_api_call.call_args.kwargs["status"] == ("error" if fail else "success")
