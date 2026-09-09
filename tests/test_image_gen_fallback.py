"""Tests for GeneratorSpec fallback chains in the image generation cog.

Covers config parsing (nested ``fallback`` dicts) and the runtime behaviour of
``_execute_generation_tasks`` when the primary generator raises.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from hollingsbot.cogs.image_gen_cog import GeneratorSpec, ImageGenCog, spec_from_dict


class TestSpecFromDict:
    def test_plain_spec_has_no_fallback(self):
        spec = spec_from_dict({"api": "replicate", "model": "a/b", "go_fast": True})
        assert spec.fallback is None
        assert spec.model_options == {"go_fast": True}

    def test_fallback_is_parsed_recursively(self):
        spec = spec_from_dict(
            {
                "api": "replicate",
                "model": "openai/gpt-image-2.5-sunburst",
                "mode": "edit",
                "quality": "medium",
                "price_per_image": 0.04,
                "fallback": {"api": "replicate", "model": "google/nano-banana-2"},
            }
        )
        assert spec.model == "openai/gpt-image-2.5-sunburst"
        assert spec.fallback is not None
        assert spec.fallback.model == "google/nano-banana-2"
        # "fallback" must not leak into model_options sent to the API
        assert spec.model_options is None

    def test_fallback_inherits_mode_and_quality_but_not_price(self):
        spec = spec_from_dict(
            {
                "api": "replicate",
                "model": "x",
                "mode": "edit",
                "quality": "high",
                "price_per_image": 0.5,
                "fallback": {"api": "replicate", "model": "y"},
            }
        )
        assert spec.fallback.mode == "edit"
        assert spec.fallback.quality == "high"
        assert spec.fallback.price_per_image is None

    def test_fallback_can_override_inherited_fields(self):
        spec = spec_from_dict(
            {
                "api": "replicate",
                "model": "x",
                "quality": "high",
                "fallback": {"api": "replicate", "model": "y", "quality": "low"},
            }
        )
        assert spec.fallback.quality == "low"

    def test_chained_fallbacks(self):
        spec = spec_from_dict(
            {
                "api": "a",
                "model": "1",
                "fallback": {"api": "b", "model": "2", "fallback": {"api": "c", "model": "3"}},
            }
        )
        assert spec.fallback.fallback.model == "3"
        assert spec.fallback.fallback.fallback is None

    def test_shipped_config_edit_prefix_falls_back_to_nano_banana(self):
        from hollingsbot.cogs.image_gen_cog import _DEFAULT_CONFIG_PATH

        raw = json.loads(_DEFAULT_CONFIG_PATH.read_text("utf8"))
        spec = spec_from_dict(raw["edit:"])
        assert spec.model.startswith("openai/gpt-image-2.5")
        assert spec.mode == "edit"
        assert spec.fallback is not None
        assert spec.fallback.model == "google/nano-banana-2"
        assert spec.fallback.mode == "edit"


class _FakeCog(ImageGenCog):
    """ImageGenCog with __init__ bypassed and _run_task scripted."""

    def __init__(self, outcomes: dict[str, object]):
        # Deliberately skip ImageGenCog.__init__ (needs a bot + DB).
        self.calls: list[str] = []
        self._outcomes = outcomes

    async def _run_task(self, prompt_id, api, model, prompt, seed, **kwargs):  # type: ignore[override]
        self.calls.append(model)
        outcome = self._outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _load_generation_result(self, value):  # type: ignore[override]
        return value.encode() if isinstance(value, str) else value


def _run(cog: _FakeCog, spec: GeneratorSpec):
    return asyncio.run(cog._execute_generation_tasks([1], ["a cat"], spec, 42, [b"img"], None, True, None))


class TestFallbackExecution:
    PRIMARY = GeneratorSpec(
        api="replicate",
        model="primary",
        mode="edit",
        fallback=GeneratorSpec(api="replicate", model="secondary", mode="edit"),
    )

    def test_primary_success_does_not_touch_fallback(self):
        cog = _FakeCog({"primary": "ok.png", "secondary": "no.png"})
        results = _run(cog, self.PRIMARY)
        assert results == [("a cat", [b"ok.png"])]
        assert cog.calls == ["primary"]

    def test_primary_failure_uses_fallback(self):
        cog = _FakeCog({"primary": RuntimeError("moderation blocked"), "secondary": "fb.png"})
        results = _run(cog, self.PRIMARY)
        assert results == [("a cat", [b"fb.png"])]
        assert cog.calls == ["primary", "secondary"]

    def test_timeout_also_triggers_fallback(self):
        cog = _FakeCog({"primary": TimeoutError("slow"), "secondary": "fb.png"})
        results = _run(cog, self.PRIMARY)
        assert results == [("a cat", [b"fb.png"])]

    def test_both_failing_returns_last_exception(self):
        cog = _FakeCog({"primary": RuntimeError("first"), "secondary": RuntimeError("second")})
        results = _run(cog, self.PRIMARY)
        assert isinstance(results[0], RuntimeError)
        assert str(results[0]) == "second"
        assert cog.calls == ["primary", "secondary"]

    def test_no_fallback_returns_exception(self):
        spec = GeneratorSpec(api="replicate", model="primary")
        cog = _FakeCog({"primary": RuntimeError("boom")})
        results = _run(cog, spec)
        assert isinstance(results[0], RuntimeError)
        assert cog.calls == ["primary"]

    def test_fallback_receives_its_own_quality_and_options(self):
        seen: dict[str, dict] = {}

        class _Cog(_FakeCog):
            async def _run_task(self, prompt_id, api, model, prompt, seed, **kwargs):  # type: ignore[override]
                seen[model] = kwargs
                return await super()._run_task(prompt_id, api, model, prompt, seed, **kwargs)

        spec = GeneratorSpec(
            api="replicate",
            model="primary",
            quality="medium",
            fallback=GeneratorSpec(api="replicate", model="secondary", quality="low", model_options={"k": 1}),
        )
        cog = _Cog({"primary": RuntimeError("x"), "secondary": "fb.png"})
        _run(cog, spec)
        assert seen["primary"]["quality"] == "medium"
        assert seen["secondary"]["quality"] == "low"
        assert seen["secondary"]["model_options"] == {"k": 1}


@pytest.mark.parametrize("bad", [None, "nope", 3])
def test_non_dict_fallback_is_ignored(bad):
    spec = spec_from_dict({"api": "a", "model": "b", "fallback": bad})
    assert spec.fallback is None
    assert spec.model_options is None
