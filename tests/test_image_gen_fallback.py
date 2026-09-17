"""Tests for GeneratorSpec fallback chains in the image generation cog.

Covers config parsing (nested ``fallback`` dicts) and the runtime behaviour of
``_execute_generation_tasks`` when the primary generator raises.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from hollingsbot.cogs.image_gen_cog import GeneratorSpec, ImageGenCog, build_prefix_map, spec_from_dict


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


class TestPrefixReferenceFallback:
    def test_string_fallback_resolves_to_named_prefix(self):
        specs = build_prefix_map(
            {
                "a:": {"api": "replicate", "model": "one", "mode": "edit", "fallback": "b:"},
                "b:": {"api": "replicate", "model": "two", "mode": "edit", "some_opt": 1, "price_per_image": 0.04},
            }
        )
        assert specs["a:"].fallback == specs["b:"]
        assert specs["a:"].fallback.model_options == {"some_opt": 1}
        # The top-level prefix's price is what gets charged; "a:" has none of its own
        assert specs["a:"].price_per_image is None

    def test_reference_inside_nested_fallback(self):
        specs = build_prefix_map(
            {
                "a:": {"api": "x", "model": "one", "fallback": {"api": "x", "model": "two", "fallback": "b:"}},
                "b:": {"api": "x", "model": "three"},
            }
        )
        assert specs["a:"].fallback.model == "two"
        assert specs["a:"].fallback.fallback.model == "three"
        assert specs["a:"].fallback.fallback.fallback is None

    def test_referenced_prefix_brings_its_own_chain(self):
        specs = build_prefix_map(
            {
                "a:": {"api": "x", "model": "one", "fallback": "b:"},
                "b:": {"api": "x", "model": "two", "fallback": {"api": "x", "model": "three"}},
            }
        )
        assert specs["a:"].fallback.fallback.model == "three"

    def test_referenced_prefix_does_not_inherit_from_referrer(self):
        specs = build_prefix_map(
            {
                "a:": {"api": "x", "model": "one", "mode": "edit", "quality": "high", "fallback": "b:"},
                "b:": {"api": "x", "model": "two"},
            }
        )
        assert specs["a:"].fallback.mode == "generate"
        assert specs["a:"].fallback.quality == "medium"

    def test_unknown_reference_is_dropped(self, caplog):
        specs = build_prefix_map({"a:": {"api": "x", "model": "one", "fallback": "nope:"}})
        assert specs["a:"].fallback is None
        assert "unknown prefix 'nope:'" in caplog.text

    def test_self_reference_is_dropped(self, caplog):
        specs = build_prefix_map({"a:": {"api": "x", "model": "one", "fallback": "a:"}})
        assert specs["a:"].fallback is None
        assert "Circular" in caplog.text

    def test_mutual_cycle_terminates(self, caplog):
        specs = build_prefix_map(
            {
                "a:": {"api": "x", "model": "one", "fallback": "b:"},
                "b:": {"api": "x", "model": "two", "fallback": "a:"},
            }
        )
        # Each prefix still gets the other as a fallback; only the closing edge is cut
        assert specs["a:"].fallback.model == "two"
        assert specs["a:"].fallback.fallback is None
        assert specs["b:"].fallback.model == "one"
        assert specs["b:"].fallback.fallback is None
        assert "Circular" in caplog.text

    def test_prefix_keys_are_stripped(self):
        specs = build_prefix_map(
            {" a: ": {"api": "x", "model": "one", "fallback": " b:"}, "b: ": {"api": "x", "model": "two"}}
        )
        assert set(specs) == {"a:", "b:"}
        assert specs["a:"].fallback.model == "two"


class TestShippedEditChain:
    def _specs(self) -> dict[str, GeneratorSpec]:
        from hollingsbot.cogs.image_gen_cog import _DEFAULT_CONFIG_PATH

        raw = json.loads(_DEFAULT_CONFIG_PATH.read_text("utf8"))
        return build_prefix_map({k: v for k, v in raw.items() if isinstance(v, dict)})

    def test_edit_ends_in_edit_low(self):
        specs = self._specs()
        chain = []
        spec = specs["edit:"]
        while spec is not None:
            chain.append(spec.model)
            spec = spec.fallback
        assert chain[0].startswith("openai/gpt-image-2.5")
        assert chain[1:] == ["google/nano-banana-2", "bytedance/seedream-4.5"]
        # The tail really is the edit low: spec, single-image options and all
        assert specs["edit:"].fallback.fallback == specs["edit low:"]
        # Still charged the base edit price, not edit low's
        assert specs["edit:"].price_per_image == 0.04

    def test_every_shipped_fallback_reference_resolves(self, caplog):
        self._specs()
        assert "fallback" not in caplog.text

    def test_chain_runs_through_to_seedream_with_its_options(self):
        seen: dict[str, dict] = {}
        specs = self._specs()
        primary = specs["edit:"].model

        class _Cog(_FakeCog):
            async def _run_task(self, prompt_id, api, model, prompt, seed, **kwargs):  # type: ignore[override]
                seen[model] = kwargs
                return await super()._run_task(prompt_id, api, model, prompt, seed, **kwargs)

        cog = _Cog(
            {
                primary: RuntimeError("moderation blocked"),
                "google/nano-banana-2": RuntimeError("also refused"),
                "bytedance/seedream-4.5": "sd.png",
            }
        )
        results = _run(cog, specs["edit:"])
        assert results == [("a cat", [b"sd.png"])]
        assert cog.calls == [primary, "google/nano-banana-2", "bytedance/seedream-4.5"]
        assert seen["bytedance/seedream-4.5"]["model_options"]["sequential_image_generation"] == "disabled"

    def test_model_listing_shows_full_chain(self):
        cog = _FakeCog({})
        cog._prefix_map = self._specs()
        cog._cfg_path = None
        cog._default_price = 0.03
        cog._allow_dms = True
        cog._allowed_channel_ids = set()
        cog._edit_channel_ids = set()
        listing = cog._format_model_listing()
        assert "replicate / google/nano-banana-2, then replicate / bytedance/seedream-4.5" in listing


class TestShippedEditLowPrefix:
    def _raw(self) -> dict:
        from hollingsbot.cogs.image_gen_cog import _DEFAULT_CONFIG_PATH

        return json.loads(_DEFAULT_CONFIG_PATH.read_text("utf8"))

    def test_edit_low_is_single_image_seedream_edit_at_four_cents(self):
        spec = spec_from_dict(self._raw()["edit low:"])
        assert spec.model == "bytedance/seedream-4.5"
        assert spec.mode == "edit"
        assert spec.price_per_image == 0.04
        # One image per charge: grouped generation must be off
        assert spec.model_options["sequential_image_generation"] == "disabled"

    def test_edit_low_prefix_wins_over_plain_edit(self):
        raw = self._raw()
        prefixes = {k: v for k, v in raw.items() if isinstance(v, dict)}
        cog = _FakeCog({})
        cog._prefix_map = build_prefix_map(prefixes)
        cog._cfg_path = None
        prompt, spec = cog._split_prompt("Edit Low: make the bird purple")
        assert prompt == "make the bird purple"
        assert spec.model == "bytedance/seedream-4.5"
        _, plain = cog._split_prompt("edit: make the bird purple")
        assert plain.model.startswith("openai/gpt-image")


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
