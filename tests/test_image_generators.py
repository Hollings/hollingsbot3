"""Tests for the image_generators package (factory + Replicate helpers).

These cover pure/local logic only - no network calls are made.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from hollingsbot.image_generators import (
    ReplicateImageGenerator,
    SvgGPTImageGenerator,
    get_image_generator,
)
from hollingsbot.image_generators.replicate_api import _scale_image_to_max_dimension

TOKEN = "test-token"


@pytest.fixture(autouse=True)
def _replicate_token(monkeypatch):
    """Ensure a token is present so the factory/default-factory path works offline."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)


def _png_bytes(width: int, height: int, mode: str = "RGB") -> bytes:
    img = Image.new(mode, (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gen(**kwargs) -> ReplicateImageGenerator:
    kwargs.setdefault("api_token", TOKEN)
    return ReplicateImageGenerator(**kwargs)


class TestGetImageGenerator:
    """Factory dispatch tests."""

    def test_replicate(self):
        gen = get_image_generator("replicate", "black-forest-labs/flux-schnell")
        assert isinstance(gen, ReplicateImageGenerator)
        assert gen.model == "black-forest-labs/flux-schnell"

    def test_replicate_passes_options(self):
        gen = get_image_generator(
            "replicate",
            "some/model",
            quality="high",
            aspect_ratio="16:9",
            model_options={"go_fast": True},
        )
        assert gen.quality == "high"
        assert gen.aspect_ratio == "16:9"
        assert gen.model_options == {"go_fast": True}

    @pytest.mark.parametrize("api", ["svg", "openai-svg"])
    def test_svg(self, api):
        gen = get_image_generator(api, "gpt-5")
        assert isinstance(gen, SvgGPTImageGenerator)
        assert gen.model == "gpt-5"

    def test_unknown_api_raises(self):
        with pytest.raises(ValueError, match="Unknown API"):
            get_image_generator("nope", "model")


class TestReplicateInit:
    def test_requires_token(self, monkeypatch):
        monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="REPLICATE_API_TOKEN"):
            ReplicateImageGenerator(model="x", api_token="")


class TestModelCapabilityFlags:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("black-forest-labs/flux-schnell", True),
            ("prunaai/hidream-l1-fast", True),
            ("some/flux-thing", True),
            ("openai/gpt-image-2", False),
            ("bytedance/seedream-4.5", False),
        ],
    )
    def test_supports_seed(self, model, expected):
        assert _make_gen(model=model)._supports_seed() is expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("black-forest-labs/flux-2-max", True),
            ("some/flux-thing", True),
            ("openai/gpt-image-2", False),
            ("bytedance/seedream-4.5", False),
        ],
    )
    def test_supports_disable_safety(self, model, expected):
        assert _make_gen(model=model)._supports_disable_safety() is expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("bytedance/seedream-4.5", True),
            ("bytedance/seedream-4", True),
            ("black-forest-labs/flux-schnell", False),
        ],
    )
    def test_is_seedream(self, model, expected):
        assert _make_gen(model=model)._is_seedream() is expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("openai/gpt-image-2", True),
            ("openai/gpt-image-1.5", True),
            ("black-forest-labs/flux-schnell", False),
        ],
    )
    def test_is_gpt_image(self, model, expected):
        assert _make_gen(model=model)._is_gpt_image() is expected


class TestScaleImage:
    def test_no_scale_when_within_limit(self):
        out = _scale_image_to_max_dimension(_png_bytes(100, 80), 1024)
        assert Image.open(io.BytesIO(out)).size == (100, 80)

    def test_scales_down_preserving_aspect(self):
        out = _scale_image_to_max_dimension(_png_bytes(2048, 1024), 1024)
        assert Image.open(io.BytesIO(out)).size == (1024, 512)

    def test_converts_rgba_to_rgb(self):
        out = _scale_image_to_max_dimension(_png_bytes(50, 50, mode="RGBA"), 1024)
        assert Image.open(io.BytesIO(out)).mode == "RGB"


class TestBuildGptImageInputs:
    def test_basic_fields(self):
        gen = _make_gen(model="openai/gpt-image-2", quality="medium")
        inputs, cleanup = gen._build_gpt_image_inputs("a cat", None, None)
        assert inputs["prompt"] == "a cat"
        assert inputs["quality"] == "medium"
        assert inputs["moderation"] == "low"
        assert "input_images" not in inputs
        assert cleanup == []

    def test_includes_aspect_ratio_and_output_format(self):
        gen = _make_gen(model="openai/gpt-image-2", aspect_ratio="3:2")
        inputs, _ = gen._build_gpt_image_inputs("x", None, "png")
        assert inputs["aspect_ratio"] == "3:2"
        assert inputs["output_format"] == "png"

    def test_merges_model_options(self):
        gen = _make_gen(model="openai/gpt-image-2", model_options={"foo": "bar"})
        inputs, _ = gen._build_gpt_image_inputs("x", None, None)
        assert inputs["foo"] == "bar"

    def test_scales_and_prepares_image_input(self):
        gen = _make_gen(model="openai/gpt-image-2")
        big = _png_bytes(2048, 2048)
        inputs, cleanup = gen._build_gpt_image_inputs("x", [big], None)
        try:
            assert "input_images" in inputs
            assert len(inputs["input_images"]) == 1
            assert len(cleanup) == 1
        finally:
            gen._cleanup_files(cleanup)


class TestPrepareImageInputs:
    def test_passthrough_urls_and_data(self):
        gen = _make_gen(model="openai/gpt-image-2")
        items = ["https://example.com/a.png", "data:image/png;base64,AAAA"]
        prepared, cleanup = gen._prepare_image_inputs(items)
        assert prepared == items
        assert cleanup == []

    def test_bytes_written_to_tempfile_and_cleaned(self):
        gen = _make_gen(model="openai/gpt-image-2")
        prepared, cleanup = gen._prepare_image_inputs([b"\x89PNG\r\n"])
        assert len(prepared) == 1
        assert len(cleanup) == 1
        fh, path = cleanup[0]
        assert path is not None
        gen._cleanup_files(cleanup)
        assert fh.closed
        import os

        assert not os.path.exists(path)

    def test_empty_raises(self):
        gen = _make_gen(model="openai/gpt-image-2")
        with pytest.raises(RuntimeError, match="empty"):
            gen._prepare_image_inputs([])


class TestNormaliseOutput:
    async def test_bytes(self):
        gen = _make_gen(model="black-forest-labs/flux-schnell")
        assert await gen._normalise_output(b"abc") == b"abc"

    async def test_list_of_bytes_joined(self):
        gen = _make_gen(model="black-forest-labs/flux-schnell")
        assert await gen._normalise_output([b"ab", b"cd"]) == b"abcd"

    async def test_nested_sequence_takes_first(self):
        gen = _make_gen(model="black-forest-labs/flux-schnell")
        assert await gen._normalise_output([b"first"]) == b"first"

    async def test_empty_sequence_raises(self):
        gen = _make_gen(model="black-forest-labs/flux-schnell")
        with pytest.raises(RuntimeError, match="empty sequence"):
            await gen._normalise_output([])

    async def test_unsupported_type_raises(self):
        gen = _make_gen(model="black-forest-labs/flux-schnell")
        with pytest.raises(RuntimeError, match="Unsupported"):
            await gen._normalise_output(12345)


class TestCollectAll:
    async def test_bytes(self):
        gen = _make_gen(model="bytedance/seedream-4.5")
        assert await gen._collect_all(b"img") == [b"img"]

    async def test_list_of_bytes(self):
        gen = _make_gen(model="bytedance/seedream-4.5")
        assert await gen._collect_all([b"a", b"b"]) == [b"a", b"b"]

    async def test_unrecognized_returns_empty(self):
        gen = _make_gen(model="bytedance/seedream-4.5")
        assert await gen._collect_all(42) == []
