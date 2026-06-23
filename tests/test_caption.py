from __future__ import annotations

from io import BytesIO

from PIL import Image

from hollingsbot.caption import add_caption, calculate_font_size


def _caption_band_fraction(width: int, caption: str) -> float:
    """Return the caption band height as a fraction of the image width."""
    img = Image.new("RGB", (width, width), (100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    out = add_caption(buf.getvalue(), caption)
    result = Image.open(BytesIO(out))
    band_height = result.height - width
    return band_height / width


def test_font_size_scales_with_image_width() -> None:
    caption = "A short caption."
    assert calculate_font_size(caption, 1024) == 40
    assert calculate_font_size(caption, 2048) == 80
    assert calculate_font_size(caption, 4096) == 160


def test_no_regression_at_baseline() -> None:
    assert calculate_font_size("x" * 50, 1024) == 40
    assert calculate_font_size("x" * 150, 1024) == 30
    assert calculate_font_size("x" * 250, 1024) == 20


def test_no_shrink_below_baseline() -> None:
    # Images smaller than 1024px keep the baseline sizing (no shrinking).
    assert calculate_font_size("x" * 50, 512) == 40
    assert calculate_font_size("x" * 50, 100) == 40


def test_caption_band_proportional_across_widths() -> None:
    caption = "A short witty caption for the image."
    fractions = [_caption_band_fraction(w, caption) for w in (1024, 2048, 4096)]
    # The caption band should occupy roughly the same fraction of the frame
    # at every width.
    for frac in fractions:
        assert abs(frac - fractions[0]) < 0.005


def test_empty_caption_returns_input_unchanged() -> None:
    img = Image.new("RGB", (1024, 1024), (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    assert add_caption(data, "") == data
