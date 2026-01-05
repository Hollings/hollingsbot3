"""Utilities for generating outpainting images and masks."""

from __future__ import annotations

from io import BytesIO
from PIL import Image


def create_outpaint_images(image_bytes: bytes) -> tuple[bytes, bytes]:
    """Create scaled image and mask for outpainting.

    Args:
        image_bytes: Original image bytes

    Returns:
        Tuple of (scaled_image_bytes, mask_bytes) where:
        - scaled_image: Original image scaled down 50% in center of canvas
        - mask: Black in center (original image area), white on outside (generation area)
    """
    # Load the original image
    original = Image.open(BytesIO(image_bytes))

    # Get original dimensions
    orig_width, orig_height = original.size

    # Calculate scaled dimensions (50% of original)
    scaled_width = orig_width // 2
    scaled_height = orig_height // 2

    # Create canvas at original size with black background
    canvas = Image.new("RGB", (orig_width, orig_height), color=(0, 0, 0))

    # Scale down the original image
    scaled_image = original.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    # Calculate position to paste scaled image in center
    paste_x = (orig_width - scaled_width) // 2
    paste_y = (orig_height - scaled_height) // 2

    # Paste scaled image in center
    canvas.paste(scaled_image, (paste_x, paste_y))

    # Create mask image (black center, white outside)
    # Black = keep original, White = generate new content
    mask = Image.new("RGB", (orig_width, orig_height), color=(255, 255, 255))

    # Create black rectangle in center where the scaled image is
    black_center = Image.new("RGB", (scaled_width, scaled_height), color=(0, 0, 0))
    mask.paste(black_center, (paste_x, paste_y))

    # Convert both to bytes
    canvas_buffer = BytesIO()
    canvas.save(canvas_buffer, format="PNG")
    canvas_bytes = canvas_buffer.getvalue()

    mask_buffer = BytesIO()
    mask.save(mask_buffer, format="PNG")
    mask_bytes = mask_buffer.getvalue()

    return canvas_bytes, mask_bytes
