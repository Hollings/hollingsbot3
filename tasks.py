import asyncio
import base64
import os
import time

from celery import Celery
from celery.utils.log import get_task_logger

from image_generators import get_image_generator
from text_generators import get_text_generator
from prompt_db import update_status

logger = get_task_logger(__name__)

celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

celery_app.conf.task_routes = {
    "tasks.generate_text":  {"queue": "text"},
    "tasks.generate_image": {"queue": "image"},
}

@celery_app.task(name="tasks.generate_image", queue="image")
def generate_image(prompt_id: int, api: str, model: str, prompt: str) -> str:
    """Generate an image, encode it as base64, and return the string."""
    start = time.monotonic()
    logger.info(
        "generate_image[%s] START | api=%s model=%s prompt=%s",
        prompt_id,
        api,
        model,
        prompt,
    )

    update_status(prompt_id, "started")
    try:
        generator = get_image_generator(api, model)
        image_bytes = asyncio.run(generator.generate(prompt))
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start
        logger.exception(
            "generate_image[%s] FAILED after %.2fs | %s",
            prompt_id,
            duration,
            exc,
        )
        update_status(prompt_id, f"failed: {exc}")
        raise

    update_status(prompt_id, "completed")
    duration = time.monotonic() - start
    logger.info(
        "generate_image[%s] FINISH in %.2fs | size=%d bytes",
        prompt_id,
        duration,
        len(image_bytes),
    )
    return base64.b64encode(image_bytes).decode()


@celery_app.task(name="tasks.generate_text", queue="text")
def generate_text(api: str, model: str, prompt: str) -> str:
    """Generate text using the specified provider with detailed logging."""
    start = time.monotonic()
    logger.info(
        "generate_text START | api=%s model=%s prompt_len=%d",
        api,
        model,
        len(prompt),
    )

    try:
        generator = get_text_generator(api, model)
        text = asyncio.run(generator.generate(prompt))
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start
        logger.exception("generate_text FAILED after %.2fs | %s", duration, exc)
        raise

    duration = time.monotonic() - start
    logger.info(
        "generate_text FINISH in %.2fs | output_len=%d",
        duration,
        len(text),
    )
    return text
