import asyncio
import base64
import os

from celery import Celery

from image_generators import get_image_generator
from text_generators import get_text_generator
from prompt_db import update_status

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
)

@celery_app.task
def generate_image(prompt_id: int, api: str, model: str, prompt: str) -> str:
    update_status(prompt_id, "started")
    try:
        generator = get_image_generator(api, model)
        image_bytes = asyncio.run(generator.generate(prompt))
    except Exception as e:  # noqa: BLE001
        update_status(prompt_id, f"failed: {e}")
        raise
    update_status(prompt_id, "completed")
    return base64.b64encode(image_bytes).decode()


@celery_app.task
def generate_text(api: str, model: str, prompt: str) -> str:
    """Generate text using the specified provider."""
    generator = get_text_generator(api, model)

    text = asyncio.run(generator.generate(prompt))
    return text
