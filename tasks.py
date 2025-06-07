import asyncio
import base64
import os

from celery import Celery

from image_generators import ReplicateImageGenerator
from prompt_db import update_status
from transformers import pipeline
import torch

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
)

@celery_app.task
def generate_image(prompt_id: int, api: str, model: str, prompt: str) -> str:
    update_status(prompt_id, "started")
    try:
        if api == 'replicate':
            generator = ReplicateImageGenerator(model)
        else:
            raise ValueError(f'Unknown API: {api}')
        image_bytes = asyncio.run(generator.generate(prompt))
    except Exception as e:  # noqa: BLE001
        update_status(prompt_id, f"failed: {e}")
        raise
    update_status(prompt_id, "completed")
    return base64.b64encode(image_bytes).decode()


@celery_app.task
def generate_text(model: str, prompt: str) -> str:
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-generation", model=model, device=device)
    data = generator(prompt, max_new_tokens=500)
    text = data[0]["generated_text"]
    return text[:2000].strip()
