import asyncio
import base64
import os

from celery import Celery

from image_generators import ReplicateImageGenerator

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
)

@celery_app.task
def generate_image(api: str, model: str, prompt: str) -> str:
    if api == 'replicate':
        generator = ReplicateImageGenerator(model)
    else:
        raise ValueError(f'Unknown API: {api}')
    image_bytes = asyncio.run(generator.generate(prompt))
    return base64.b64encode(image_bytes).decode()
