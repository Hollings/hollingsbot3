import asyncio
import os
import time
from inspect import signature
from pathlib import Path

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

# Directory that both the worker *and* the Discord bot container can see
OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", "/app/generated"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@celery_app.task(
    name="tasks.generate_image",
    queue="image",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def generate_image(  # noqa: C901
    self,
    prompt_id: int,
    api: str,
    model: str,
    prompt: str,
    seed: int | None = None,
    *,
    timeout: float = float(os.getenv("IMAGE_TIMEOUT", "30.0")),
) -> str:
    """
    Generate an image, **write it to disk**, and return the file‑path.

    No raw image bytes are pushed through Redis anymore.
    """
    start = time.monotonic()
    logger.info(
        "generate_image[%s] START | api=%s model=%s seed=%s prompt=%s",
        prompt_id,
        api,
        model,
        seed,
        prompt,
    )
    update_status(prompt_id, "started")

    generator = get_image_generator(api, model)
    gen_sig = signature(generator.generate)

    async def _run() -> bytes:
        kwargs = {"seed": seed} if "seed" in gen_sig.parameters else {}
        return await asyncio.wait_for(generator.generate(prompt, **kwargs), timeout)

    try:
        image_bytes: bytes = asyncio.run(_run())
    except asyncio.TimeoutError as exc:
        err = f"Generation exceeded {timeout}s timeout."
        logger.error("generate_image[%s] TIMEOUT: %s", prompt_id, err)
        update_status(prompt_id, f"failed: {err}")
        raise RuntimeError(err) from exc
    except Exception as exc:
        duration = time.monotonic() - start
        logger.exception(
            "generate_image[%s] FAILED after %.2fs | %s", prompt_id, duration, exc
        )
        update_status(prompt_id, f"failed: {exc}")
        raise
    finally:
        try:
            asyncio.run(generator.aclose())
        except Exception:
            logger.debug("generate_image[%s] aclose() raised, ignored.", prompt_id)

    # Persist to shared directory
    filename = f"{prompt_id}_{int(time.time())}.png"
    file_path = OUTPUT_DIR / filename
    file_path.write_bytes(image_bytes)

    update_status(prompt_id, "completed")
    duration = time.monotonic() - start
    logger.info(
        "generate_image[%s] FINISH in %.2fs | wrote %s (size=%d bytes)",
        prompt_id,
        duration,
        file_path,
        len(image_bytes),
    )
    return str(file_path)

@celery_app.task(name="tasks.generate_text", queue="text")
def generate_text(
    api: str,
    model: str,
    prompt: str,
    temperature: float = 1.0,
) -> str:
    """Generate text with detailed logging, supporting a configurable temperature."""
    start = time.monotonic()
    logger.info(
        "generate_text START | api=%s model=%s temp=%.2f prompt_len=%d",
        api,
        model,
        temperature,
        len(prompt),
    )

    try:
        generator = get_text_generator(api, model)
        gen_sig = signature(generator.generate)
        if "temperature" in gen_sig.parameters:
            text = asyncio.run(generator.generate(prompt, temperature=temperature))
        else:
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
