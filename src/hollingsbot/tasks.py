import asyncio
import os
import time
from inspect import signature
from pathlib import Path

from celery import Celery
from celery.utils.log import get_task_logger

from hollingsbot.image_generators import get_image_generator
from hollingsbot.prompt_db import log_llm_api_call, update_status
from hollingsbot.text_generators import get_text_generator

logger = get_task_logger(__name__)

celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

celery_app.conf.task_routes = {
    "tasks.generate_text":  {"queue": "text"},
    "tasks.generate_image": {"queue": "image"},
    "tasks.generate_llm_chat_response": {"queue": "text"},
    "tasks.repair_wendy": {"queue": "text"},
}

OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", "/app/generated"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@celery_app.task(
    name="tasks.generate_image",
    queue="image",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def generate_image(
    self,
    prompt_id: int,
    api: str,
    model: str,
    prompt: str,
    seed: int | None = None,
    *,
    # New optional kwargs for editing and outpainting
    image_input: list[bytes] | None = None,
    mask: str | None = None,
    output_format: str | None = None,
    quality: str = "medium",
    aspect_ratio: str | None = None,
    model_options: dict | None = None,  # Extra model-specific options
    # Existing kwarg
    timeout: float = float(os.getenv("IMAGE_TIMEOUT", "30.0")),
) -> str | list[str]:
    """
    Generate an image, write it to disk, and return the file path.
    Supports editing when image_input is provided, and outpainting when mask is also provided.
    """
    start = time.monotonic()
    logger.info(
        "generate_image[%s] START | api=%s model=%s seed=%s quality=%s prompt=%s images=%s mask=%s fmt=%s",
        prompt_id,
        api,
        model,
        seed,
        quality,
        prompt,
        (len(image_input) if image_input else 0),
        ("provided" if mask else "none"),
        output_format,
    )
    update_status(prompt_id, "started")

    generator = get_image_generator(api, model, quality=quality, aspect_ratio=aspect_ratio, model_options=model_options)
    gen_sig = signature(generator.generate)
    gen_many = getattr(generator, "generate_many", None)
    gen_many_sig = signature(gen_many) if callable(gen_many) else None

    async def _run_single() -> bytes:
        kwargs = {}
        if "seed" in gen_sig.parameters:
            kwargs["seed"] = seed
        if image_input and "image_input" in gen_sig.parameters:
            kwargs["image_input"] = image_input
        if mask and "mask" in gen_sig.parameters:
            kwargs["mask"] = mask
        if output_format and "output_format" in gen_sig.parameters:
            kwargs["output_format"] = output_format
        return await asyncio.wait_for(generator.generate(prompt, **kwargs), timeout)

    async def _run_many() -> list[bytes]:
        assert gen_many_sig is not None
        kwargs = {}
        if "seed" in gen_many_sig.parameters:
            kwargs["seed"] = seed
        if image_input and "image_input" in gen_many_sig.parameters:
            kwargs["image_input"] = image_input
        if mask and "mask" in gen_many_sig.parameters:
            kwargs["mask"] = mask
        if output_format and "output_format" in gen_many_sig.parameters:
            kwargs["output_format"] = output_format
        return await asyncio.wait_for(gen_many(prompt, **kwargs), timeout)  # type: ignore[misc]

    try:
        # Prefer multi-image generation when available
        if callable(gen_many):
            images_bytes: list[bytes] = asyncio.run(_run_many())
        else:
            image_bytes: bytes = asyncio.run(_run_single())
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

    ts = int(time.time())
    if callable(gen_many):
        # Write all images and return list of paths
        if not images_bytes:
            raise RuntimeError("Generator returned no images.")
        paths: list[str] = []
        for idx, img in enumerate(images_bytes, start=1):
            filename = f"{prompt_id}_{ts}_{idx:02d}.png"
            fp = OUTPUT_DIR / filename
            fp.write_bytes(img)
            paths.append(str(fp))
        update_status(prompt_id, "completed")
        duration = time.monotonic() - start
        logger.info(
            "generate_image[%s] FINISH in %.2fs | wrote %d files", prompt_id, duration, len(paths)
        )
        return paths
    else:
        filename = f"{prompt_id}_{ts}.png"
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
    except Exception as exc:
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


def _image_placeholder(img: dict[str, object]) -> str:
    """Return a short textual placeholder for an image attachment."""
    name = str(img.get("name") or "image")
    url = img.get("url")
    if isinstance(url, str) and url:
        return f"{name} <{url}>"
    dims = []
    width = img.get("width")
    height = img.get("height")
    if isinstance(width, int) and isinstance(height, int):
        dims.append(f"{width}x{height}")
    size = img.get("size")
    if isinstance(size, int) and size > 0:
        dims.append(f"{size} bytes")
    if dims:
        return f"{name} ({', '.join(dims)})"
    return name


def _with_image_placeholders(text: str, images: list[dict[str, object]]) -> str:
    if not images:
        return text
    placeholders = ", ".join(_image_placeholder(img) for img in images)
    text = text.strip()
    if text:
        return f"{text}\n[images: {placeholders}]"
    return f"[images: {placeholders}]"


def _build_messages_for_generator(
    api: str,
    conversation: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert a normalized conversation into provider-specific message payloads."""

    api_normalized = api.lower().strip()
    messages: list[dict[str, object]] = []

    for entry in conversation:
        role = str(entry.get("role") or "user")
        text = str(entry.get("text") or "").strip()
        images = entry.get("images") or []
        if not isinstance(images, list):
            images = []

        if api_normalized in {"openai", "chatgpt", "openrouter"} and role == "user":
            content_parts: list[dict[str, object]] = []
            if text:
                content_parts.append({"type": "input_text", "text": text})
            for img in images:
                data_url = img.get("data_url")
                url = img.get("url")
                image_payload: dict[str, object] | None = None
                if isinstance(data_url, str) and data_url:
                    image_payload = {"type": "input_image", "image_url": data_url, "detail": "auto"}
                elif isinstance(url, str) and url:
                    image_payload = {"type": "image_url", "image_url": {"url": url}}
                if image_payload:
                    content_parts.append(image_payload)
            if not content_parts:
                content: str | list[dict[str, object]] = text or "[user attachment without text]"
            else:
                content = content_parts
        elif api_normalized == "anthropic" and role == "user" and images:
            # Anthropic uses content blocks for images
            content_parts: list[dict[str, object]] = []
            if text:
                content_parts.append({"type": "text", "text": text})
            for img in images:
                data_url = img.get("data_url")
                if isinstance(data_url, str) and data_url and data_url.startswith("data:"):
                    # Extract media type and base64 data
                    try:
                        header, b64_data = data_url.split(",", 1)
                        media_type = header.split(";")[0].replace("data:", "")
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        })
                    except Exception:
                        logger.warning("Failed to parse data URL for image %s", img.get("name"))
            if not content_parts:
                content = text or "[user attachment without text]"
            else:
                content = content_parts
        else:
            # For non-OpenAI/Anthropic providers (and assistant/system turns),
            # collapse images into lightweight textual markers to preserve context
            content = _with_image_placeholders(text, images)

        messages.append({"role": role, "content": content})

    return messages


def _conversation_to_text(conversation: list[dict[str, object]]) -> str:
    """Fallback string prompt for providers without structured chat support."""

    blocks: list[str] = []
    for entry in conversation:
        role = str(entry.get("role") or "user")
        text = str(entry.get("text") or "")
        images = entry.get("images") or []
        if isinstance(images, list):
            text = _with_image_placeholders(text, images)
        blocks.append(f"[{role}] {text}".strip())
    return "\n\n".join(block for block in blocks if block)


@celery_app.task(name="tasks.generate_llm_chat_response", queue="text")
def generate_llm_chat_response(
    api: str,
    model: str,
    conversation: list[dict[str, object]],
    *,
    temperature: float = 1.0,
    channel_id: int | None = None,
) -> dict[str, object]:
    """Generate an LLM chat response based on structured conversation history."""

    start = time.monotonic()
    logger.info(
        "generate_llm_chat_response START | api=%s model=%s turns=%d",
        api,
        model,
        len(conversation),
    )

    generator = get_text_generator(api, model)
    gen_sig = signature(generator.generate)
    kwargs: dict[str, object] = {}
    if "temperature" in gen_sig.parameters:
        kwargs["temperature"] = temperature
    # Pass channel_id to generators that support it (e.g., claude-cli)
    if "channel_id" in gen_sig.parameters and channel_id:
        kwargs["channel_id"] = channel_id

    api_normalized = api.lower().strip()

    # Serialize conversation for logging
    import json
    import traceback
    conversation_json = json.dumps(conversation, indent=2)
    response_text = ""
    status = "error"
    error_message = None
    error_traceback = None

    try:
        if api_normalized in {"openai", "chatgpt", "openrouter"} or api_normalized == "anthropic":
            payload = _build_messages_for_generator(api_normalized, conversation)
        elif api_normalized == "claude-cli":
            # Pass conversation directly - claude_cli.py handles its own formatting
            payload = conversation
        else:
            payload = _conversation_to_text(conversation)

        text = asyncio.run(generator.generate(payload, **kwargs))
        response_text = text
        status = "success"
    except TypeError:
        # Some generators insist on plain text; fall back to flattened transcript.
        text = asyncio.run(generator.generate(_conversation_to_text(conversation)))
        response_text = text
        status = "success"
    except Exception as exc:
        duration = time.monotonic() - start
        error_message = str(exc)
        error_traceback = traceback.format_exc()
        logger.exception(
            "generate_llm_chat_response FAILED after %.2fs | %s",
            duration,
            exc,
        )
        # Log failed API call
        log_llm_api_call(
            provider=api,
            model=model,
            conversation_json=conversation_json,
            response_text="",
            status="error",
            error_message=error_message,
        )
        raise
    finally:
        # Log successful API call
        if status == "success":
            log_llm_api_call(
                provider=api,
                model=model,
                conversation_json=conversation_json,
                response_text=response_text,
                status="success",
            )

    duration = time.monotonic() - start
    logger.info(
        "generate_llm_chat_response FINISH in %.2fs | output_len=%d",
        duration,
        len(text),
    )

    # Build debug info for return
    debug_info = {
        "provider": api,
        "model": model,
        "status": status,
        "duration": duration,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "temperature": temperature,
    }

    # Try to get token usage from generator if available
    token_usage = getattr(generator, 'last_token_usage', None)
    if token_usage:
        debug_info["token_usage"] = token_usage

    return {
        "text": text,
        "debug": debug_info,
        "conversation": conversation,
    }


@celery_app.task(
    name="tasks.repair_wendy",
    queue="text",
    bind=True,
)
def repair_wendy(self, channel_id: int) -> dict:
    """Run Claude Code to debug and repair Wendy when she's not responding.

    This task runs a fresh Claude CLI session with a debugging prompt that:
    1. Investigates why Wendy isn't responding
    2. Checks logs, session files, and error messages
    3. Attempts to fix the issue
    4. Sends a summary to the channel via send_message.sh
    """
    import shutil
    import subprocess

    logger.info("repair_wendy START for channel %d", channel_id)
    start = time.monotonic()

    # Find claude CLI
    cli_path = shutil.which("claude")
    if not cli_path:
        return {"success": False, "error": "Claude CLI not found"}

    repair_prompt = f'''EMERGENCY REPAIR MODE - Channel {channel_id}

CRITICAL: You are debugging why Wendy (the Discord bot) is not responding. This is NOT for adding features or minor issues - only for fixing crashes and critical failures.

INVESTIGATION STEPS:
1. Check recent celery_text logs for errors:
   - Look for ClaudeCliError, session errors, API errors
   - Check if tasks are being received but failing

2. Check Wendy's session state:
   - Read /data/wendy/session_state.json
   - If session exists, try to identify corruption issues

3. Check recent debug logs:
   - List /data/wendy/debug_logs/ for recent activity
   - Read the most recent ones to see what happened

4. Common fixes:
   - If session is corrupted: Delete the session entry from session_state.json
   - If CLI is failing: Check error messages in logs
   - If rate limited: Note this in summary

5. If you edited Python code in /app/, restart the affected container:
   docker compose restart celery_text   # For Wendy/text generation issues
   docker compose restart bot           # For Discord bot issues

6. After investigating and fixing, send a SHORT summary to the channel:
   ./send_message.sh {channel_id} "your summary here"

IMPORTANT:
- Do NOT add new features
- Do NOT refactor code
- Do NOT fix minor issues like double-posting
- ONLY fix critical crashes that prevent Wendy from responding
- Keep your fix minimal and targeted
- Always send a summary message when done
- If you edit code, you MUST restart the relevant container

START by checking the docker logs and session state.'''

    try:
        # Run claude CLI with the repair prompt (fresh session, no resume)
        result = subprocess.run(
            [
                cli_path,
                "-p",  # Print mode
                "--model", "opus",
                "--allowedTools", "Read,Bash,Edit,Write",
            ],
            input=repair_prompt,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd="/data/wendy",
        )

        duration = time.monotonic() - start
        logger.info("repair_wendy FINISH in %.2fs | returncode=%d", duration, result.returncode)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        logger.error("repair_wendy TIMEOUT after %.2fs", duration)
        return {"success": False, "error": "Repair timed out after 5 minutes"}
    except Exception as exc:
        duration = time.monotonic() - start
        logger.exception("repair_wendy FAILED after %.2fs", duration)
        return {"success": False, "error": str(exc)}
