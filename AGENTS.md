# Repository Guidelines

## Project Structure & Modules
- Source: `src/hollingsbot/` (entrypoint: `python -m hollingsbot`).
- Cogs: `src/hollingsbot/cogs/` (commands/features by domain).
- Workers: Celery tasks in `src/hollingsbot/tasks.py` using Redis.
- Generators: `src/hollingsbot/image_generators/` and `src/hollingsbot/text_generators/`.
- Utils/Assets: `src/hollingsbot/utils/`, runtime outputs in `generated/`.
- Config: `.env` for secrets, `src/hollingsbot/image_gen_config.json` for image prefix→provider mapping.

## Build, Test, and Development
- Build/run all services: `docker-compose up --build` (bot, Redis, Celery workers).
- Local bot only: `pip install -r requirements.txt && python -m hollingsbot`.
- Celery workers (outside Docker):
  - Text queue: `celery -A hollingsbot.tasks worker -Q text --loglevel=info`
  - Image queue: `celery -A hollingsbot.tasks worker -Q image --loglevel=info`
- Tests: `pytest` (supports `pytest-asyncio`). Run a file: `pytest tests/test_image_gen_cog.py -q`.

## Coding Style & Naming
- Python 3, PEP 8, 4-space indentation. Prefer type hints and short docstrings.
- Modules under `hollingsbot` use absolute imports (e.g., `from hollingsbot.cogs import ...`).
- Names: snake_case for functions/variables, PascalCase for classes, lowercase module filenames.
- Cogs: one feature per file in `cogs/`, expose `async def setup(bot): ...`.
- Keep tasks idempotent; log clearly without leaking secrets.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio` for coroutine tests (`@pytest.mark.asyncio`).
- Location: create `tests/` with `test_*.py` files mirroring package structure.
- Aim for fast unit tests of cogs, generators, and task routing. Add regression tests with minimal fixtures.

## Commit & Pull Request Guidelines
- Commits: use clear, imperative subjects; Conventional Commits (`feat:`, `fix:`) preferred.
- PRs: include summary, rationale, test instructions, and linked issues. Add screenshots for user-facing Discord commands when helpful.
- Keep diffs focused; update README/`image_gen_config.json` if behavior or prefixes change.

## Security & Configuration Tips
- Required env: `DISCORD_TOKEN`, `GITHUB_TOKEN` (repo scope). Common optional: `WEBHOOK_URL`, `ENABLE_STARBOARD`, `STARBOARD_CHANNEL_ID`, `STABLE_DIFFUSION_CHANNEL_IDS`, `STABLE_DIFFUSION_PRIVACY`, `PROMPT_DB_PATH`, `BOT_RESTART_INTERVAL`.
- Do not commit `.env` or tokens; avoid logging secrets.
- Redis backs Celery; SQLite stores lightweight state. Review data paths before deploying.

