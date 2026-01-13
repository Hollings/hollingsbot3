# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Discord bot (`hollingsbot3`) built with discord.py, running in Docker with Celery workers for asynchronous task processing. The bot provides LLM chat, image generation (including editing), starboard functionality, and various utilities.

## Architecture

### Service Architecture
The bot runs as three separate Docker containers orchestrated by docker-compose:
- **bot**: Main Discord bot (no GPU, uses watchfiles for auto-reload)
- **celery_text**: Text generation worker (queue: `text`, concurrency: 1, no GPU)
- **celery_image**: Image generation worker (queue: `image`, concurrency: 2, requires NVIDIA GPU)
- **redis**: Message broker for Celery tasks

All services share the same codebase mounted at `/app`, with state persisted in Docker volumes (`prompt_db_data`).

### Core Components

**Cog System**: The bot uses discord.py's Cog pattern for modular functionality:
- `general.py` - Basic commands (ping, help)
- `image_gen_cog.py` - Image generation/editing via Celery
- `llm_chat.py` - LLM-powered chat with conversation history
- `gpt2_chat.py` - Legacy GPT-2 chat
- `admin.py` - Admin commands
- `gif_chain.py` - GIF utilities
- `starboard.py` - Message reposting on reactions (optional)

**Task Queue**: `tasks.py` defines Celery tasks with specific queue routing:
- `generate_image` → `image` queue (GPU-accelerated)
- `generate_text` → `text` queue
- `generate_llm_chat_response` → `text` queue

**Generator Pattern**: Both image and text generation use a factory pattern:
- `image_generators/__init__.py::get_image_generator(api, model)` - Returns `ImageGeneratorAPI` instance
  - Supports: `replicate` (ReplicateImageGenerator), `svg`/`openai-svg` (SvgGPTImageGenerator)
- `text_generators/__init__.py::get_text_generator(api, model)` - Returns `TextGeneratorAPI` instance
  - Supports: `huggingface`, `anthropic`, `openai`/`chatgpt`

### Image Generation Configuration

Image prompts are routed by prefix via `image_gen_config.json`:
```json
{
  "!": {"api": "replicate", "model": "black-forest-labs/flux-schnell"},
  "$": {"api": "replicate", "model": "google/imagen-4-fast", "daily_limit": 200},
  "edit:": {"api": "replicate", "model": "google/nano-banana", "mode": "edit"}
}
```

Daily limits are tracked per user in the SQLite database (`hollingsbot.db`). The `edit:` prefix enables image editing mode where the generator receives `image_input` (attachment bytes).

### LLM Chat Architecture

The `llm_chat.py` cog maintains per-channel conversation history using a deque-based structure:
- **History Management**: `channel_histories: dict[int, Deque[ConversationTurn]]` stores message history per channel
- **History Warming**: On first message in a channel, the bot backfills history by reading past Discord messages
- **Turn Structure**: Each turn contains role (user/assistant), text content, images (as ImageAttachment with data URLs), and Discord metadata (message_id, author_id)
- **Reply Context**: When users reply to messages, the bot automatically includes the replied-to message's content and images in context
- **Image Handling**: Attachments are downloaded, resized (max 2048px), converted to JPEG, and base64-encoded as data URLs
- **Text Attachments**: Uploaded text files are inlined into the conversation (max 120KB)
- **SVG Rendering**: SVG code blocks are automatically converted to PNG (requires cairosvg)
- **Model Selection**: Users can set per-user model preferences via `!model` command, stored in state JSON

The system prompt is loaded from `config/system_prompt.txt` (or env var `SYSTEM_PROMPT_FILE`) with mtime-based caching. Users can override via `!system <prompt>` or reset with `!system reset`.

### Conversation Summarization

When `LLM_SUMMARY_ENABLED=1`, the bot uses progressive summarization to reduce token costs and enable prompt caching:

**How it works:**
- Messages are cached to SQLite as they arrive
- Background worker generates 30-minute window summaries using Claude Haiku
- Summaries are combined into higher levels (1hr, 2hr, 4hr, etc.)
- LLM context becomes: `[system] + [summaries (cached)] + [recent raw messages]`

**Benefits:**
- ~75% reduction in input tokens (summaries vs full history)
- Anthropic prompt caching on stable summary prefix (~90% cheaper cache hits)
- Summaries persist across bot restarts

**Files:**
- `src/hollingsbot/summarization/` - Core summarization logic
- Summary data stored in unified `hollingsbot.db` (tables: `message_groups`, `cached_messages`, `channel_clear_points`)

### Temporary LLM Bots

The `temp_bot_cog.py` allows spawning temporary webhook-based LLM bots that share conversation history with the main bot and can converse with each other:

**Core Behavior**:
- **Spawning**: Use `!spawn <reply_count> <initial_prompt>` to create a temporary bot (max 100 replies)
- **Initial Response**: The temp bot immediately responds to the initial prompt using the channel's conversation context
- **Auto-Generated Names**: Bot names are randomly generated (e.g., "Swift Fox", "Wise Owl")
- **Reply Limit**: Each temp bot has a limited number of replies (decremented with each response)
- **Bot-to-Bot Conversation**: Temp bots can respond to each other AND to the main bot, enabling multi-bot conversations
- **Random Response Selection**: When a user or bot sends a message, there's a 50% chance a temp bot will respond (if any exist)
- **Auto-Despawn**: Temp bots automatically delete when they run out of replies
- **Manual Removal**: Use `!despawn <name>` to manually remove a temp bot, or `!despawn` to list active bots

**Implementation Details**:
- **Shared History**: Temp bots and the main bot share the same per-channel conversation history (`channel_histories[channel_id]`)
- **Webhook-Based**: Uses Discord webhooks for custom names and avatars
- **Database Tracking**: Metadata stored in `temp_bots` table (channel_id, webhook_id, name, avatar_url, replies_remaining)
- **Auto-Cleanup**: Background task runs every 60 seconds to delete depleted temp bots
- **Conversation Tracking**: `ConversationTurn` includes `webhook_id` field for proper attribution

**Example Flow**:
1. User: `!spawn 10 convince wendy to cheer up`
2. Bot creates "Clever Owl" with 10 replies
3. Clever Owl immediately responds to the prompt
4. Main bot and Clever Owl can now respond to each other's messages
5. Clever Owl auto-despawns after 10 replies

### Command Prefix System

The bot uses dynamic prefixes to prevent conflicts between image generation and commands:
- **Mention prefix**: Always enabled (`@bot command`)
- **Bang prefix (`!`)**: Disabled in image-gen channels (defined by `STABLE_DIFFUSION_CHANNEL_IDS`) to allow image prompts like `!a cat`
- Image-gen cogs listen on `on_message` and check for configured prefixes directly

## Development Workflow

### Running the Bot

**Start/Restart** (required after every code change):
```bash
docker compose down && docker compose up -d
```

**View logs**:
```bash
docker compose logs -f bot
docker compose logs -f celery_text
docker compose logs -f celery_image
```

**Build from scratch**:
```bash
docker compose up --build
```

**For ARM/Orange Pi** (uses CPU-only, no GPU):
```bash
docker compose -f docker-compose.yml -f docker-compose.arm.yml up --build
```

### Auto-reload

The bot and celery workers use `watchfiles` to monitor Python files and auto-restart on changes. However, **you must still restart the containers** via `docker compose down && docker compose up -d` for code changes to take effect, as watchfiles only reloads within the running container.

### Testing

Tests run outside Docker and require local dependencies:
```bash
pip install -r requirements.txt
pytest
```

### State Persistence

- **SQLite database** (`hollingsbot.db` in `/data` volume): Unified database storing all bot data including image generation tracking, cost tracking, message history, summaries, starboard posts, temp bots, and more
- **State files** in `generated/`: LLM chat state (`llm_chat_new_state.json` with system prompts and model preferences)
- **Generated images**: Written to `generated/` directory, mounted as volume

## Environment Variables

Key configuration (see `.env` file):
- **Core**: `DISCORD_TOKEN`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
- **LLM**: `DEFAULT_LLM_PROVIDER` (openai/anthropic), `DEFAULT_LLM_MODEL`, `TEXT_TIMEOUT` (default 180s)
- **LLM History**: `LLM_HISTORY_LIMIT` (default 50), `LLM_MAX_TURNS_SENT` (default 8)
- **Summarization**: `LLM_SUMMARY_ENABLED` (0/1), `LLM_SUMMARY_MODEL` (default claude-haiku-4-5), `LLM_SUMMARY_CHAR_LIMIT` (default 8000)
- **Image Gen**: `IMAGE_TIMEOUT` (default 30s), `IMAGE_OUTPUT_DIR`
- **Channel Allowlists**: `STABLE_DIFFUSION_CHANNEL_IDS` (comma-separated), `EDIT_CHANNEL_IDS`, `LLM_WHITELIST_CHANNELS`
- **Starboard**: `ENABLE_STARBOARD`, `STARBOARD_CHANNEL_ID`, `STARBOARD_IGNORE_CHANNELS`, `STARBOARD_WHITELIST_CHANNEL_IDS`
- **System Prompt**: `SYSTEM_PROMPT_FILE` (path to custom prompt file)
- **Webhooks**: `WEBHOOK_URL` (for PR notifications)
- **Database**: `PROMPT_DB_PATH` (default `/data/hollingsbot.db`) - unified database for all bot data

## Adding New Features

### Adding a New Image Generator
1. Create a new class in `image_generators/` implementing `ImageGeneratorAPI`
2. Register in `image_generators/__init__.py::get_image_generator()`
3. Add entry to `image_gen_config.json` with a unique prefix

### Adding a New Text Generator
1. Create a new class in `text_generators/` implementing `TextGeneratorAPI`
2. Register in `text_generators/__init__.py::get_text_generator()`
3. Update environment variables for default provider/model

### Adding a New Cog
1. Create a new file in `cogs/` inheriting from `commands.Cog`
2. Register in `__main__.py` via `await bot.load_extension("hollingsbot.cogs.your_cog")`
3. Restart the bot

## Important Notes

- **GPU Requirement**: The `celery_image` worker requires an NVIDIA GPU with the NVIDIA Container Toolkit installed on the host
- **Privacy Mode**: Set `STABLE_DIFFUSION_PRIVACY=1` to disable logging of message content
- **Bot Restart Interval**: The bot auto-restarts every 6 hours by default (`BOT_RESTART_INTERVAL` in seconds)
- **Conversation History**: LLM chat history is in-memory only and cleared on bot restart
- **Model Availability**: Check `src/hollingsbot/available_models.json` for valid LLM models (loaded dynamically by `llm_chat.py`)

## Wendy Deployment System

Wendy can deploy static websites and multiplayer game servers to `wendy.monster` using a unified deployment script.

### Architecture

```
Wendy (Claude CLI in celery_text)
      |
      | runs ./scripts/wendy/deploy.sh <project-path> [target-url]
      v
wendy_proxy (this repo) ────────────────────────────────────────────┐
      |                                                              |
      | POST /api/deploy_site                POST /api/deploy_game   |
      | (adds WENDY_DEPLOY_TOKEN)            (adds WENDY_GAMES_TOKEN)|
      v                                                              v
wendy-sites (Orange Pi:8910)              wendy-games manager (Orange Pi:8920)
      |                                                              |
      v                                                              v
https://wendy.monster/<name>/             Docker: wendy-game-<name>
                                                   |
                                                   v
                                          https://wendy.monster/game/<name>/
                                          wss://wendy.monster/game/<name>/ws
```

### Unified Deploy Script

Location: `scripts/wendy/deploy.sh`

**Auto-detection:**
- Has `server.ts` → Game server deployment
- Has `index.html` → Static site deployment

**Usage:**
```bash
deploy.sh <project-path> [target-url]

# Examples:
deploy.sh landing              # site -> wendy.monster/landing/
deploy.sh landing my-site      # site -> wendy.monster/my-site/
deploy.sh snake-game           # game -> wendy.monster/game/snake-game/
```

### Environment Variables

In `.env`:
- `WENDY_DEPLOY_TOKEN` - Token for wendy-sites service
- `WENDY_GAMES_TOKEN` - Token for wendy-games service

In `docker-compose.yml`, wendy_proxy receives both tokens:
```yaml
wendy_proxy:
  environment:
    - WENDY_DEPLOY_TOKEN=${WENDY_DEPLOY_TOKEN}
    - WENDY_GAMES_TOKEN=${WENDY_GAMES_TOKEN}
```

### Related Files

- `scripts/wendy/deploy.sh` - Unified deployment script
- `src/hollingsbot/wendy_proxy.py` - FastAPI proxy with `/api/deploy_site` and `/api/deploy_game`
- `config/system_prompt.txt` - Wendy's instructions (includes deployment docs)
- `data/wendy/wendys_folder/` - Wendy's workspace for projects
