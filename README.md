# Hollingsbot3

A feature-rich Discord bot built with discord.py that provides LLM chat, image generation, image editing, GIF creation, and admin utilities. The bot runs as multiple Docker services with Celery workers for asynchronous task processing.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements and Setup](#requirements-and-setup)
- [Development Workflow](#development-workflow)
- [Cogs](#cogs)
- [Configuration](#configuration)
- [Architecture](#architecture)

## Project Overview

Hollingsbot3 is designed to integrate advanced AI capabilities into Discord:

- **LLM Chat**: Multi-turn conversation with Claude or GPT models, supporting images, URLs, file uploads, and tool execution
- **Image Generation**: Generate, edit, and extend images using various AI providers (Replicate, OpenAI)
- **Image Processing**: Create animated GIFs from reply chains
- **Starboard**: Mirror reacted messages to a dedicated channel
- **Admin Tools**: Container restart, credit management, and configuration management

## Requirements and Setup

### System Requirements

- **Python 3.10+**
- **Docker & Docker Compose** (for running the bot and workers)
- **NVIDIA GPU** (for the image generation worker; CPU-only mode not fully supported)
- **Redis** (message broker for Celery, via docker-compose)

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd hollingsbot3
```

2. Install local dependencies (for testing/development):
```bash
pip install -r requirements.txt
```

3. Configure environment variables by creating/updating `.env`:
```bash
# Discord
DISCORD_TOKEN=your_token_here

# LLM Configuration
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-3-5-sonnet-20241022
AVAILABLE_MODELS=anthropic/claude-3-5-sonnet-20241022,openai/gpt-4o
TEXT_TIMEOUT=180

# Image Generation
IMAGE_TIMEOUT=30
IMAGE_OUTPUT_DIR=generated/
DAILY_FREE_BUDGET=0.50

# Channel Configuration
STABLE_DIFFUSION_CHANNEL_IDS=123456789,987654321
EDIT_CHANNEL_IDS=111111111
LLM_WHITELIST_CHANNELS=222222222
STARBOARD_CHANNEL_ID=333333333
STARBOARD_IGNORE_CHANNELS=444444444

# LLM Chat
LLM_HISTORY_LIMIT=50
LLM_MAX_TURNS_SENT=8
GPT2_CHANNEL_ID=555555555

# Admin
ADMIN_USER_IDS=12345,67890

# Celery / Redis
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Database
PROMPT_DB_PATH=prompts.db
```

### Running the Bot

**Start the stack:**
```bash
docker-compose down && docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f bot              # Main bot
docker-compose logs -f celery_text      # Text generation worker
docker-compose logs -f celery_image     # Image generation worker
```

**Build from scratch:**
```bash
docker-compose up --build
```

## Development Workflow

### Auto-reload

The bot and workers use `watchfiles` to detect Python changes and auto-restart containers. However, you must still run `docker-compose down && docker-compose up -d` to apply changes, as watchfiles only reloads within the running container.

### Testing

Tests run outside Docker and require local dependencies:
```bash
pip install -r requirements.txt
pytest
```

### Code Organization

```
src/hollingsbot/
├── __main__.py              # Bot initialization and cog loading
├── cogs/                    # Discord cogs (modular features)
│   ├── general.py           # Basic commands (ping, help)
│   ├── image_gen_cog.py     # Image generation/editing/outpainting
│   ├── llm_chat.py          # LLM chat with history and tools
│   ├── gpt2_chat.py         # Lightweight GPT-2 responses
│   ├── admin.py             # Admin commands and utilities
│   ├── gif_chain.py         # GIF creation from reply chains
│   ├── starboard.py         # Message reposting on reactions
│   └── credits_cog.py       # User credit/usage commands
├── tasks.py                 # Celery task definitions
├── text_generators/         # LLM text generation abstraction
├── image_generators/        # Image generation API abstraction
├── tools/                   # Tool execution system
├── prompt_db.py             # SQLite database for prompts/tracking
├── cost_tracking.py         # Credit/budget management
└── settings.py              # Configuration and defaults
```

## Cogs

### General
**File**: `src/hollingsbot/cogs/general.py`

Provides fundamental bot interactions for all users.

**Commands**:
- `!ping` - Bot responsiveness test, responds with "Pong!"
- `!help` - Comprehensive help documentation covering all bot features

**Key Classes**:
- `General(commands.Cog)` - Main cog class

**Purpose**: Basic utility commands available everywhere. The help command dynamically documents all features and is auto-truncated to fit Discord's 2000-character limit.

---

### ImageGenCog
**File**: `src/hollingsbot/cogs/image_gen_cog.py`

Comprehensive image generation, editing, and outpainting with cost tracking and multiple AI providers.

**Features**:
- Image generation with configurable prefixes (`!`, `$`, etc.)
- Image editing mode (reply with edited prompt)
- Outpainting/zoom-out mode
- Seed specification with `{seed}` syntax
- Prompt expansion with `<a, b, c>` syntax
- Daily budget and per-image cost tracking
- File size compression for Discord limits
- Multi-API support (Replicate, OpenAI, etc.)

**Commands**:
- `! prompt` - Quick image via Replicate (default)
- `$ prompt` - Higher quality image
- `edit: prompt` - Edit an image (reply to image message)
- `zoom out` - Extend image boundaries
- `!models` - List available generators

**Key Classes**:
- `ImageGenCog(commands.Cog)` - Main cog (900+ lines)
- `GeneratorSpec` - Configuration for a specific generator
  - `api`: API provider name
  - `model`: Model identifier
  - `mode`: "generate", "edit", or "outpaint"
  - `price_per_image`: Cost override

**Key Data Structures**:
- `_prefix_map: dict[str, GeneratorSpec]` - Maps prefixes to generator configs (loaded from `image_gen_config.json`)
- `_cost_tracker: CostTracker` - Tracks daily budgets and per-image costs
- `_pending: set[asyncio.Task]` - Tracks active generation tasks

**Key Methods**:
- `_handle_generation()` - Main workflow for image generation
- `_split_prompt()` - Extract prompt and match prefix config
- `_expand_prompt_list()` - Handle `<a, b, c>` syntax
- `_collect_images_for_editing()` - Gather images from message/reply/history
- `_generate_outpaint_prompt()` - Use LLM to describe zoomed-out scene
- `_execute_generation_tasks()` - Run Celery tasks concurrently
- `_prepare_discord_files()` - Compress and format images for Discord
- `on_message()` - Entry point; filters and routes to generation

**Configuration Files**:
- `src/hollingsbot/image_gen_config.json` - Generator specs by prefix with optional `daily_limit`, `price_per_image`

**Environment Variables**:
- `STABLE_DIFFUSION_CHANNEL_IDS` - Comma-separated guild channel IDs where image generation is allowed
- `EDIT_CHANNEL_IDS` - Separate allowlist for `edit:` command
- `STABLE_DIFFUSION_ALLOW_DMS` - Allow image generation in DMs (default: "1")
- `DAILY_FREE_BUDGET` - Default daily free budget per user
- `PROMPT_DB_PATH` - Path to SQLite database for tracking
- `IMAGE_TIMEOUT` - Generation timeout in seconds (default: 30)

**Dependencies**:
- Celery for async task execution
- PIL for image manipulation
- cairosvg for SVG rendering (optional)
- CostTracker for budget management
- PromptDB for storage

**Architectural Notes**:
- Configuration is reloaded on-demand (mtime-based caching)
- Cost deduction only happens after successful generation
- Images are compressed aggressively to stay under Discord's 25 MiB limit
- History search limits image lookup to 10 recent messages to prevent excessive API calls
- File size limits are guild-specific, with fallback to 8 MiB

---

### LLMChatNewCog
**File**: `src/hollingsbot/cogs/llm_chat.py`

Multi-turn LLM conversation with persistent history, image support, URL metadata extraction, and tool execution.

**Features**:
- Per-channel conversation history (configurable deque with size limits)
- Image attachment support (resized, converted to JPEG, base64-encoded)
- URL metadata extraction (Open Graph, Twitter Card) with image downloading
- Text file attachment inlining (up to 120 KB per file)
- SVG code block detection and conversion to PNG (cairosvg)
- Reply context preservation (quoted messages included in history)
- Per-user model preferences (saved to state file)
- Global and per-user system prompts
- Tool execution capability
- Message history warming on first message
- Long response splitting (auto-attach text files for >2000 chars)

**Commands**:
- `!models` - List available LLM models
- `!model provider/model` - Set personal model preference
- `!system` - Show current system prompt
- `!system <text>` - Set custom system prompt
- `!system reset` - Reset to default system prompt
- `!clear` - Clear channel conversation history
- `!h` - Quick help for LLM features

**Key Classes**:
- `LLMChatNewCog(commands.Cog)` - Main cog (1300+ lines)
- `ImageAttachment` - Image metadata container
  - `name`, `url`, `data_url`, `width`, `height`, `size`
  - `to_payload()` - Convert to dict for API transmission
  - `clone()` - Create deep copy
- `ConversationTurn` - Message in history
  - `role` ("user" or "assistant")
  - `content` (text)
  - `images` (list of ImageAttachment)
  - `message_id`, `author_id`, `author_name` (metadata)
- `ModelTurn` - Message for LLM
  - `role`, `text`, `images` (same structure but minimal metadata)
- `GenerationJob` - Tracking active generation
  - `task` (asyncio.Task)
  - `result` (Celery AsyncResult)

**Key Data Structures**:
- `channel_histories: dict[int, Deque[ConversationTurn]]` - History per channel (maxlen from env)
- `model_preferences: dict[guild_id, dict[user_id, dict[provider, model]]]` - Saved state
- `_history_locks: dict[int, asyncio.Lock]` - Per-channel locks for thread safety
- `_warmed_channels: set[int]` - Tracks which channels have been history-preloaded
- `_active_generations: dict[int, GenerationJob]` - Tracks in-flight responses per channel
- `available_models: list[tuple[str, str]]` - Available (provider, model) pairs
- `system_prompt: str` - Full system prompt (base + notebook + tools)
- `base_system_prompt: str` - Editable portion

**Key Methods**:
- `on_message()` - Entry point; filters, warms, builds turn, queues generation
- `_prepare_user_turn()` - Build full message with attachments, URLs, replies
- `_collect_image_attachments()` - Download and process image attachments
- `_collect_text_attachments_full()` - Read text files (max 120 KB)
- `_extract_url_images()` - Parse URLs, extract metadata, download images
- `_build_reply_hint()` - Handle quoted messages
- `_ensure_channel_warm()` - Backfill history on first message
- `_build_conversation_payload()` - Format for LLM with recent history
- `_run_generation()` - Execute Celery task with timeout
- `_extract_and_convert_svgs()` - Detect SVG blocks and convert to PNG
- `_deliver_response()` - Format, split, and send response
- `_execute_tool_calls()` - Parse and execute tool directives in response
- `_clean_mentions()` - Replace `<@123>` with `@Username`

**File Paths**:
- `generated/llm_chat_new_state.json` - Persisted system prompt and model preferences
- `generated/notebook_state.json` - Notebook state for tool execution
- `config/system_prompt.txt` - Default system prompt (if exists)

**Environment Variables**:
- `LLM_WHITELIST_CHANNELS` - Comma-separated channel IDs where LLM chat is enabled (required; empty disables)
- `DEFAULT_LLM_PROVIDER` - Default provider (default: "openai")
- `DEFAULT_LLM_MODEL` - Default model (default: "gpt-4o")
- `AVAILABLE_MODELS` - Comma-separated list of `provider/model` options
- `TEXT_TIMEOUT` - Max seconds to wait for LLM response (default: 180)
- `LLM_HISTORY_LIMIT` - Max conversation turns per channel (default: 50)
- `LLM_MAX_TURNS_SENT` - Max recent turns to send to LLM (default: 8)
- `SYSTEM_PROMPT_FILE` - Path to custom system prompt file

**Dependencies**:
- Celery for async task execution
- PIL for image processing
- cairosvg for SVG→PNG conversion
- URL metadata extractors
- Tool execution system

**Architectural Notes**:
- History is in-memory; cleared on bot restart
- Per-channel locks ensure thread-safe history access
- Long responses are chunked at 1900 characters to fit Discord
- SVG code blocks detected with regex; converted before sending
- Images resized to max 2048px edge, converted to JPEG, base64-encoded
- URL images downloaded during message processing (with size limits)
- Only the last 8 turns sent to LLM to manage context window
- Model preferences stored per (guild_id, user_id) for multi-server support
- Tool execution results appended as separate messages to response

---

### GPT2Chat
**File**: `src/hollingsbot/cogs/gpt2_chat.py`

Lightweight, fast text generation using GPT-2 in a designated channel.

**Features**:
- Single designated channel for responses
- Optional temperature randomization (triangular distribution)
- Message queuing (newer messages override pending ones)
- Timeout handling with user-friendly error messages
- Configurable API and model

**Commands**:
- No user-facing commands; responds to all non-command messages in designated channel
- Messages starting with `!`, `-`, or `edit:` are ignored

**Key Classes**:
- `GPT2Chat(commands.Cog)` - Main cog (280 lines)

**Key Methods**:
- `_should_respond()` - Filter: not bot, not ignored keyword, in target channel
- `_celery_task()` - Async wrapper around Celery task
- `_generate()` - Generate with error handling
- `on_message()` - Entry point; checks channel, tracks newest message ID, generates

**Environment Variables**:
- `GPT2_CHANNEL_ID` - Channel ID where responses are enabled
- `GPT2_RESPONSE_TIMEOUT` - Max seconds to wait (default: 180)

**Dependencies**:
- Celery for `generate_text` task

**Architectural Notes**:
- Tracks latest message ID per channel to discard stale responses if multiple arrive
- Response truncated to 2000 characters (Discord limit)
- Temperature uses triangular distribution with mode at 1.0, range [0.5, 1.5]
- Ignores the word "enhance" by design

---

### Admin
**File**: `src/hollingsbot/cogs/admin.py`

Operational and admin-only commands for container management and credit administration.

**Features**:
- Non-destructive container restart (with fallback to bot restart)
- Credit granting/deduction (admin only)
- User balance and usage checking
- Per-prefix price configuration
- Daily free budget adjustment

**Commands**:
- `!reset` - Restart Docker containers (or bot if Docker unavailable)
- `!grant @user <amount>` - Grant/deduct credits (admin only)
- `!balance [@user]` - Check balance (own or others if admin)
- `!set_price <prefix> <price>` - Set cost per image (admin only)
- `!set_budget <amount>` - Set daily free budget (admin only)

**Key Classes**:
- `Admin(commands.Cog)` - Main cog (480 lines)

**Key Methods**:
- `_restart_containers()` - Multi-strategy restart (override command → Docker Compose → bot restart)
- `_try_override_command()` - Execute custom RESET_COMMAND if set
- `_try_docker_compose_restart()` - Try modern "docker compose" then legacy "docker-compose"
- `_execute_compose_restart()` - Subprocess execution with working directory
- `_fallback_to_bot_restart()` - Last resort: exit bot process
- `_find_compose_file()` - Search upward for docker-compose.yml
- `_run_shell_command()` / `_run_subprocess()` - Async subprocess utilities
- `_is_admin()` - Check admin user ID

**Configuration Files**:
- `src/hollingsbot/image_gen_config.json` - Modified by `!set_price` and `!set_budget`

**Environment Variables**:
- `ADMIN_USER_IDS` - Comma-separated user IDs with admin privileges
- `RESET_COMMAND` - Custom override for restart (e.g., systemctl restart hollingsbot)
- `PROMPT_DB_PATH` - Database path for cost tracker
- `DAILY_FREE_BUDGET` - Default budget (used if config not found)

**Dependencies**:
- CostTracker for user credit management
- Settings for admin ID loading

**Architectural Notes**:
- Searches for compose files from bot startup dir and repository root
- Logs all admin actions with user ID
- Graceful close attempt before force exit
- Cost tracker and config file modifications logged for audit trail

---

### GifReplyChainCog
**File**: `src/hollingsbot/cogs/gif_chain.py`

Create animated GIFs from Discord reply chains.

**Features**:
- Traverse full reply chains (ancestors and descendants)
- Collect images from all messages in chain
- Scale and center images uniformly
- Aggressive compression with multiple strategies (scaling, frame dropping, palette reduction)
- Automatic quality adjustment to fit Discord's 8 MiB limit

**Commands**:
- Reply to any message with just `gif` to build an animated GIF from the entire reply chain
- Shows 🤔 (thinking emoji) while processing, replaced with result on completion

**Key Classes**:
- `GifReplyChainCog(commands.Cog)` - Main cog (630 lines)

**Key Data Structures**:
- Compression strategies (module-level constants):
  - `SCALE_FACTORS = [1.0, 0.85, 0.72, 0.6, 0.5, 0.4, 0.33]` - Progressive downscaling
  - `FRAME_DROPS = [1, 2, 3]` - Keep every Nth frame
  - `PALETTE_SIZES = [128, 64, 32]` - Color palette reduction

**Key Methods**:
- `_validate_gif_request()` - Check: not bot, guild only, correct channel, "gif" text, is reply
- `_collect_ancestor_messages()` - Traverse backward to root via message references
- `_collect_descendant_messages()` - Traverse forward to leaf via reply lookups
- `_collect_chain_images()` - Combine ancestors/descendants, extract image attachments (limit frames)
- `_prepare_frames()` - Decode, scale, center on uniform canvases
- `_build_gif_bytes()` - Try compression combos; stop at first <8 MiB
- `_encode_gif()` - Quantize frames and save with PIL

**Environment Variables**:
- `STABLE_DIFFUSION_CHANNEL_IDS` - If set, restrict GIF to these channels; empty = all guild channels
- `GIF_CHAIN_MAX_FRAMES` - Max images to include (default: 30)
- `GIF_CHAIN_MAX_SIDE` - Max image dimension in pixels (default: 768)
- `GIF_CHAIN_FRAME_MS` - Milliseconds per frame (default: 700)
- `GIF_CHAIN_MAX_BYTES` - Max GIF file size (default: 8 MiB)
- `GIF_CHAIN_SCAN_LIMIT` - Max messages to scan for replies (default: 400)

**Dependencies**:
- PIL for image processing

**Architectural Notes**:
- Handles broken references gracefully; skips inaccessible messages
- Scanning limit prevents runaway history fetches
- Canvases use black background (RGBA: 0,0,0,255)
- All frames converted to RGB before GIF encoding
- Attempts 21+ compression combinations (7 scales × 3 drops × palette options)
- Last-resort fallback: single frame with maximum compression

---

### Starboard
**File**: `src/hollingsbot/cogs/starboard.py`

Mirror reacted bot messages to a dedicated starboard channel.

**Features**:
- Listen for reactions on bot messages
- Repost eligible messages to starboard with metadata
- Channel whitelisting/blacklisting
- Duplicate prevention via message cache (2048 recent message IDs)
- Attachment and embed image preservation
- Database logging of starboard posts

**Commands**:
- No user-facing commands; automatic on reaction to bot message

**Key Classes**:
- `Starboard(commands.Cog)` - Main cog (450 lines)

**Key Data Structures**:
- `_seen_message_ids: deque[int]` - Cache of recently posted messages (maxlen=2048)

**Key Methods**:
- `on_raw_reaction_add()` - Main handler; multi-stage validation
- `_is_valid_reactor()` - Check: not bot, not self-reaction
- `_is_channel_eligible()` - Check: not ignored, whitelisted if applicable
- `_is_message_eligible()` - Check: from bot, not already posted
- `_send_to_starboard()` - Format message with jump URL, preview, attachments
- `_format_forward_content()` - Create compact preview (max 1900 chars)
- `_extract_message_attachments()` - Collect file attachments and embed images
- `_log_starboard_entry()` - Write to database with full metadata

**Environment Variables**:
- `STARBOARD_CHANNEL_ID` - Channel ID for starboard (required; cog idles if not set)
- `STARBOARD_IGNORE_CHANNELS` - Comma-separated channel IDs to exclude
- `STARBOARD_WHITELIST_CHANNEL_IDS` - If set, restrict to these channels

**Dependencies**:
- PromptDB for `log_starboard_post()`

**Architectural Notes**:
- Only reposts bot messages (filters `message.author.bot`)
- Maintains in-memory cache of 2048 message IDs to prevent duplicates
- Formats forward content with author name, preview text, and all attachment/embed URLs
- Stores attachments as JSON in database for archival
- Gracefully handles missing channels, deleted messages, and permission errors

---

### CreditsCog
**File**: `src/hollingsbot/cogs/credits_cog.py`

User-facing commands for checking image generation usage and credit balance.

**Features**:
- Check personal usage and balance
- View today's free budget usage
- View credits spent and remaining balance
- View generation count
- Display time until daily reset

**Commands**:
- `!usage` - Display current usage and balance

**Key Classes**:
- `CreditsCog(commands.Cog)` - Main cog (80 lines)

**Key Methods**:
- `usage_command()` - Fetch status and format response

**Environment Variables**:
- `PROMPT_DB_PATH` - Database path for cost tracker
- `DAILY_FREE_BUDGET` - Daily free budget amount

**Dependencies**:
- CostTracker for user status queries

**Architectural Notes**:
- Simple wrapper around CostTracker status queries
- Displays reset time relative to now (in human-readable format)
- Grouped display: free budget → credits spent → generation count → balance

---

## Configuration

### Environment Variables Reference

**Discord**:
- `DISCORD_TOKEN` (required) - Bot authentication token

**LLM**:
- `DEFAULT_LLM_PROVIDER` (default: "openai") - Default text generation provider
- `DEFAULT_LLM_MODEL` (default: "gpt-4o") - Default model
- `AVAILABLE_MODELS` - Comma-separated `provider/model` options
- `TEXT_TIMEOUT` (default: 180) - LLM generation timeout in seconds
- `LLM_WHITELIST_CHANNELS` - Required; empty disables LLM chat
- `LLM_HISTORY_LIMIT` (default: 50) - Conversation turns per channel
- `LLM_MAX_TURNS_SENT` (default: 8) - Recent turns to send to LLM
- `SYSTEM_PROMPT_FILE` - Path to custom system prompt

**Image Generation**:
- `STABLE_DIFFUSION_CHANNEL_IDS` - Guild channels allowing image gen
- `EDIT_CHANNEL_IDS` - Separate allowlist for `edit:` command
- `STABLE_DIFFUSION_ALLOW_DMS` (default: "1") - Allow DM image gen
- `IMAGE_TIMEOUT` (default: 30) - Image generation timeout
- `IMAGE_OUTPUT_DIR` (default: "generated/") - Output directory
- `DAILY_FREE_BUDGET` (default: 0.50) - Free budget per user per day
- `PROMPT_DB_PATH` (default: "prompts.db") - Database path

**GIF Chain**:
- `GIF_CHAIN_MAX_FRAMES` (default: 30) - Max frames in GIF
- `GIF_CHAIN_MAX_SIDE` (default: 768) - Max image dimension
- `GIF_CHAIN_FRAME_MS` (default: 700) - Milliseconds per frame
- `GIF_CHAIN_MAX_BYTES` (default: 8MB) - Max GIF file size
- `GIF_CHAIN_SCAN_LIMIT` (default: 400) - Max messages to scan

**Starboard**:
- `STARBOARD_CHANNEL_ID` - Channel to repost to (required if enabled)
- `STARBOARD_IGNORE_CHANNELS` - Channels to exclude
- `STARBOARD_WHITELIST_CHANNEL_IDS` - If set, restrict to these

**GPT-2**:
- `GPT2_CHANNEL_ID` - Designated channel for GPT-2 responses
- `GPT2_RESPONSE_TIMEOUT` (default: 180) - Response timeout

**Admin**:
- `ADMIN_USER_IDS` - Comma-separated admin user IDs
- `RESET_COMMAND` - Custom container restart command

**Celery / Redis**:
- `CELERY_BROKER_URL` (default: "redis://redis:6379/0") - Celery broker
- `CELERY_RESULT_BACKEND` (default: "redis://redis:6379/0") - Result storage

### Configuration Files

**image_gen_config.json**:
```json
{
  "daily_free_budget": 0.50,
  "default_price_per_image": 0.03,
  "!": {
    "api": "replicate",
    "model": "black-forest-labs/flux-schnell",
    "price_per_image": 0.03
  },
  "$": {
    "api": "replicate",
    "model": "google/imagen-4-fast",
    "daily_limit": 200
  },
  "edit:": {
    "api": "replicate",
    "model": "google/nano-banana",
    "mode": "edit"
  }
}
```

**State Files**:
- `generated/llm_chat_new_state.json` - System prompt and model preferences (auto-created)
- `generated/notebook_state.json` - Tool/notebook state (auto-created)

## Architecture

### Service Architecture

The bot runs as three Docker containers orchestrated by docker-compose:

- **bot**: Main Discord bot (python 3.10+, watchfiles auto-reload)
- **celery_text**: Text generation worker (concurrency: 1, no GPU)
- **celery_image**: Image generation worker (concurrency: 2, requires NVIDIA GPU)
- **redis**: Message broker for Celery task queue

All services share the codebase mounted at `/app` with volumes for persistence:
- `prompt_db_data` - SQLite database volume
- `generated/` - Generated images and state files

### Task Queue System

Celery tasks are routed by queue:

- `generate_image` → `image` queue (GPU-accelerated, concurrency: 2)
- `generate_text` → `text` queue (CPU, concurrency: 1)
- `generate_llm_chat_response` → `text` queue (CPU, concurrency: 1)

### Generator Pattern

Both image and text generation use factory functions that return provider-agnostic interfaces:

**Image Generation**:
- `image_generators/__init__.py::get_image_generator(api, model)` returns `ImageGeneratorAPI` instance
- Supported APIs: `replicate`, `svg`/`openai-svg`

**Text Generation**:
- `text_generators/__init__.py::get_text_generator(api, model)` returns `TextGeneratorAPI` instance
- Supported APIs: `huggingface`, `anthropic`, `openai`/`chatgpt`

### Database

**SQLite** (`prompts.db`):
- Stores image generation tracking (user ID, prompt, model, timestamp)
- Stores PR notifications
- Stores starboard posts with full metadata
- Stores cost tracking (credits, daily usage, limits)

### Key Dependencies

- **discord.py**: Discord API wrapper
- **Celery + Redis**: Async task execution
- **Pillow (PIL)**: Image processing
- **cairosvg**: SVG to PNG conversion (optional)
- **anthropic/openai**: LLM APIs
- **requests**: HTTP client for metadata extraction

## Important Notes

- **GPU Requirement**: The `celery_image` worker requires an NVIDIA GPU with the NVIDIA Container Toolkit installed on the host
- **Privacy Mode**: Set `STABLE_DIFFUSION_PRIVACY=1` to disable logging of message content
- **Conversation History**: LLM chat history is in-memory only and cleared on bot restart
- **Bot Restart Interval**: The bot auto-restarts every 6 hours by default (`BOT_RESTART_INTERVAL` environment variable in seconds)
- **Model Availability**: Available LLM models are loaded from the `AVAILABLE_MODELS` environment variable at startup
