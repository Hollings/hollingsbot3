# Hollingsbot3

A feature-rich Discord bot built with discord.py, featuring LLM chat, image generation, and various utilities. The bot runs in Docker with Celery workers for asynchronous task processing.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Requirements & Setup](#requirements--setup)
4. [Development Workflow](#development-workflow)
5. [Cogs Documentation](#cogs-documentation)
6. [Configuration](#configuration)
7. [Architecture Overview](#architecture-overview)

---

## Project Overview

Hollingsbot3 is a modular Discord bot built with discord.py's Cog system. It uses a distributed task queue (Celery with Redis broker) to handle CPU-intensive operations like image generation and LLM inference asynchronously. The bot is containerized with Docker and orchestrated using docker-compose.

### Service Architecture

The bot runs as four separate Docker containers:

- **bot**: Main Discord bot (no GPU, uses watchfiles for auto-reload)
- **celery_text**: Text generation worker (queue: `text`, concurrency: 1, no GPU)
- **celery_image**: Image generation worker (queue: `image`, concurrency: 2, requires NVIDIA GPU)
- **redis**: Message broker for Celery tasks

All services share the same codebase mounted at `/app`, with state persisted in Docker volumes (`prompt_db_data` for SQLite database).

---

## Key Features

### Image Generation
- Multiple image generation APIs (Replicate, SVG-based generators)
- Prefix-based routing (configurable via `image_gen_config.json`)
- Support for image editing and outpainting
- Daily usage limits and cost tracking per user
- Automatic image compression for Discord limits

### LLM Chat
- Per-channel conversation history with configurable limits
- Support for text and image attachments
- URL metadata extraction (Open Graph, Twitter Card)
- SVG rendering to PNG
- Tool execution capabilities
- Per-user model preferences
- Customizable system prompts

### Utilities
- GIF creation from reply chains
- Starboard (optional) for mirroring reacted messages
- GPT-2 lightweight chat in designated channel
- Bot admin commands (restart, credit management)
- User credit and usage tracking

---

## Requirements & Setup

### Prerequisites

- **Docker** and **docker-compose** (or Docker Desktop)
- **NVIDIA GPU** (for image generation worker; optional if only using text features)
- **NVIDIA Container Toolkit** (required for GPU support in Docker)
- Python 3.9+ (for local development/testing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd hollingsbot3
   ```

2. **Create `.env` file** with required credentials and configuration:
   ```bash
   DISCORD_TOKEN=your_bot_token
   CELERY_BROKER_URL=redis://redis:6379/0
   CELERY_RESULT_BACKEND=redis://redis:6379/0
   DEFAULT_LLM_PROVIDER=openai
   DEFAULT_LLM_MODEL=gpt-4o
   ```

3. **Start the bot:**
   ```bash
   docker-compose up -d
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f bot
   docker-compose logs -f celery_text
   docker-compose logs -f celery_image
   ```

### Local Development (without Docker)

For testing without Docker, install dependencies and run tests:

```bash
pip install -r requirements.txt
pytest
```

---

## Development Workflow

### Making Code Changes

1. **Edit Python files** in the `src/hollingsbot/` directory
2. **Restart containers** to apply changes:
   ```bash
   docker-compose down && docker-compose up -d
   ```

The bot and workers use `watchfiles` to monitor Python files for changes, but containers must be restarted for code changes to take effect.

### Building from Scratch

```bash
docker-compose up --build
```

### Running Tests

```bash
pytest
```

### State Persistence

- **SQLite database** (`prompts.db` in `/data` volume): Stores image generation tracking, PR notifications, starboard posts
- **State files** in `generated/`: LLM chat state (`llm_chat_new_state.json` with system prompts and model preferences)
- **Generated images**: Written to `generated/` directory, mounted as volume

---

## Cogs Documentation

Cogs are modular components loaded by the bot, implementing specific features. All cogs are registered in `src/hollingsbot/__main__.py`.

### General (general.py)

**Purpose**: Basic commands available to all users.

**Key Classes**:
- `General(commands.Cog)`: Main class

**Commands**:
- `!ping` - Test bot responsiveness (responds "Pong!")
- `!help` - Display comprehensive bot help documentation with command overview

**Implementation Details**:
- Help text is built dynamically and truncated to Discord's 2000-character limit
- Organized help message covers image generation, GIF creation, LLM chat, GPT-2, admin commands

**File Path**: `src/hollingsbot/cogs/general.py`

---

### Admin (admin.py)

**Purpose**: Operational commands including container restart and credit management.

**Key Classes**:
- `Admin(commands.Cog)`: Main class with Docker/container restart logic

**Admin-Only Commands**:
- `!grant @user amount` - Grant/deduct credits to a user
- `!balance [@user]` - Check user's balance and daily usage (check own or others' if admin)
- `!set_price prefix price` - Set price per image for a model prefix
- `!set_budget budget` - Set global daily free budget

**Public Commands**:
- `!reset` - Restart Docker containers (or bot process if Docker unavailable)

**Technical Implementation**:
- **Container restart strategy**: Tries custom override command → modern `docker compose` → legacy `docker-compose` → bot process restart fallback
- **Compose file discovery**: Searches current directory and repository root for `docker-compose.{yml,yaml}` or `compose.{yml,yaml}`
- **Cost tracking integration**: Uses `CostTracker` class for credit management (stored in SQLite)
- **Admin users**: Loaded from `ADMIN_USER_IDS` environment variable

**Key Methods**:
- `_restart_containers()`: Handles container restart with multiple fallback strategies
- `_try_docker_compose_restart()`: Attempts Docker Compose restart
- `_find_compose_file()`: Locates docker-compose file in project tree
- `_cost_tracker`: CostTracker instance for credit operations

**Dependencies**:
- `CostTracker` (from `hollingsbot.cost_tracking`)
- Docker/docker-compose binaries (for restart)

**File Path**: `src/hollingsbot/cogs/admin.py`

---

### ImageGenCog (image_gen_cog.py)

**Purpose**: Handle image generation, editing, and outpainting through Celery tasks.

**Key Classes**:
- `ImageGenCog(commands.Cog)`: Main class handling image generation workflow
- `GeneratorSpec`: Dataclass specifying API, model, mode, and price for a generator

**Features**:
- Multi-API support (Replicate, SVG generators) via configurable prefixes
- Image editing and outpainting modes
- Seed support via `{seed}` syntax
- Prompt list expansion via `<a, b, c>` syntax
- Daily usage limits and cost tracking per user
- Automatic image compression and Discord file size limit handling
- LLM-generated prompts for outpainting

**Environment Variables**:
- `STABLE_DIFFUSION_CHANNEL_IDS`: Comma-separated channel IDs where image generation is allowed
- `EDIT_CHANNEL_IDS`: Separate allowlist for `edit:` and `zoom out` commands
- `STABLE_DIFFUSION_ALLOW_DMS`: Enable/disable DM support (default: 1)
- `IMAGE_TIMEOUT`: Generation timeout in seconds (default: 30s)
- `PROMPT_DB_PATH`: Path to SQLite database

**Key Data Structures**:
- `_prefix_map: dict[str, GeneratorSpec]`: Maps command prefix → generator configuration
- `_allowed_channel_ids: set[int]`: Guild channels where image generation is allowed
- `_edit_channel_ids: set[int]`: Channels where edit/outpaint commands work
- `_pending: set[asyncio.Task[None]]`: Pending generation tasks for cleanup

**Configuration**:
Loaded from `image_gen_config.json` (path: `src/hollingsbot/image_gen_config.json`):
```json
{
  "daily_free_budget": 0.50,
  "default_price_per_image": 0.03,
  "!": {"api": "replicate", "model": "black-forest-labs/flux-schnell", "mode": "generate"},
  "$": {"api": "replicate", "model": "google/imagen-4-fast", "mode": "generate", "daily_limit": 200},
  "edit:": {"api": "replicate", "model": "google/nano-banana", "mode": "edit"},
  "zoom out": {"api": "replicate", "model": "model-xyz", "mode": "outpaint"}
}
```

**Key Methods**:
- `on_message()`: Entry point for image generation requests
- `_handle_generation()`: Main generation workflow
- `_split_prompt()`: Parse prefix and clean prompt from message content
- `_execute_generation_tasks()`: Launch multiple generation tasks concurrently
- `_collect_images_for_editing()`: Gather images from message, reply, or history for edit mode
- `_prepare_outpaint()`: Prepare images and generate LLM prompt for outpaint mode
- `_expand_prompt_list()`: Expand `<a, b, c>` syntax into multiple prompts
- `_parse_seed_from_prompt()`: Extract `{seed}` from prompt
- `_prepare_discord_files()`: Create Discord File objects with compression
- `_process_and_send_results()`: Send generated images to Discord
- `_default_celery_runner()`: Execute Celery task and poll for results

**Utilities**:
- `detect_mime_type()`: Detect image MIME type from magic bytes
- `detect_image_extension()`: Detect image extension from magic bytes
- `bytes_to_data_url()`: Convert image bytes to data URL for JSON serialization

**Celery Task**:
- Task: `generate_image` (queue: `image`)
- Parameters: prompt_id, api, model, prompt, seed, image_input, mask, output_format

**Database**:
- Stores prompts, generation tracking, and daily limits in SQLite
- Uses `bulk_add_prompts()` to record generation requests

**File Path**: `src/hollingsbot/cogs/image_gen_cog.py`

---

### LLMChatNewCog (llm_chat.py)

**Purpose**: LLM-powered chat with conversation history, image support, URL metadata, and tool execution.

**Key Classes**:
- `LLMChatNewCog(commands.Cog)`: Main chat handler
- `ConversationTurn`: Dataclass representing a single turn (user/assistant)
- `ModelTurn`: Dataclass for LLM-specific turn data
- `ImageAttachment`: Dataclass for managing image attachments with metadata
- `GenerationJob`: Tracks active generation task and Celery result

**Features**:
- Per-channel conversation history with configurable depth
- Text and image attachment support (with size limits)
- URL metadata extraction and image downloading
- SVG rendering to PNG (via cairosvg)
- Tool execution (custom tools framework)
- Per-user and per-guild model preferences
- Customizable system prompts (stored in state file)
- Reply context preservation
- Message history warming on first channel message

**Environment Variables**:
- `LLM_WHITELIST_CHANNELS`: Comma-separated channel IDs where chat is enabled (required)
- `DEFAULT_LLM_PROVIDER`: Default provider (openai/anthropic, default: openai)
- `DEFAULT_LLM_MODEL`: Default model (default: gpt-4o)
- `AVAILABLE_MODELS`: Comma-separated `provider/model` list for model selection
- `LLM_HISTORY_LIMIT`: Max conversation history per channel (default: 50)
- `LLM_MAX_TURNS_SENT`: Max turns sent to LLM per request (default: 8)
- `TEXT_TIMEOUT`: Generation timeout in seconds (default: 180)
- `SYSTEM_PROMPT_FILE`: Path to custom system prompt file

**Key Data Structures**:
- `channel_histories: dict[int, Deque[ConversationTurn]]`: Per-channel conversation history
- `_history_locks: dict[int, asyncio.Lock]`: Thread-safe history access
- `_warmed_channels: set[int]`: Channels whose history has been preloaded
- `_active_generations: dict[int, GenerationJob]`: Track active LLM generations per channel
- `model_preferences: dict[str, dict[str, dict[str, str]]]`: User model preferences per guild
- `ImageAttachment`: Stores name, url, data_url, width, height, size

**System Prompt**:
- Loaded from `SYSTEM_PROMPT_FILE` (default: `config/system_prompt.txt`)
- Can be overridden per-user/globally via `!system` command
- Built from: base prompt + notebook contents + tool definitions
- Persisted in `generated/llm_chat_new_state.json`

**Attachment Processing**:
- **Text attachments**: Inlined into message for current LLM call, replaced with placeholders in history (max 120KB)
- **Image attachments**: Downloaded, resized to max 2048px, converted to JPEG, base64-encoded as data URLs
- **SVG code blocks**: Detected and converted to PNG via cairosvg
- **URL images**: Extracted from Open Graph/Twitter Card metadata

**Reply Context**:
- When replying to a message, includes replied-to message content and images
- Hints user with `(Replying to <user>: ...)` prefix

**Message Chunking**:
- Responses longer than 2000 chars are uploaded as text files
- Code blocks may be attached if they exceed message limits

**Key Methods**:
- `on_message()`: Entry point for incoming messages
- `_ensure_channel_warm()`: Preload channel history on first message
- `_prepare_user_turn()`: Build turn with full LLM content and lighter history version
- `_build_conversation_payload()`: Construct conversation array for LLM with recent history
- `_generate_and_send_response()`: Main response generation and delivery
- `_run_generation()`: Execute Celery task and poll for results
- `_extract_and_convert_svgs()`: Extract SVG blocks and convert to PNG
- `_deliver_response()`: Send response to channel, handling long messages and SVG files
- `_execute_tool_calls()`: Execute tool calls embedded in response
- `_clean_mentions()`: Replace user mentions with display names
- `_extract_url_images()`: Extract images from URL metadata
- Tool execution: `parse_tool_calls()`, `execute_tool_call()`

**Commands**:
- `!models` - List available LLM models
- `!model [provider/model]` - View or set preferred model (per-guild)
- `!system` - Show current system prompt
- `!system <text>` - Set custom system prompt
- `!system reset` - Reset to default system prompt
- `!clear` - Clear channel conversation history
- `!h` - Show help for LLM chat features

**Celery Task**:
- Task: `generate_llm_chat_response` (queue: `text`)
- Parameters: provider, model, conversation, temperature

**Database**:
- State file: `generated/llm_chat_new_state.json` (JSON)
- Stores system prompt and per-user model preferences

**File Path**: `src/hollingsbot/cogs/llm_chat.py`

---

### GPT2Chat (gpt2_chat.py)

**Purpose**: Lightweight GPT-2 based chat for designated channel(s).

**Key Classes**:
- `GPT2Chat(commands.Cog)`: Main class

**Features**:
- Responds to all messages in designated channel
- Lightweight text generation (GPT-2 or configurable model)
- Random temperature for varied responses
- Ignores "enhance" keyword
- Stale response filtering (only newest message per channel is answered)

**Environment Variables**:
- `GPT2_CHANNEL_ID`: Channel ID where GPT-2 responses are enabled
- `GPT2_RESPONSE_TIMEOUT`: Timeout in seconds (default: 180)

**Configuration**:
- Constructor parameters: channel_id, api, model, task_func, timeout
- Default API: "huggingface"
- Default model: "gpt2-medium"

**Key Methods**:
- `on_message()`: Listener for incoming messages
- `_should_respond()`: Validate message (not bot, not "enhance", in target channel)
- `_generate()`: Run text generation with error handling
- `_celery_task()`: Execute text generation via Celery task queue
- `_generate_temperature()`: Random temperature with triangular distribution (0.5-1.5, mode 1.0)

**Behavior**:
- Tracks latest message per channel (`_latest`)
- Only responds to most recent prompt if multiple arrive quickly
- Allows old generation tasks to finish but discards results
- Responses truncated to 2000 characters (Discord limit)

**Celery Task**:
- Task: `generate_text` (queue: `text`)
- Parameters: api, model, prompt, temperature

**File Path**: `src/hollingsbot/cogs/gpt2_chat.py`

---

### GifReplyChainCog (gif_chain.py)

**Purpose**: Create animated GIFs from Discord reply chains.

**Key Classes**:
- `GifReplyChainCog(commands.Cog)`: Main class

**Features**:
- Reply with "gif" to collect and animate all images in a reply chain
- Traverses both ancestors (replies to) and descendants (replies to this message)
- Progressive compression strategies (scaling, frame dropping, palette reduction)
- Automatic frame size limiting (max 30 frames by default)
- Supports multiple image formats (PNG, JPG, WebP, BMP, GIF, TIFF)

**Environment Variables**:
- `STABLE_DIFFUSION_CHANNEL_IDS`: Allowed channels (default: all channels)
- `GIF_CHAIN_MAX_FRAMES`: Maximum frames in GIF (default: 30)
- `GIF_CHAIN_MAX_SIDE`: Maximum image dimension in pixels (default: 768)
- `GIF_CHAIN_FRAME_MS`: Frame duration in milliseconds (default: 700)
- `GIF_CHAIN_MAX_BYTES`: Maximum GIF file size (default: 8 MiB)
- `GIF_CHAIN_SCAN_LIMIT`: Max messages to scan for reply chain (default: 400)

**Compression Strategy**:
Tries multiple combinations in order until GIF fits size limit:
- Scale factors: 1.0 → 0.85 → 0.72 → 0.6 → 0.5 → 0.4 → 0.33
- Frame drops: Keep every 1st, 2nd, or 3rd frame
- Palette sizes: 128 → 64 → 32 colors

**Key Methods**:
- `on_message()`: Listener for "gif" replies
- `_validate_gif_request()`: Check if message is valid GIF request
- `_collect_chain_images()`: Traverse reply chain and collect image attachments
- `_collect_ancestor_messages()`: Traverse backward to root message
- `_collect_descendant_messages()`: Traverse forward to leaf message
- `_prepare_frames()`: Decode, scale, and canvas images
- `_create_uniform_canvas()`: Center images on uniform-sized black canvases
- `_build_gif_bytes()`: Encode GIF with progressive compression
- `_encode_gif()`: Quantize frames and create animated GIF

**Image Processing**:
- Decodes images to PIL Image objects in RGBA
- Scales images to max dimension (default: 768px)
- Centers all images on uniform canvas (black background)
- Converts to RGB for GIF encoding

**Reaction**:
- Shows thinking emoji while processing
- Removed on completion or error

**File Path**: `src/hollingsbot/cogs/gif_chain.py`

---

### Starboard (starboard.py)

**Purpose**: Mirror reacted bot messages to a dedicated "starboard" channel (optional feature).

**Key Classes**:
- `Starboard(commands.Cog)`: Main class

**Features**:
- Listens for reactions on bot messages
- Reposts eligible messages to designated starboard channel
- Channel whitelisting/blacklisting
- Deduplication via seen message cache (2048 messages)
- Logs starboard entries to SQLite database

**Environment Variables**:
- `STARBOARD_CHANNEL_ID`: Target channel ID (required to enable)
- `STARBOARD_IGNORE_CHANNELS`: Comma-separated channel IDs to exclude
- `STARBOARD_WHITELIST_CHANNEL_IDS`: If set, only these channels are eligible

**Configuration**:
- Starboard only works if `STARBOARD_CHANNEL_ID` is set
- Supports channel allowlisting and blocklisting

**Key Methods**:
- `on_raw_reaction_add()`: Listener for reactions on messages
- `_is_valid_reactor()`: Check if reactor is a non-bot human
- `_is_channel_eligible()`: Check if origin channel is allowed
- `_is_message_eligible()`: Check if message is from bot and not seen before
- `_send_to_starboard()`: Format and send message to starboard
- `_format_forward_content()`: Format message with jump URL, preview, and attachments
- `_log_starboard_entry()`: Record post metadata in database
- `_extract_message_attachments()`: Get attachment URLs and metadata
- `_serialize_attachments()`: Serialize attachments to JSON for database

**Message Format**:
- Jump URL (clickable link back to original)
- Author name and content preview (truncated to 800 chars)
- Attachment URLs

**Database**:
- Logs to SQLite table with: guild_id, channel_id, message_ids, reactor info, content preview, attachments

**File Path**: `src/hollingsbot/cogs/starboard.py`

---

### CreditsCog (credits_cog.py)

**Purpose**: User-facing commands for checking image generation usage and credits.

**Key Classes**:
- `CreditsCog(commands.Cog)`: Main class

**Commands**:
- `!usage` - Show user's current usage and credit balance

**Information Displayed**:
- Free budget used today / total free budget
- Credits spent today (if any)
- Total generations today
- Current credit balance
- Time until daily reset

**Integration**:
- Uses `CostTracker` for querying usage data
- Reads configuration from environment variables

**File Path**: `src/hollingsbot/cogs/credits_cog.py`

---

## Configuration

### Environment Variables

#### Core Discord & Celery

| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_TOKEN` | (required) | Discord bot token |
| `CELERY_BROKER_URL` | (required) | Redis broker URL (e.g., `redis://redis:6379/0`) |
| `CELERY_RESULT_BACKEND` | (required) | Redis result backend URL |

#### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM_PROVIDER` | openai | Default LLM provider (openai/anthropic/etc) |
| `DEFAULT_LLM_MODEL` | gpt-4o | Default LLM model |
| `AVAILABLE_MODELS` | (empty) | Comma-separated list of `provider/model` pairs |
| `TEXT_TIMEOUT` | 180 | LLM generation timeout in seconds |
| `LLM_WHITELIST_CHANNELS` | (empty) | Comma-separated channel IDs for LLM chat |
| `LLM_HISTORY_LIMIT` | 50 | Max conversation history entries per channel |
| `LLM_MAX_TURNS_SENT` | 8 | Max turns sent to LLM per request |
| `SYSTEM_PROMPT_FILE` | config/system_prompt.txt | Path to custom system prompt |

#### Image Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `STABLE_DIFFUSION_CHANNEL_IDS` | (empty) | Comma-separated channel IDs for image generation |
| `STABLE_DIFFUSION_ALLOW_DMS` | 1 | Allow image generation in DMs (1/0) |
| `EDIT_CHANNEL_IDS` | (empty) | Channel IDs for `edit:` and `zoom out` commands |
| `IMAGE_TIMEOUT` | 30 | Image generation timeout in seconds |
| `PROMPT_DB_PATH` | prompts.db | Path to SQLite database |

#### Cost Tracking & Budgets

| Variable | Default | Description |
|----------|---------|-------------|
| `DAILY_FREE_BUDGET` | 0.50 | Daily free budget per user in dollars |

#### GIF Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `GIF_CHAIN_MAX_FRAMES` | 30 | Max frames in generated GIF |
| `GIF_CHAIN_MAX_SIDE` | 768 | Max dimension for images in pixels |
| `GIF_CHAIN_FRAME_MS` | 700 | Duration per frame in milliseconds |
| `GIF_CHAIN_MAX_BYTES` | 8388608 | Max GIF file size in bytes (8 MiB) |
| `GIF_CHAIN_SCAN_LIMIT` | 400 | Max messages to scan for reply chains |

#### GPT-2 Chat

| Variable | Default | Description |
|----------|---------|-------------|
| `GPT2_CHANNEL_ID` | (empty) | Channel ID for GPT-2 responses |
| `GPT2_RESPONSE_TIMEOUT` | 180 | GPT-2 generation timeout in seconds |

#### Starboard

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_STARBOARD` | (empty) | Enable starboard feature (any value enables) |
| `STARBOARD_CHANNEL_ID` | (empty) | Target channel for starboard posts |
| `STARBOARD_IGNORE_CHANNELS` | (empty) | Comma-separated channels to exclude |
| `STARBOARD_WHITELIST_CHANNEL_IDS` | (empty) | Comma-separated whitelist (if set, only these) |

#### Admin

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_USER_IDS` | (empty) | Comma-separated Discord user IDs with admin access |
| `RESET_COMMAND` | (empty) | Custom restart command override |

### Configuration Files

#### image_gen_config.json

Located at: `src/hollingsbot/image_gen_config.json`

Defines image generation prefixes, models, and pricing:

```json
{
  "daily_free_budget": 0.50,
  "default_price_per_image": 0.03,
  "!": {
    "api": "replicate",
    "model": "black-forest-labs/flux-schnell",
    "mode": "generate",
    "price_per_image": 0.005
  },
  "$": {
    "api": "replicate",
    "model": "google/imagen-4-fast",
    "mode": "generate",
    "price_per_image": 0.030,
    "daily_limit": 200
  },
  "edit:": {
    "api": "replicate",
    "model": "google/nano-banana",
    "mode": "edit",
    "price_per_image": 0.015
  }
}
```

#### System Prompt

Default path: `config/system_prompt.txt`

Customizable via environment variable `SYSTEM_PROMPT_FILE`. Can be overridden per-user via `!system` command. The full system prompt is built from:

1. Base system prompt (from file or override)
2. Notebook contents (if available)
3. Tool definitions (if available)

---

## Architecture Overview

### Cog System

All features are implemented as discord.py Cogs (modular components). Cogs are automatically loaded in `src/hollingsbot/__main__.py`:

```python
await bot.load_extension("hollingsbot.cogs.general")
await bot.load_extension("hollingsbot.cogs.admin")
await bot.load_extension("hollingsbot.cogs.image_gen_cog")
await bot.load_extension("hollingsbot.cogs.llm_chat")
await bot.load_extension("hollingsbot.cogs.gpt2_chat")
await bot.load_extension("hollingsbot.cogs.gif_chain")
await bot.load_extension("hollingsbot.cogs.starboard")
await bot.load_extension("hollingsbot.cogs.credits_cog")
```

### Task Queue (Celery)

CPU-intensive operations are delegated to Celery workers via Redis:

**Text Generation**: `generate_text` task (queue: `text`)
- Handled by `celery_text` worker (concurrency: 1, no GPU)
- Used by: GPT-2 Chat, LLM Chat, Image Gen (for outpaint prompts)

**Image Generation**: `generate_image` task (queue: `image`)
- Handled by `celery_image` worker (concurrency: 2, requires GPU)
- Used by: Image Gen Cog

**LLM Response**: `generate_llm_chat_response` task (queue: `text`)
- Handled by `celery_text` worker
- Used by: LLM Chat Cog

### Database

**SQLite** (`prompts.db`):
- Stores image generation history and tracking
- Stores user cost/credit information
- Stores starboard posts (if enabled)
- Daily limits and usage tracking

**State Files** (JSON):
- `generated/llm_chat_new_state.json`: LLM chat state (system prompts, model preferences)
- `generated/notebook_state.json`: Tool/notebook state (if available)

### Generator Pattern

Both image and text generation use factory pattern for extensibility:

**Image Generators** (`image_generators/__init__.py`):
```python
def get_image_generator(api: str, model: str) -> ImageGeneratorAPI:
    # Returns appropriate generator instance
    # Supports: replicate, svg, openai-svg
```

**Text Generators** (`text_generators/__init__.py`):
```python
def get_text_generator(api: str, model: str) -> TextGeneratorAPI:
    # Returns appropriate generator instance
    # Supports: huggingface, anthropic, openai, etc.
```

### Command Prefix System

The bot uses dynamic prefixes to prevent command conflicts:

- **Mention prefix**: Always enabled (e.g., `@bot help`)
- **Bang prefix (`!`)**: Disabled in image-gen channels to allow prompts like `! a cat`
- **Image generation prefixes**: Defined in `image_gen_config.json` (e.g., `!`, `$`, `edit:`)

Image generation and other features listen directly on `on_message` and check for their prefixes.

---

## Key Dependencies

- **discord.py**: Discord API wrapper
- **celery**: Distributed task queue
- **redis**: Message broker
- **PIL/Pillow**: Image processing
- **cairosvg**: SVG to PNG conversion
- **openai/anthropic**: LLM APIs (as configured)

---

## Troubleshooting

### Image generation not working
- Ensure `NVIDIA_VISIBLE_DEVICES=all` is set for Docker if using GPU
- Check `celery_image` logs: `docker-compose logs celery_image`
- Verify `STABLE_DIFFUSION_CHANNEL_IDS` includes the target channel

### LLM chat not responding
- Check whitelist: `LLM_WHITELIST_CHANNELS` must include the channel
- Verify API credentials and models are configured
- Check `celery_text` logs for generation errors
- Ensure message isn't filtered (doesn't start with `!`, `-`, or `edit:`)

### GIF creation fails
- Check that images are actually in the reply chain
- Verify `GIF_CHAIN_MAX_BYTES` isn't too restrictive
- Look for PIL/image decoding errors in bot logs

### Stale responses in GPT-2 channel
- This is expected behavior; only newest message per channel is answered
- Multiple quick messages may result in only the latest being responded to

---

For more details on specific features or development, see the individual cog documentation above.
