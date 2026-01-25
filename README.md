# Hollingsbot3

A feature-rich Discord bot built with [discord.py](https://github.com/Rapptz/discord.py), powered by LLMs and image generation APIs. The bot runs in Docker with asynchronous task processing via Celery, enabling GPU-accelerated image generation, multi-model LLM chat with conversation history, and various utility features.

## Table of Contents

1. [Key Features](#key-features)
2. [Requirements & Setup](#requirements--setup)
3. [Development Workflow](#development-workflow)
4. [Service Architecture](#service-architecture)
5. [Cogs Documentation](#cogs-documentation)
6. [Configuration](#configuration)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

## Developer Documentation

For in-depth technical documentation, see the [docs/](docs/) directory:

- **[Architecture](docs/ARCHITECTURE.md)** - System design, component interactions, data flow
- **[Configuration](docs/CONFIGURATION.md)** - Complete environment variable reference
- **[Integrations](docs/INTEGRATIONS.md)** - AI provider integrations and how to add new ones

---

## Key Features

- **LLM-powered Chat** (`llm_chat.py`): Multi-provider support (OpenAI, Anthropic) with per-channel conversation history, image attachments, URL metadata extraction, and SVG rendering
- **Image Generation** (`image_gen_cog.py`): Prefix-based routing to multiple generation APIs (Replicate, etc.), image editing, outpainting, cost tracking, and daily rate limiting
- **GPT-2 Chat** (`gpt2_chat.py`): Lightweight text generation in designated channels
- **GIF Creation** (`gif_chain.py`): Build animated GIFs from reply chain images with adaptive compression
- **Starboard** (`starboard.py`): Mirror reacted bot messages to a dedicated channel with metadata logging
- **Admin Commands** (`admin.py`): Container restart, credit management, pricing/budget configuration
- **Usage Tracking** (`credits_cog.py`): Per-user credit balance and daily generation statistics
- **Basic Utilities** (`general.py`): Ping and help commands

---

## Requirements & Setup

### Prerequisites

- **Docker** and **docker-compose** (or Docker Desktop)
- **NVIDIA GPU** (for image generation worker; optional if image gen is disabled)
- **NVIDIA Container Toolkit** (for GPU support in Docker)
- **Python 3.10+** (for local development/testing)
- Discord bot token with proper intents enabled

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd hollingsbot3
   ```

2. **Configure environment**
   ```bash
   # Copy .env.example to .env (or create .env)
   cp .env.example .env
   # Edit .env with your configuration (Discord token, API keys, etc.)
   ```

3. **Start the bot**
   ```bash
   docker-compose up -d
   ```

4. **View logs**
   ```bash
   docker-compose logs -f bot
   docker-compose logs -f celery_text
   docker-compose logs -f celery_image
   ```

5. **Stop the bot**
   ```bash
   docker-compose down
   ```

### Local Development (Testing Only)

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Note: Full bot functionality requires Docker and external services
```

---

## Development Workflow

### Making Changes

1. **Edit code** in your editor (e.g., `src/hollingsbot/cogs/llm_chat.py`)
2. **Bot auto-reloads** within the running container (via watchfiles) for Python file changes
3. **For full effect** (especially Celery workers), restart containers:
   ```bash
   docker-compose down && docker-compose up -d
   ```
4. **View changes** in logs:
   ```bash
   docker-compose logs -f bot
   ```

### Testing

```bash
# Run tests (locally, not in Docker)
pytest

# Run specific test
pytest tests/test_image_gen_cog.py::test_seed_parsing

# Run with coverage
pytest --cov=src/hollingsbot
```

---

## Service Architecture

The bot runs as coordinated Docker services managed by `docker-compose`:

### **`bot`** - Main Discord Bot Instance
- **Purpose**: Listens for Discord messages and events
- **GPU**: Not required
- **Behavior**: Dispatches to cogs; monitors channels and reactions; auto-restarts via watchfiles
- **Volume**: `/app` (codebase mounted from host)

### **`celery_text`** - Text Generation Worker
- **Purpose**: Handles text generation tasks (LLM chat, GPT-2)
- **Queue**: `text`
- **Concurrency**: 1 (processes one task at a time)
- **GPU**: Not required
- **Tasks**: `generate_llm_chat_response`, `generate_text`

### **`celery_image`** - Image Generation Worker
- **Purpose**: Handles image generation/editing tasks
- **Queue**: `image`
- **Concurrency**: 2 (processes up to two images simultaneously)
- **GPU**: **Required** (NVIDIA GPU + Container Toolkit)
- **Tasks**: `generate_image`

### **`redis`** - Message Broker
- **Purpose**: Queues Celery tasks; routes to appropriate workers
- **Port**: 6379 (internal)

### **Volume Persistence**
- `prompt_db_data` → SQLite database (`prompts.db`) containing:
  - Image generation prompts and cost tracking
  - PR notifications
  - Starboard post metadata
- `generated/` → Generated images and state files

---

## Cogs Documentation

Each cog is a modular feature. This section documents each cog's purpose, implementation, key classes/methods, configuration, and dependencies.

### 1. General (`src/hollingsbot/cogs/general.py`)

**Purpose**: Provide basic bot commands (ping, help).

**Key Commands**:
- `!ping` - Responds with "Pong!" to verify bot is online
- `!help` - Displays comprehensive help documentation

**Key Classes**:
- `General(commands.Cog)` - Main cog class

**Key Methods**:
- `ping(ctx)` - Command handler for ping
- `help_cmd(ctx)` - Command handler for help; builds dynamic help text
- `_build_help_text()` - Constructs comprehensive help message with feature descriptions
- `_truncate_for_discord(text)` - Truncates help text to Discord's 2000-char limit

**Implementation Details**:
- Help text is dynamically built and includes all major features (image gen, LLM chat, GIF, admin, etc.)
- Respects Discord's 2000-character message limit with a 1900-char safety margin

**Configuration**: None (uses defaults)

**File Location**: `src/hollingsbot/cogs/general.py` (121 lines)

---

### 2. LLM Chat (`src/hollingsbot/cogs/llm_chat.py`)

**Purpose**: Multi-turn conversation with LLMs, supporting images, text attachments, URL metadata, SVG rendering, and tool execution.

**Key Features**:
- **Per-channel conversation history** - Deque-based storage with configurable limits
- **History warming** - Backfills channel history on first message for context
- **Image support** - Download, resize (max 2048px), convert to JPEG, base64-encode as data URLs
- **Text attachment handling** - Inlines text files (max 120KB) with placeholders for history
- **URL metadata extraction** - Fetches Open Graph/Twitter Card data; downloads images from URLs
- **SVG rendering** - Converts `<svg>` blocks and code blocks to PNG (requires `cairosvg`)
- **Per-user model preferences** - Users set preferred provider/model via `!model` command
- **Tool execution** - Parses and executes tool calls in LLM responses
- **Custom system prompts** - Global prompt with per-user overrides, cached with mtime checking
- **Reply context** - When users reply to messages, the bot includes that message's content and images

**Key Classes & Data Structures**:

- `LLMChatNewCog(commands.Cog)` - Main cog class
  - `channel_histories: dict[int, Deque[ConversationTurn]]` - Per-channel message history (deque-based)
  - `_history_locks: dict[int, asyncio.Lock]` - Thread-safe history access per channel
  - `_active_generations: dict[int, GenerationJob]` - Tracks ongoing generation tasks
  - `_warmed_channels: set[int]` - Tracks which channels have been backfilled
  - `model_preferences: dict[guild_id][user_id] = {provider, model}` - User model selection stored in JSON

- `ConversationTurn` - Represents a single message in history
  - `role: str` - "user" or "assistant"
  - `content: str` - Text body
  - `images: list[ImageAttachment]` - Associated images
  - `message_id, author_id, author_name: int | str` - Discord metadata

- `ImageAttachment` - Embedded image with metadata
  - `name, url, data_url: str` - File name, original URL, base64 data URL
  - `width, height, size: int` - Image dimensions and file size

- `GenerationJob` - Tracks a single generation task
  - `task: asyncio.Task[None]` - The running task
  - `result: AsyncResult` - Celery task result

**Key Methods**:
- `on_message(message)` - Listens to all messages; triggers response generation
- `_ensure_channel_warm(channel)` - Backfills history on first message
- `_prepare_user_turn(message)` - Builds conversation turn from Discord message
- `_extract_url_images(text)` - Fetches and embeds images from URLs in text
- `_extract_and_convert_svgs(text)` - Converts SVG blocks to PNG files
- `_run_generation(provider, model, conversation, job)` - Calls Celery task and polls result
- `_execute_tool_calls(text)` - Parses and executes tools in LLM response
- `_build_conversation_payload(history, current)` - Builds message array for LLM
- `_clean_mentions(message)` - Replace @user mentions with display names
- Commands: `!model <provider/model>` (set preference), `!models` (list), `!clear` (clear history), `!system` (manage prompt), `!h` (recent history)

**Environment Variables**:
- `DEFAULT_LLM_PROVIDER` - Default provider: `openai`, `anthropic` (default: openai)
- `DEFAULT_LLM_MODEL` - Default model name (default: gpt-4o)
- `AVAILABLE_MODELS` - CSV of provider/model pairs (e.g., `openai/gpt-4o,anthropic/claude-3-opus`)
- `LLM_WHITELIST_CHANNELS` - CSV of channel IDs where LLM chat is enabled
- `LLM_HISTORY_LIMIT` - Max turns to store per channel (default: 50)
- `LLM_MAX_TURNS_SENT` - Recent turns to send to LLM (default: 8)
- `TEXT_TIMEOUT` - Generation timeout in seconds (default: 180)
- `SYSTEM_PROMPT_FILE` - Path to system prompt file (default: `config/system_prompt.txt`)

**Configuration Files**:
- `config/system_prompt.txt` - Default system prompt (cached with mtime checking)
- `generated/llm_chat_new_state.json` - Persisted system prompt and model preferences
- `generated/notebook_state.json` - Tool/notebook context storage

**Size Limits**:
- `_MAX_TEXT_ATTACHMENT_BYTES = 120_000` - Max text attachment size
- `_IMAGE_MAX_EDGE = 2048` - Max image dimension (resized if larger)
- `_IMAGE_MAX_BYTES = 9_500_000` - Max image file size
- `_MESSAGE_CHUNK = 1900` - Max chars per Discord message (splits long responses)

**Dependencies**:
- `hollingsbot.tasks::generate_llm_chat_response` - Celery task for LLM generation
- `hollingsbot.tools` - Tool definitions and execution
- `hollingsbot.url_metadata` - URL metadata fetching
- `hollingsbot.settings` - System prompt caching
- `cairosvg` (optional) - SVG to PNG conversion

**File Location**: `src/hollingsbot/cogs/llm_chat.py` (1298 lines)

---

### 3. Image Generation (`src/hollingsbot/cogs/image_gen_cog.py`)

**Purpose**: Multi-API image generation with editing, outpainting, cost tracking, and rate limiting.

**Key Features**:
- **Prefix-based routing** - Different prefixes → different generators via `image_gen_config.json`
- **Image editing** - Send `edit: prompt` with image attachment/reply to modify images
- **Outpainting** - Send `zoom out` command to expand image context (uses LLM to generate descriptive prompt)
- **Prompt expansion** - `<a, b, c>` syntax generates multiple images
- **Seed parsing** - `{seed} prompt` syntax for reproducibility
- **Daily limits** - Per-user daily free budget with credit overflow
- **Cost tracking** - Database tracks generation costs, credits, and usage
- **Compression** - Adaptive JPEG compression to fit Discord's 25MB per-file limit
- **Captions** - Adds prompt text to image files

**Key Classes & Data Structures**:

- `ImageGenCog(commands.Cog)` - Main cog
  - `_prefix_map: dict[str, GeneratorSpec]` - Maps prefix → API/model config (loaded from `image_gen_config.json`)
  - `_allowed_channel_ids: set[int]` - Channels where image gen is enabled
  - `_edit_channel_ids: set[int]` - Separate allowlist for edit/outpaint
  - `_allow_dms: bool` - Whether to allow image gen in DMs
  - `_cost_tracker: CostTracker` - Tracks user budgets and costs
  - `_cfg_path: Path` - Path to config file (allows hot reload)
  - `_cfg_mtime: float` - Config file modification time (for detecting changes)
  - `_pending: set[asyncio.Task[None]]` - Tracks active generation tasks

- `GeneratorSpec` - Configuration for a single generator
  - `api: str` - API name (e.g., "replicate")
  - `model: str` - Model identifier (e.g., "black-forest-labs/flux-schnell")
  - `mode: str` - "generate", "edit", or "outpaint"
  - `price_per_image: float` - Cost in dollars

**Key Methods**:
- `on_message(message)` - Listens for image prompts with configured prefixes
- `_split_prompt(content)` - Extract prompt and spec from message content
- `_parse_seed_from_prompt(prompt)` - Extract seed from `{seed} prompt` syntax
- `_expand_prompt_list(prompt)` - Expand `<a, b, c>` to multiple prompts
- `_collect_images_for_editing(message)` - Gather images from message, reply, or history
- `_generate_outpaint_prompt(image_bytes)` - Use LLM to describe zoomed-out scene
- `_prepare_outpaint(message, images, needs_prompt)` - Create mask and prepare outpaint
- `_handle_generation(message, prompt, spec)` - Complete generation workflow
- `_execute_generation_tasks(...)` - Run multiple concurrent generation tasks
- `_prepare_discord_files(...)` - Build Discord File objects with compression
- `_process_and_send_results(...)` - Send results to Discord with error handling
- `_react(message, emoji, remove=False)` - Add/remove reaction emoji for feedback

**Reaction Emojis**:
- `🤔` (THINKING) - Generation in progress
- `✅` (SUCCESS) - Generation succeeded
- `❌` (FAILURE) - Generation failed

**Configuration**:
- `image_gen_config.json` - Maps prefixes to API/model specs with optional pricing
  ```json
  {
    "!": {
      "api": "replicate",
      "model": "black-forest-labs/flux-schnell"
    },
    "$": {
      "api": "replicate",
      "model": "google/imagen-4-fast",
      "price_per_image": 0.05,
      "daily_limit": 200
    },
    "edit:": {
      "api": "replicate",
      "model": "google/nano-banana",
      "mode": "edit"
    },
    "daily_free_budget": 0.50,
    "default_price_per_image": 0.03
  }
  ```

**Environment Variables**:
- `STABLE_DIFFUSION_CHANNEL_IDS` - CSV of allowed guild channels
- `EDIT_CHANNEL_IDS` - CSV of channels where edit/outpaint is allowed (in addition to main channels)
- `STABLE_DIFFUSION_ALLOW_DMS` - Allow image gen in DMs (default: 1/true)
- `STABLE_DIFFUSION_PRIVACY` - Disable logging of message content (default: 0/false)
- `IMAGE_TIMEOUT` - Generation timeout in seconds (default: 30)
- `IMAGE_OUTPUT_DIR` - Directory for generated images
- `PROMPT_DB_PATH` - SQLite database path (default: prompts.db)
- `DAILY_FREE_BUDGET` - Global daily free budget per user (default: 0.50)
- `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL` - For outpaint prompt generation

**File Size & Compression**:
- `_MAX_DISCORD_FILESIZE = 25 MiB` - Discord hard limit
- `_DEFAULT_FILESIZE_LIMIT = 8 MiB` - Conservative default for safety
- `_COMPRESSION_SAFETY_FACTOR = 0.95` - Compress to 95% of limit for safety

**Utility Functions**:
- `detect_mime_type(data: bytes) → str` - Detect image MIME type from magic bytes
- `detect_image_extension(data: bytes) → str` - Detect image extension
- `bytes_to_data_url(data: bytes) → str` - Convert image bytes to data URL
- `bytes_list_to_data_urls(images: list[bytes]) → list[str]`

**Dependencies**:
- `hollingsbot.tasks::generate_image` - Celery task for generation
- `hollingsbot.cost_tracking::CostTracker` - Budget/credit management
- `hollingsbot.prompt_db` - Database functions
- `hollingsbot.text_generators::get_text_generator` - LLM for outpaint prompts
- `hollingsbot.utils.image_utils::compress_image_to_fit` - JPEG compression
- `hollingsbot.utils.outpaint_utils::create_outpaint_images` - Mask generation
- `hollingsbot.caption::add_caption` - Add prompt text to images

**File Location**: `src/hollingsbot/cogs/image_gen_cog.py` (988 lines)

---

### 4. GPT-2 Chat (`src/hollingsbot/cogs/gpt2_chat.py`)

**Purpose**: Lightweight text generation in a designated channel using GPT-2 or similar small models.

**Key Features**:
- **Channel-specific** - Responds in a single configured channel
- **Simple prompts** - No conversation history; one-shot generation
- **Temperature randomization** - Adds variation using triangular distribution (biased toward 1.0)
- **Stale response handling** - If multiple messages arrive before generation finishes, only responds to the latest
- **Timeout handling** - Graceful error messages if generation exceeds timeout

**Key Classes & Data Structures**:

- `GPT2Chat(commands.Cog)` - Main cog
  - `channel_id: int | None` - Configured channel ID for responses
  - `api: str` - Text generation API (e.g., "huggingface")
  - `model: str` - Model name (e.g., "gpt2-medium")
  - `timeout: int` - Generation timeout in seconds
  - `_latest: dict[int, int]` - Tracks newest message ID per channel (prevents stale responses)

**Key Methods**:
- `on_message(message)` - Listens for messages in configured channel
- `_should_respond(message)` - Check if message should be answered (filters bots, keywords, channels)
- `_generate(prompt)` - Generate text with error handling and temperature
- `_celery_task(api, model, prompt, temperature)` - Call Celery task and await result
- `_generate_temperature()` - Generate random temperature (triangular distribution: 0.5 to 1.5, mode at 1.0)

**Configuration**:
- Constructor parameters: `channel_id`, `api`, `model`, `task_func` (for testing), `timeout`
- Environment variables: `GPT2_CHANNEL_ID`, `GPT2_RESPONSE_TIMEOUT`

**Environment Variables**:
- `GPT2_CHANNEL_ID` - Discord channel ID for GPT-2 responses
- `GPT2_RESPONSE_TIMEOUT` - Timeout in seconds (default: 180)

**Constants**:
- `_MAX_DISCORD_LEN = 2000` - Max message length
- `_IGNORED_KEYWORD = "enhance"` - Won't generate responses for this keyword
- `_TEMP_MIN = 0.5`, `_TEMP_MAX = 1.5`, `_TEMP_MODE = 1.0` - Temperature distribution

**Dependencies**:
- `hollingsbot.tasks::generate_text` - Celery task

**File Location**: `src/hollingsbot/cogs/gpt2_chat.py` (278 lines)

---

### 5. GIF Chain (`src/hollingsbot/cogs/gif_chain.py`)

**Purpose**: Create animated GIFs from image reply chains.

**Key Features**:
- **Reply chain traversal** - Walks both ancestors (backward) and descendants (forward) to collect all images
- **Adaptive compression** - Tries multiple strategies (scaling, frame dropping, palette reduction) to fit Discord size limit
- **Uniform frames** - Centers all images on uniform-sized canvases with black background
- **Frame limiting** - Configurable max number of frames to include
- **Dimension limiting** - Scales large images to max side length

**Key Classes & Data Structures**:

- `GifReplyChainCog(commands.Cog)` - Main cog
  - `_allowed_channel_ids: set[int]` - Channels where GIF creation is allowed
  - `_max_frames: int` - Maximum frames to include in GIF
  - `_max_side_px: int` - Maximum dimension for any frame (scales down if larger)
  - `_frame_ms: int` - Duration per frame in milliseconds
  - `_max_bytes: int` - Maximum GIF file size (Discord limit: 8 MB)
  - `_scan_limit: int` - How many recent messages to scan for reply chains

**Key Methods**:
- `on_message(message)` - Listen for "gif" command in replies
- `_validate_gif_request(message)` - Check message is valid GIF request
- `_collect_chain_images(start_msg)` - Walk reply chain and collect images
- `_collect_ancestor_messages(start, seen)` - Walk backward to root of chain
- `_collect_descendant_messages(start, seen)` - Walk forward to leaf of chain
- `_find_direct_reply(base)` - Find message that replies to another
- `_prepare_frames(image_blobs)` - Decode, scale, and canvas images
- `_create_uniform_canvas(images)` - Center all images on uniform-sized canvases
- `_build_gif_bytes(frames)` - Encode GIF with adaptive compression
- `_encode_gif(frames, palette_size)` - Quantize and save as GIF
- `_resize_frames(frames, scale)` - Scale all frames by factor
- `_create_and_send_gif(message, target)` - Main GIF creation workflow

**Compression Strategies**:
- **Scale factors**: [1.0, 0.85, 0.72, 0.6, 0.5, 0.4, 0.33]
- **Frame drops**: [1, 2, 3] (keep every Nth frame)
- **Palette sizes**: [128, 64, 32] (colors per palette)

**Constants**:
- `DEFAULT_MAX_FRAMES = 30` - Max frames in GIF
- `DEFAULT_MAX_SIDE_PX = 768` - Max dimension per frame
- `DEFAULT_FRAME_MS = 700` - Duration per frame
- `DEFAULT_MAX_BYTES = 8 * 1024 * 1024` - Discord limit (8 MiB)
- `DEFAULT_SCAN_LIMIT = 400` - Messages to scan for reply chains
- `CANVAS_BACKGROUND_COLOR = (0, 0, 0, 255)` - Black background

**Environment Variables**:
- `STABLE_DIFFUSION_CHANNEL_IDS` - CSV of allowed channels
- `GIF_CHAIN_MAX_FRAMES` - Max frames (default: 30)
- `GIF_CHAIN_MAX_SIDE` - Max dimension (default: 768)
- `GIF_CHAIN_FRAME_MS` - Duration per frame (default: 700)
- `GIF_CHAIN_MAX_BYTES` - Max size (default: 8 MB)
- `GIF_CHAIN_SCAN_LIMIT` - Messages to scan (default: 400)

**Dependencies**:
- PIL/Pillow - Image processing

**File Location**: `src/hollingsbot/cogs/gif_chain.py` (627 lines)

---

### 6. Starboard (`src/hollingsbot/cogs/starboard.py`)

**Purpose**: Mirror reacted bot messages to a dedicated channel.

**Key Features**:
- **Reaction-based triggering** - Any reaction on a bot message can trigger reposting
- **Channel filtering** - Whitelist/blacklist channels for starboard posts
- **Duplicate prevention** - Cache tracks seen message IDs to avoid reposts
- **Metadata logging** - Database records which reactions triggered posts and from whom
- **Embed support** - Fallback handling for archived messages with only embed data

**Key Classes & Data Structures**:

- `Starboard(commands.Cog)` - Main cog
  - `starboard_channel_id: int | None` - Channel ID where messages are reposted
  - `ignore_channel_ids: set[int]` - Channels to exclude from starboard
  - `whitelist_channel_ids: set[int]` - If non-empty, only channels in this set post to starboard
  - `_seen_message_ids: deque[int]` - Cache of recently seen message IDs (prevents duplicates)

**Key Methods**:
- `on_raw_reaction_add(payload)` - Listen for reactions on messages
- `_is_valid_reactor(payload)` - Check reactor is a human (not a bot)
- `_is_channel_eligible(channel)` - Check channel passes whitelist/blacklist
- `_is_message_eligible(message)` - Check message is from a bot and not already seen
- `_fetch_message_safely(channel, msg_id)` - Fetch message with error handling
- `_send_to_starboard(channel, message)` - Repost to starboard
- `_format_forward_content(message)` - Build formatted preview for starboard
- `_log_starboard_entry(payload, message, ...)` - Record post to database
- `_extract_message_attachments(message)` - Get attachment metadata
- `_serialize_attachments(message)` - JSON encode attachments for storage

**Constants**:
- `MAX_CONTENT_PREVIEW_LENGTH = 800` - Max chars for content preview
- `MAX_FORWARD_MESSAGE_LENGTH = 1900` - Max Discord message length
- `SEEN_MESSAGE_CACHE_SIZE = 2048` - Size of duplicate cache

**Helper Functions**:
- `_parse_channel_ids(raw: str) → set[int]` - Parse comma-separated channel IDs
- `_parse_channel_id(raw: str) → int | None` - Parse single channel ID

**Environment Variables**:
- `STARBOARD_CHANNEL_ID` - Channel ID for starboard posts
- `STARBOARD_IGNORE_CHANNELS` - CSV of channels to exclude
- `STARBOARD_WHITELIST_CHANNEL_IDS` - CSV of channels to include (if non-empty, overrides ignore list)

**Dependencies**:
- `hollingsbot.prompt_db::log_starboard_post` - Database logging

**File Location**: `src/hollingsbot/cogs/starboard.py` (447 lines)

---

### 7. Admin (`src/hollingsbot/cogs/admin.py`)

**Purpose**: Operational commands for container restart and credit management.

**Key Features**:
- **Container restart** - Graceful Docker container restart with fallback to process restart
- **Admin-only commands** - `!grant`, `!balance`, `!set_price`, `!set_budget` restricted to admins
- **Credit management** - Grant/deduct credits, check balances (users can check own; admins check anyone)
- **Dynamic pricing** - Admins can adjust per-prefix pricing and global daily budget
- **Cost tracking integration** - Reads/updates cost tracker and config files

**Key Classes & Data Structures**:

- `Admin(commands.Cog)` - Main cog
  - `_cost_tracker: CostTracker` - Track user credits and budgets
  - `_admin_ids: set[int]` - IDs of users with admin privileges

**Key Methods**:
- `reset(ctx)` - Restart containers (command)
- `_restart_containers(ctx)` - Main restart logic with fallbacks
- `_try_override_command()` - Execute custom `RESET_COMMAND` if set
- `_try_docker_compose_restart(compose_file)` - Try `docker compose` then `docker-compose`
- `_execute_compose_restart(docker_bin, subcommand, compose_file)` - Run compose restart
- `_fallback_to_bot_restart(ctx)` - Last-resort: restart bot process only
- `_run_shell_command(command)` - Execute shell command asynchronously
- `_run_subprocess(cmd, cwd)` - Execute subprocess command asynchronously
- `_find_compose_file()` - Search project tree for compose file
- `_is_admin(user_id)` - Check if user is admin

**Commands**:
- `!reset` - Restart containers
- `!grant @user amount` - Grant credits (admin only)
- `!balance [@user]` - Check balance (self or user if admin)
- `!set_price prefix price` - Set pricing for prefix (admin only)
- `!set_budget budget` - Set global daily free budget (admin only)

**Restart Logic Hierarchy**:
1. Try custom `RESET_COMMAND` if set (env var)
2. Try modern `docker compose` CLI
3. Try legacy `docker-compose` CLI
4. Fallback to bot process restart (container will auto-restart if using restart policy)

**Configuration**:
- Environment variables: `RESET_COMMAND` (custom restart command), `PROMPT_DB_PATH`, `DAILY_FREE_BUDGET`, `ADMIN_USER_IDS`
- Config file: `image_gen_config.json` (read/written for pricing/budget updates)

**Constants**:
- `_COMPOSE_FILE_NAMES` - Filenames to search for: docker-compose.yml, docker-compose.yaml, compose.yml, compose.yaml
- `_REPO_ROOT_OFFSET = 3` - Offset from cog directory to repo root

**Dependencies**:
- `hollingsbot.cost_tracking::CostTracker` - Credit management
- `hollingsbot.settings::get_admin_user_ids` - Load admin list

**File Location**: `src/hollingsbot/cogs/admin.py` (481 lines)

---

### 8. Credits (`src/hollingsbot/cogs/credits_cog.py`)

**Purpose**: User-facing commands for checking usage and credit balance.

**Key Features**:
- **Balance checking** - `!usage` shows daily free budget used, credits spent, and remaining credit balance
- **Time to reset** - Shows when daily budget resets
- **Generation count** - Tracks total generations today

**Key Classes & Data Structures**:

- `CreditsCog(commands.Cog)` - Main cog
  - `_cost_tracker: CostTracker` - Track user credits and budgets

**Key Commands**:
- `!usage` - Display user's usage and balance (shows free budget used/total, credits spent, generation count, time to reset)

**Implementation Details**:
- Queries cost tracker for user status (budget used, credits, generation count, reset time)
- Formats response with clear breakdown of free vs. paid usage
- Displays reset time (when daily free budget resets)

**Configuration**:
- Environment variables: `PROMPT_DB_PATH`, `DAILY_FREE_BUDGET`

**Dependencies**:
- `hollingsbot.cost_tracking::CostTracker` - Budget tracking

**File Location**: `src/hollingsbot/cogs/credits_cog.py` (81 lines)

---

## Configuration

### Environment Variables

**Core Discord**:
- `DISCORD_TOKEN` - Discord bot token (required)
- `BOT_RESTART_INTERVAL` - Auto-restart interval in seconds (default: 21600 / 6 hours)

**Message Broker & Tasks**:
- `CELERY_BROKER_URL` - Redis broker URL (default: redis://redis:6379/0)
- `CELERY_RESULT_BACKEND` - Celery result backend (default: redis://redis:6379/0)

**LLM Chat**:
- `DEFAULT_LLM_PROVIDER` - Default LLM provider: `openai`, `anthropic` (default: openai)
- `DEFAULT_LLM_MODEL` - Default LLM model (default: gpt-4o)
- `AVAILABLE_MODELS` - CSV of available models (format: `provider/model,provider/model`)
- `LLM_WHITELIST_CHANNELS` - CSV of channel IDs where LLM chat is enabled
- `LLM_HISTORY_LIMIT` - Max conversation turns to store per channel (default: 50)
- `LLM_MAX_TURNS_SENT` - Recent turns to include in LLM context (default: 8)
- `TEXT_TIMEOUT` - LLM generation timeout in seconds (default: 180)
- `SYSTEM_PROMPT_FILE` - Path to system prompt file (default: config/system_prompt.txt)

**Image Generation**:
- `STABLE_DIFFUSION_CHANNEL_IDS` - CSV of guild channels where image gen is enabled
- `EDIT_CHANNEL_IDS` - CSV of channels where edit/outpaint is allowed (in addition to main channels)
- `STABLE_DIFFUSION_ALLOW_DMS` - Allow image gen in DMs (default: 1/true)
- `STABLE_DIFFUSION_PRIVACY` - Disable logging of message content (default: 0/false)
- `IMAGE_TIMEOUT` - Image generation timeout in seconds (default: 30)
- `IMAGE_OUTPUT_DIR` - Directory for generated images
- `PROMPT_DB_PATH` - SQLite database path (default: prompts.db)
- `DAILY_FREE_BUDGET` - Daily free budget per user in dollars (default: 0.50)

**GPT-2 Chat**:
- `GPT2_CHANNEL_ID` - Discord channel ID for GPT-2 responses
- `GPT2_RESPONSE_TIMEOUT` - Generation timeout in seconds (default: 180)

**GIF Chain**:
- `GIF_CHAIN_MAX_FRAMES` - Max frames in GIF (default: 30)
- `GIF_CHAIN_MAX_SIDE` - Max dimension per frame in pixels (default: 768)
- `GIF_CHAIN_FRAME_MS` - Duration per frame in milliseconds (default: 700)
- `GIF_CHAIN_MAX_BYTES` - Max GIF file size (default: 8 MB)
- `GIF_CHAIN_SCAN_LIMIT` - Max messages to scan for reply chains (default: 400)

**Starboard**:
- `STARBOARD_CHANNEL_ID` - Channel ID for starboard posts
- `STARBOARD_IGNORE_CHANNELS` - CSV of channels to exclude
- `STARBOARD_WHITELIST_CHANNEL_IDS` - CSV of channels to include (if non-empty, overrides ignore list)

**Admin**:
- `RESET_COMMAND` - Custom restart command (optional)
- `ADMIN_USER_IDS` - CSV of Discord user IDs with admin privileges

### Configuration Files

**`image_gen_config.json`** - Image generator routing and pricing:
```json
{
  "!": {
    "api": "replicate",
    "model": "black-forest-labs/flux-schnell"
  },
  "$": {
    "api": "replicate",
    "model": "google/imagen-4-fast",
    "price_per_image": 0.05,
    "daily_limit": 200
  },
  "edit:": {
    "api": "replicate",
    "model": "google/nano-banana",
    "mode": "edit"
  },
  "daily_free_budget": 0.50,
  "default_price_per_image": 0.03
}
```

**`config/system_prompt.txt`** - Default LLM system prompt (can be overridden with `SYSTEM_PROMPT_FILE` env var or `!system` command)

**`generated/llm_chat_new_state.json`** - Persisted LLM state:
```json
{
  "system_prompt": "...",
  "model_preferences": {
    "guild_id": {
      "user_id": {
        "provider": "openai",
        "model": "gpt-4o"
      }
    }
  }
}
```

**`generated/notebook_state.json`** - Tool/notebook context (managed by tool system)

### Database (`prompts.db`)

SQLite database persists:
- Image generation prompts, costs, and daily tracking
- PR notifications
- Starboard post metadata

Location: Volume-mounted `prompt_db_data` (persists across container restarts)

---

## Advanced Topics

### Adding a New Cog

1. Create `src/hollingsbot/cogs/my_feature.py`:
   ```python
   from discord.ext import commands

   class MyFeature(commands.Cog):
       def __init__(self, bot: commands.Bot):
           self.bot = bot

       @commands.Cog.listener()
       async def on_message(self, message: discord.Message):
           # Handle message
           pass

   async def setup(bot: commands.Bot):
       await bot.add_cog(MyFeature(bot))
   ```

2. Register in `src/hollingsbot/__main__.py`:
   ```python
   await bot.load_extension("hollingsbot.cogs.my_feature")
   ```

3. Restart the bot:
   ```bash
   docker-compose down && docker-compose up -d
   ```

### Adding a New Image Generator

1. Create `src/hollingsbot/image_generators/my_api.py` implementing `ImageGeneratorAPI`:
   ```python
   class MyGenerator(ImageGeneratorAPI):
       async def generate(self, prompt: str, seed: int, ...):
           # Return image bytes or file path
           pass
   ```

2. Register in `src/hollingsbot/image_generators/__init__.py`:
   ```python
   def get_image_generator(api, model):
       if api == "my_api":
           return MyGenerator(model)
       # ... other APIs
   ```

3. Add entry to `image_gen_config.json`:
   ```json
   {
     "prefix": {
       "api": "my_api",
       "model": "model-name"
     }
   }
   ```

### Adding a New Text Generator

1. Create `src/hollingsbot/text_generators/my_provider.py` implementing `TextGeneratorAPI`:
   ```python
   class MyTextGen(TextGeneratorAPI):
       async def generate(self, messages: list[dict]):
           # Return generated text
           pass
   ```

2. Register in `src/hollingsbot/text_generators/__init__.py`:
   ```python
   def get_text_generator(provider, model):
       if provider == "my_provider":
           return MyTextGen(model)
       # ... other providers
   ```

3. Use in LLM chat by setting `DEFAULT_LLM_PROVIDER=my_provider` and `DEFAULT_LLM_MODEL=model-name`

### Tool System

Tools allow LLMs to execute code. Define tools in `src/hollingsbot/tools/`:

1. Create `src/hollingsbot/tools/my_tool.py` implementing `Tool`:
   ```python
   class MyTool(Tool):
       name = "my_tool"
       description = "Does something useful"

       def execute(self, **kwargs):
           # Return result
           pass
   ```

2. Register in `src/hollingsbot/tools/__init__.py` in `AVAILABLE_TOOLS`

3. LLM can now call `TOOL_CALL: my_tool(param1=value1, ...)`

### Debugging

**View bot logs**:
```bash
docker-compose logs -f bot
```

**View Celery task logs**:
```bash
docker-compose logs -f celery_text
docker-compose logs -f celery_image
```

**Enter bot container**:
```bash
docker-compose exec bot bash
```

**Inspect SQLite database**:
```bash
sqlite3 prompts.db
> .tables
> .schema prompts
> SELECT * FROM prompts LIMIT 5;
```

### Performance Considerations

- **History warming** - First message in a channel triggers backfill; can be slow for large channels
- **Image compression** - Tries multiple strategies; may take time for large batches
- **GPU memory** - Image generation worker concurrency is limited to prevent OOM
- **Redis persistence** - Ensure Redis volume is backed up (tasks are queued, not persisted on restart)
- **Database cleanup** - Old prompt records should be archived/pruned periodically

### Known Limitations

- **Conversation history is in-memory** - Lost on bot restart (design choice for privacy)
- **No user authentication** - Any bot user can use all features
- **Rate limiting** - Per-user daily budgets prevent abuse but aren't per-minute
- **Image generation timeout** - Fixed 30s timeout; some models may need more time
- **SVG rendering** - Requires `cairosvg` library (optional); falls back to raw SVG if unavailable

---

## Troubleshooting

**Bot won't start**:
- Check `DISCORD_TOKEN` is set and valid
- Verify bot intents are configured in Discord Developer Portal
- Check Docker services are running: `docker-compose ps`

**Image generation hangs**:
- Check GPU worker status: `docker-compose logs celery_image`
- Verify GPU is accessible: `nvidia-smi`
- Increase timeout: `IMAGE_TIMEOUT=60`

**LLM chat not responding**:
- Verify channel is in `LLM_WHITELIST_CHANNELS`
- Check LLM provider API keys are set
- View logs: `docker-compose logs -f bot`
- Increase timeout: `TEXT_TIMEOUT=300`

**Starboard not working**:
- Verify `STARBOARD_CHANNEL_ID` is set
- Check channel is readable and writable by bot
- Ensure messages being reacted to are from bots

**GIF creation fails**:
- Check channel is in `STABLE_DIFFUSION_CHANNEL_IDS`
- Verify reply chain has valid images
- Increase `GIF_CHAIN_MAX_BYTES` if GIF is too large
- Check Pillow is installed: `pip list | grep Pillow`

---

## Contributing

See `CLAUDE.md` for development guidelines and architecture notes.

---

## License

[Add your license here]
