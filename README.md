# Hollingsbot3

A feature-rich Discord bot for image generation, AI chat, and server utilities, built with discord.py and asynchronous task processing via Celery.

## Overview

Hollingsbot3 is a multi-purpose Discord bot that provides:

- **Image Generation**: Multiple image generation APIs with prefix-based routing (FLUX, Imagen, SVG generation)
- **Image Editing & Outpainting**: Transform and expand existing images with AI
- **LLM Chat**: Conversational AI with vision support and conversation history (Claude, GPT, etc.)
- **Cost Tracking**: Built-in credit system with daily free budgets and user balances
- **Reply Chain GIFs**: Automatically create animated GIFs from Discord reply threads
- **Starboard**: Repost reacted messages to a dedicated channel
- **Admin Controls**: Container management and credit administration

## Key Features

- **Multi-Model Support**: Route prompts to different models using configurable prefixes
- **GPU Acceleration**: Separate worker for GPU-intensive image generation
- **Conversation History**: LLM chat maintains per-channel context with image support
- **URL Metadata Extraction**: Automatically fetches and processes images from URLs (Open Graph, Twitter Cards)
- **SVG Rendering**: Converts SVG code blocks to PNG for Discord compatibility
- **Tool Execution**: LLM can execute Python code via notebook environment
- **Auto-reload**: Development mode with watchfiles for rapid iteration
- **Cost Management**: Per-user daily budgets with admin-controlled credit system

## Requirements

### System Requirements

- Docker with Docker Compose
- **NVIDIA GPU with NVIDIA Container Toolkit** (required for image generation worker)
- Linux/macOS or Windows with WSL2
- Minimum 8GB RAM (16GB recommended)
- 10GB+ free disk space

### API Keys

Required API keys (set in `.env` file):

- `DISCORD_TOKEN`: Discord bot token
- `REPLICATE_API_TOKEN`: For image generation models
- `ANTHROPIC_API_KEY`: For Claude models (optional)
- `OPENAI_API_KEY`: For GPT models (optional)
- `GITHUB_TOKEN`: For PR notifications (optional)

## Setup

### 1. Clone and Configure

```bash
git clone <repository-url>
cd hollingsbot3
cp .env.example .env  # If available, or create from scratch
```

### 2. Configure Environment

Edit `.env` and set required variables:

```bash
# Core
DISCORD_TOKEN=your_discord_token_here
REPLICATE_API_TOKEN=your_replicate_token_here

# LLM Provider (anthropic or openai)
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_API_KEY=your_anthropic_key_here

# Channel Allowlists (comma-separated channel IDs)
STABLE_DIFFUSION_CHANNEL_IDS=123456789,987654321
LLM_WHITELIST_CHANNELS=123456789
```

### 3. Configure Image Generation

Edit `src/hollingsbot/image_gen_config.json` to customize prefixes and pricing:

```json
{
  "daily_free_budget": 1.0,
  "default_price_per_image": 0.03,
  "!": {
    "api": "replicate",
    "model": "black-forest-labs/flux-schnell",
    "price_per_image": 0.02
  },
  "edit:": {
    "api": "replicate",
    "model": "google/nano-banana",
    "mode": "edit",
    "price_per_image": 0.04
  }
}
```

### 4. Build and Start

```bash
# Build containers
docker-compose up --build -d

# View logs
docker-compose logs -f bot
docker-compose logs -f celery_image
docker-compose logs -f celery_text
```

## Development Workflow

### Making Code Changes

After any code change, restart the containers:

```bash
docker-compose down && docker-compose up -d
```

While watchfiles provides auto-reload within containers, you must restart for changes to take effect.

### Running Tests

Tests run outside Docker and require local dependencies:

```bash
pip install -r requirements.txt
pytest
```

### Viewing Logs

```bash
# Follow all logs
docker-compose logs -f

# Specific service
docker-compose logs -f bot
docker-compose logs -f celery_image
docker-compose logs -f celery_text
```

### Database Access

The SQLite database is persisted in the `prompt_db_data` Docker volume:

```bash
# Access database directly
docker exec -it hollingsbot3-bot-1 sqlite3 /data/prompts.db
```

## Cogs Documentation

### Image Generation Cog (`image_gen_cog.py`)

**Purpose**: Routes image generation requests to various APIs based on message prefixes.

**Technical Implementation**:
- **Prefix Matching**: Parses message content and matches against `image_gen_config.json` prefixes
- **Mode Detection**: Supports `generate`, `edit`, and `outpaint` modes
- **Cost Tracking**: Integrates with `CostTracker` for budget enforcement and credit management
- **Celery Integration**: Dispatches generation tasks to the `image` queue with configurable timeouts
- **Multi-Image Support**: Handles generators that return multiple images via `generate_many()` method
- **Compression Pipeline**: Automatically compresses images to fit Discord's file size limits
- **Caption System**: Adds text captions to generated images (skipped for edit/outpaint modes)

**Key Classes**:
- `ImageGenCog`: Main cog class with message listener
- `GeneratorSpec`: Dataclass holding API, model, mode, and pricing configuration

**Data Structures**:
- `_prefix_map`: Maps prefixes to GeneratorSpec instances
- `_pending`: Tracks active generation tasks for cleanup

**File Paths**:
- Config: `src/hollingsbot/image_gen_config.json`
- Implementation: `src/hollingsbot/cogs/image_gen_cog.py`
- Generators: `src/hollingsbot/image_generators/`
- Task: `src/hollingsbot/tasks.py::generate_image`

**Dependencies**:
- `tasks.generate_image`: Celery task for async generation
- `prompt_db.bulk_add_prompts`: Database logging
- `cost_tracking.CostTracker`: Budget enforcement
- `image_generators.get_image_generator`: Generator factory pattern

**Configuration Options**:
- `STABLE_DIFFUSION_CHANNEL_IDS`: Comma-separated channel IDs where generation is allowed
- `EDIT_CHANNEL_IDS`: Additional channels where `edit:` prefix is allowed
- `STABLE_DIFFUSION_ALLOW_DMS`: Enable/disable DM support (default: `1`)
- `IMAGE_TIMEOUT`: Timeout for generation tasks in seconds (default: `30`)
- `IMAGE_OUTPUT_DIR`: Directory for generated images (default: `/app/generated`)

**Architectural Patterns**:
- Factory pattern for generator selection
- Strategy pattern for compression algorithms
- Observer pattern for reactions (thinking/success/failure emojis)

---

### LLM Chat Cog (`llm_chat.py`)

**Purpose**: Provides conversational AI with multi-modal support, conversation history, and tool execution capabilities.

**Technical Implementation**:
- **History Management**: Per-channel conversation history using `Deque[ConversationTurn]` with configurable max length
- **History Warming**: On first message, backfills history by reading past Discord messages
- **Image Processing**: Downloads, resizes (max 2048px), converts to JPEG, and base64-encodes as data URLs
- **Text Attachments**: Inlines uploaded text files (max 120KB) with truncation markers
- **Reply Context**: Automatically includes replied-to message content and images
- **URL Metadata**: Extracts Open Graph/Twitter Card data and downloads referenced images
- **SVG Rendering**: Converts SVG code blocks to PNG using cairosvg
- **Tool Execution**: Parses and executes `TOOL_CALL:` markers in LLM responses
- **Model Preferences**: Per-user, per-guild model selection stored in `llm_chat_new_state.json`
- **System Prompts**: User-customizable system prompts with file-based defaults

**Key Classes**:
- `LLMChatNewCog`: Main cog class
- `ConversationTurn`: History entry with role, content, images, and Discord metadata
- `ModelTurn`: Prepared turn for LLM with full content (images, text attachments)
- `ImageAttachment`: Image metadata with data URL, dimensions, and file size
- `GenerationJob`: Tracks active Celery task and result

**Data Structures**:
- `channel_histories`: `dict[int, Deque[ConversationTurn]]` for per-channel context
- `_history_locks`: `dict[int, asyncio.Lock]` for thread-safe history access
- `_warmed_channels`: `set[int]` tracks channels with pre-loaded history
- `_active_generations`: `dict[int, GenerationJob]` for cancellation support
- `model_preferences`: Nested dict `[guild_id][user_id]` → `{provider, model}`

**File Paths**:
- Implementation: `src/hollingsbot/cogs/llm_chat.py`
- State: `generated/llm_chat_new_state.json`
- System Prompt: `config/system_prompt.txt` (or `SYSTEM_PROMPT_FILE` env var)
- Notebook State: `generated/notebook_state.json`
- Task: `src/hollingsbot/tasks.py::generate_llm_chat_response`
- Tool Definitions: `src/hollingsbot/tools/__init__.py`

**Dependencies**:
- `tasks.generate_llm_chat_response`: Celery task for LLM generation
- `text_generators.get_text_generator`: Generator factory
- `tools.execute_tool_call`: Tool execution system
- `url_metadata.extract_url_metadata`: URL processing
- `cairosvg`: SVG to PNG conversion (optional)

**Configuration Options**:
- `LLM_WHITELIST_CHANNELS`: Comma-separated channel IDs (empty disables feature)
- `LLM_HISTORY_LIMIT`: Max conversation turns to store per channel (default: `50`)
- `LLM_MAX_TURNS_SENT`: Max history turns sent to LLM (default: `8`)
- `TEXT_TIMEOUT`: Generation timeout in seconds (default: `180`)
- `DEFAULT_LLM_PROVIDER`: Default provider (`openai` or `anthropic`)
- `DEFAULT_LLM_MODEL`: Default model name
- `AVAILABLE_MODELS`: Comma-separated list of `provider/model` strings
- `SYSTEM_PROMPT_FILE`: Path to default system prompt file

**Architectural Patterns**:
- Deque-based ring buffer for conversation history
- Lazy loading with history warming on first use
- Two-phase turn preparation (full content for LLM, lightweight for history)
- Factory pattern for text generator selection
- Strategy pattern for provider-specific message formatting

**Commands**:
- `!models`: List available LLM models
- `!model [provider/model]`: View or set user's preferred model
- `!clear`: Clear channel conversation history
- `!system [text|reset]`: View, set, or reset system prompt
- `!h`: Display help information

---

### Starboard Cog (`starboard.py`)

**Purpose**: Mirrors bot messages to a designated starboard channel when users react to them.

**Technical Implementation**:
- **Event-Driven**: Listens to `on_raw_reaction_add` events for real-time processing
- **Deduplication**: Maintains `deque[int]` cache of seen message IDs (maxlen 2048)
- **Channel Filtering**: Supports both whitelist and blacklist for source channels
- **Lazy Resolution**: Attempts cached channel lookup before fetching
- **Metadata Logging**: Records starboard posts in database with attachment serialization
- **Attachment Handling**: Serializes both direct attachments and embed images to JSON

**Key Classes**:
- `Starboard`: Main cog class with reaction listener

**Data Structures**:
- `_seen_message_ids`: `deque[int]` for duplicate prevention
- `ignore_channel_ids`: `set[int]` for blacklisted channels
- `whitelist_channel_ids`: `set[int]` for whitelisted channels (if set, only these are eligible)

**File Paths**:
- Implementation: `src/hollingsbot/cogs/starboard.py`
- Database: `prompts.db::starboard_posts` table

**Dependencies**:
- `prompt_db.log_starboard_post`: Database logging function

**Configuration Options**:
- `ENABLE_STARBOARD`: Enable starboard feature (`1` to enable)
- `STARBOARD_CHANNEL_ID`: Target channel ID for reposts
- `STARBOARD_IGNORE_CHANNELS`: Comma-separated channel IDs to ignore
- `STARBOARD_WHITELIST_CHANNEL_IDS`: If set, only these channels are eligible

**Architectural Patterns**:
- Event-driven architecture with raw event listeners
- Cache-aside pattern for message deduplication
- Strategy pattern for attachment extraction (handles both direct attachments and embeds)

---

### General Cog (`general.py`)

**Purpose**: Provides basic utility commands available to all users.

**Technical Implementation**:
- Simple command handlers with no complex state
- Auto-truncation for help text exceeding Discord limits

**Key Classes**:
- `General`: Main cog class

**Commands**:
- `!ping`: Returns "Pong!" to test bot responsiveness
- `!help`: Displays comprehensive bot help documentation

**File Paths**:
- Implementation: `src/hollingsbot/cogs/general.py`

---

### Admin Cog (`admin.py`)

**Purpose**: Provides operational commands for container management and admin-only credit system controls.

**Technical Implementation**:
- **Container Restart**: Multi-strategy restart with fallbacks:
  1. Custom override command (via `RESET_COMMAND` env var)
  2. Modern Docker CLI (`docker compose`)
  3. Legacy Docker Compose (`docker-compose`)
  4. Bot process restart (last resort)
- **Compose File Discovery**: Searches upward from working directory and module path
- **Credit Management**: Admin commands for granting credits and checking balances
- **Config Editing**: In-place JSON editing for pricing and budget configuration

**Key Classes**:
- `Admin`: Main cog class with admin permission checks

**File Paths**:
- Implementation: `src/hollingsbot/cogs/admin.py`
- Config: `src/hollingsbot/image_gen_config.json`
- Database: `prompts.db`

**Dependencies**:
- `cost_tracking.CostTracker`: Credit system integration
- `settings.get_admin_user_ids`: Admin permission lookup

**Configuration Options**:
- `ADMIN_USER_IDS`: Comma-separated Discord user IDs with admin privileges
- `RESET_COMMAND`: Custom override command for container restart
- `PROMPT_DB_PATH`: Path to SQLite database (default: `prompts.db`)
- `DAILY_FREE_BUDGET`: Default daily free budget (default: `0.50`)

**Commands**:
- `!reset`: Restart Docker containers or bot process
- `!grant @user <amount>`: Grant credits to a user (admin only)
- `!balance [@user]`: Check balance and usage (own or admin-only for others)
- `!set_price <prefix> <price>`: Update price for a prefix (admin only)
- `!set_budget <amount>`: Update global daily free budget (admin only)

**Architectural Patterns**:
- Strategy pattern for restart methods with fallback chain
- Guard pattern for admin permission checks

---

### Credits Cog (`credits_cog.py`)

**Purpose**: User-facing commands for checking image generation usage and credit balances.

**Technical Implementation**:
- Read-only access to cost tracker
- Formatted status messages with time-to-reset calculation

**Key Classes**:
- `CreditsCog`: Main cog class

**File Paths**:
- Implementation: `src/hollingsbot/cogs/credits_cog.py`
- Database: `prompts.db`

**Dependencies**:
- `cost_tracking.CostTracker`: Balance and usage retrieval

**Configuration Options**:
- `PROMPT_DB_PATH`: Path to SQLite database
- `DAILY_FREE_BUDGET`: Daily free budget amount

**Commands**:
- `!usage`: Display current usage, credit balance, and time to reset

---

### GPT-2 Chat Cog (`gpt2_chat.py`)

**Purpose**: Responds to all messages in a designated channel using GPT-2 (or similar) models.

**Technical Implementation**:
- **Latest-Only Processing**: Tracks newest message ID per channel, discards stale generations
- **Temperature Randomization**: Uses triangular distribution for varied outputs
- **Celery Integration**: Dispatches to `text` queue with configurable timeout
- **Keyword Filtering**: Ignores specific keywords (e.g., "enhance")

**Key Classes**:
- `GPT2Chat`: Main cog class with message listener

**Data Structures**:
- `_latest`: `dict[int, int]` maps channel ID to newest message ID

**File Paths**:
- Implementation: `src/hollingsbot/cogs/gpt2_chat.py`
- Task: `src/hollingsbot/tasks.py::generate_text`

**Dependencies**:
- `tasks.generate_text`: Celery task for text generation
- `text_generators.get_text_generator`: Generator factory

**Configuration Options**:
- `GPT2_CHANNEL_ID`: Channel ID where bot responds to all messages
- `GPT2_RESPONSE_TIMEOUT`: Timeout in seconds (default: `180`)

**Architectural Patterns**:
- Latest-wins pattern for concurrent message handling
- Factory pattern for generator selection

---

### GIF Chain Cog (`gif_chain.py`)

**Purpose**: Creates animated GIFs from all images in a Discord reply chain.

**Technical Implementation**:
- **Bidirectional Traversal**: Walks reply chain both backward (ancestors) and forward (descendants)
- **Image Collection**: Extracts all image attachments up to `GIF_CHAIN_MAX_FRAMES`
- **Frame Preparation**:
  1. Decode images to RGBA
  2. Scale to fit `GIF_CHAIN_MAX_SIDE` constraint
  3. Center on uniform canvases (max dimensions across all images)
  4. Convert to RGB for GIF encoding
- **Adaptive Compression**: Tries combinations of:
  - Scale factors: `[1.0, 0.85, 0.72, 0.6, 0.5, 0.4, 0.33]`
  - Frame drops: `[1, 2, 3]` (keep every Nth frame)
  - Palette sizes: `[128, 64, 32]` colors
- **Size Enforcement**: Ensures final GIF fits within Discord's file size limit

**Key Classes**:
- `GifReplyChainCog`: Main cog class with message listener

**File Paths**:
- Implementation: `src/hollingsbot/cogs/gif_chain.py`

**Configuration Options**:
- `STABLE_DIFFUSION_CHANNEL_IDS`: Comma-separated channel IDs where GIF generation is allowed
- `GIF_CHAIN_MAX_FRAMES`: Maximum frames in output GIF (default: `30`)
- `GIF_CHAIN_MAX_SIDE`: Max pixel dimension for scaling (default: `768`)
- `GIF_CHAIN_FRAME_MS`: Frame duration in milliseconds (default: `700`)
- `GIF_CHAIN_MAX_BYTES`: Max output file size (default: `8388608` = 8 MiB)
- `GIF_CHAIN_SCAN_LIMIT`: Max messages to scan when searching for replies (default: `400`)

**Usage**:
Reply to any message with the text "gif" to generate a GIF from the entire reply chain.

**Architectural Patterns**:
- Iterator pattern for chain traversal
- Strategy pattern for compression with fallback chain
- Template method pattern for frame preparation pipeline

---

## Configuration

### Environment Variables

#### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DISCORD_TOKEN` | Discord bot token (required) | - |
| `GITHUB_TOKEN` | GitHub token for PR notifications | - |
| `CELERY_BROKER_URL` | Redis broker URL | `redis://redis:6379/0` |
| `CELERY_RESULT_BACKEND` | Redis result backend URL | `redis://redis:6379/0` |

#### Bot Behavior

| Variable | Description | Default |
|----------|-------------|---------|
| `BOT_RESTART_INTERVAL` | Auto-restart interval in seconds | `21600` (6 hours) |

#### Starboard

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_STARBOARD` | Enable starboard feature | `0` |
| `STARBOARD_CHANNEL_ID` | Target channel ID | - |
| `STARBOARD_IGNORE_CHANNELS` | Comma-separated channel IDs to ignore | - |
| `STARBOARD_WHITELIST_CHANNEL_IDS` | Whitelist (if set, only these eligible) | - |

#### Text / LLM

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_LLM_PROVIDER` | LLM provider (`openai` or `anthropic`) | `openai` |
| `DEFAULT_LLM_MODEL` | Default LLM model name | `gpt-4o` |
| `AVAILABLE_MODELS` | Comma-separated `provider/model` list | - |
| `TEXT_TIMEOUT` | Text generation timeout (seconds) | `180` |
| `GPT2_RESPONSE_TIMEOUT` | GPT-2 timeout (seconds) | `300` |
| `LLM_WHITELIST_CHANNELS` | Comma-separated channel IDs | - |
| `LLM_HISTORY_LIMIT` | Max conversation turns per channel | `50` |
| `LLM_MAX_TURNS_SENT` | Max turns sent to LLM | `8` |
| `SYSTEM_PROMPT_FILE` | Path to system prompt file | `config/system_prompt.txt` |

#### API Keys

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `REPLICATE_API_TOKEN` | Replicate API token | - |

#### Image Generation

| Variable | Description | Default |
|----------|-------------|---------|
| `STABLE_DIFFUSION_CHANNEL_IDS` | Comma-separated channel IDs | - |
| `EDIT_CHANNEL_IDS` | Channels allowing `edit:` prefix | - |
| `STABLE_DIFFUSION_PRIVACY` | Disable content logging (`1` = private) | `0` |
| `STABLE_DIFFUSION_ALLOW_DMS` | Allow image gen in DMs | `1` |
| `IMAGE_TIMEOUT` | Image generation timeout (seconds) | `30` |
| `IMAGE_OUTPUT_DIR` | Output directory for images | `/app/generated` |

#### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `PROMPT_DB_PATH` | SQLite database path | `prompts.db` |

#### Webhooks

| Variable | Description | Default |
|----------|-------------|---------|
| `WEBHOOK_URL_<CHANNEL_ID>` | Per-channel webhook URLs for personality system | - |

#### Admin

| Variable | Description | Default |
|----------|-------------|---------|
| `ADMIN_USER_IDS` | Comma-separated admin user IDs | - |
| `RESET_COMMAND` | Custom container restart command | - |

#### Cost Management

| Variable | Description | Default |
|----------|-------------|---------|
| `DAILY_FREE_BUDGET` | Daily free budget in dollars | `0.50` |

### Image Generation Config (`image_gen_config.json`)

The image generation configuration file defines:

- **Global Settings**:
  - `daily_free_budget`: Daily free budget for all users
  - `default_price_per_image`: Default price if not specified in prefix

- **Prefix Configuration**:
  Each prefix maps to a generator specification:
  ```json
  {
    "prefix": {
      "api": "replicate",
      "model": "black-forest-labs/flux-schnell",
      "mode": "generate",  // or "edit", "outpaint"
      "price_per_image": 0.02
    }
  }
  ```

**Example Configuration**:
```json
{
  "daily_free_budget": 1.0,
  "default_price_per_image": 0.03,
  "!": {
    "api": "replicate",
    "model": "black-forest-labs/flux-schnell",
    "price_per_image": 0.02
  },
  "$": {
    "api": "replicate",
    "model": "google/imagen-4-fast",
    "price_per_image": 0.03
  },
  "edit:": {
    "api": "replicate",
    "model": "google/nano-banana",
    "mode": "edit",
    "price_per_image": 0.04
  },
  "zoom out": {
    "api": "replicate",
    "model": "black-forest-labs/flux-fill-dev",
    "mode": "outpaint",
    "price_per_image": 0.05
  }
}
```

## Architecture

### Service Architecture

The bot runs as four Docker containers orchestrated by `docker-compose.yml`:

```
┌─────────────────────────────────────────────────────────────┐
│                         docker-compose                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     bot      │  │ celery_text  │  │ celery_image │      │
│  │              │  │              │  │              │      │
│  │ Discord.py   │  │  Queue: text │  │ Queue: image │      │
│  │ Main Bot     │  │  CPU-only    │  │ GPU-required │      │
│  │ Watchfiles   │  │  Concurrency │  │ Concurrency  │      │
│  │ Auto-reload  │  │      = 1     │  │      = 2     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                  │
│                  ┌────────▼────────┐                        │
│                  │     redis       │                        │
│                  │                 │                        │
│                  │  Message Broker │                        │
│                  │  Result Backend │                        │
│                  └─────────────────┘                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Shared Volumes:
- ./:/app (Code, mounted for development)
- ./generated:/app/generated (Generated images and state)
- prompt_db_data:/data (Persistent SQLite database)
- /var/run/docker.sock:/var/run/docker.sock (Docker control for bot)
```

**Key Architectural Decisions**:

1. **Separate Workers**: Text and image generation run in isolated workers to prevent GPU contention
2. **Queue-Based Routing**: Celery routes tasks by queue (`text` vs `image`) for resource isolation
3. **GPU Isolation**: Only `celery_image` requests GPU resources via Docker Compose `deploy.resources`
4. **Shared State**: All services share the same codebase and database via Docker volumes

### Task Processing Flow

```
Discord Message
      │
      ▼
Discord.py Listener (bot container)
      │
      ├─→ Image Generation Request
      │         │
      │         ▼
      │   tasks.generate_image.apply_async(queue="image")
      │         │
      │         ▼
      │   Celery Worker (celery_image, GPU)
      │         │
      │         ├─→ get_image_generator(api, model)
      │         ├─→ generator.generate(prompt, **kwargs)
      │         └─→ Save to disk → Return path
      │
      └─→ LLM Chat Request
                │
                ▼
          tasks.generate_llm_chat_response.apply_async(queue="text")
                │
                ▼
          Celery Worker (celery_text, CPU-only)
                │
                ├─→ get_text_generator(api, model)
                ├─→ _build_messages_for_generator(conversation)
                └─→ generator.generate(messages) → Return text
```

### Generator Pattern

Both image and text generation use a factory pattern for API abstraction:

**Image Generators** (`src/hollingsbot/image_generators/`):
- Base class: `ImageGeneratorAPI`
- Factory: `get_image_generator(api: str, model: str) -> ImageGeneratorAPI`
- Implementations:
  - `ReplicateImageGenerator`: Replicate API models
  - `SvgGPTImageGenerator`: SVG generation via LLM

**Text Generators** (`src/hollingsbot/text_generators/`):
- Base class: `TextGeneratorAPI`
- Factory: `get_text_generator(api: str, model: str) -> TextGeneratorAPI`
- Implementations:
  - `HuggingFaceGenerator`: Local transformers models
  - `AnthropicGenerator`: Claude API
  - `OpenAIGenerator`: ChatGPT API

### Database Schema

SQLite database (`prompts.db`) with tables:

- **prompts**: Image generation tracking
  - `id`, `prompt`, `user_id`, `api`, `model`, `status`, `timestamp`
- **starboard_posts**: Starboard message tracking
  - Original and starboard message IDs, reactor info, attachments
- **user_budgets**: Daily free budget tracking
  - `user_id`, `date`, `amount_used`, `generation_count`
- **user_credits**: Credit balance tracking
  - `user_id`, `balance`, `last_updated`
- **credit_transactions**: Credit transaction log
  - `user_id`, `amount`, `transaction_type`, `timestamp`

### State Persistence

- **LLM Chat State** (`generated/llm_chat_new_state.json`):
  - System prompts (per-guild override or global)
  - Model preferences (per-user, per-guild)
- **Notebook State** (`generated/notebook_state.json`):
  - Python notebook environment for tool execution
- **Conversation History**: In-memory only (cleared on bot restart)

### Command Prefix System

Dynamic prefixes prevent conflicts between image generation and commands:

- **Mention prefix**: Always enabled (`@bot command`)
- **Bang prefix (`!`)**: Disabled in image-gen channels to allow prompts like `!a cat`
- **Image-gen cogs**: Listen on `on_message` and check for configured prefixes directly

### Cost Tracking System

```
User Request
     │
     ▼
CostTracker.can_afford(user_id, cost)
     │
     ├─→ Check daily free budget
     │   ├─→ Has remaining free budget? → Approve
     │   └─→ Free budget exhausted? → Check credit balance
     │
     ├─→ Check credit balance
     │   ├─→ Has sufficient credits? → Approve
     │   └─→ Insufficient credits? → Deny
     │
     └─→ [Generation completes successfully]
           │
           ▼
         CostTracker.deduct_cost(user_id, cost)
           │
           ├─→ Deduct from free budget (if available)
           └─→ Deduct from credits (if free exhausted)
```

**Key Features**:
- Daily free budgets reset at midnight UTC
- Credit balance persists indefinitely
- Admins can grant/deduct credits via `!grant`
- Users check status via `!usage` or `!balance`

## Troubleshooting

### Bot Not Starting

```bash
# Check logs
docker-compose logs bot

# Common issues:
# - Invalid DISCORD_TOKEN
# - Missing API keys
# - Redis connection failure
```

### Image Generation Failing

```bash
# Check GPU worker logs
docker-compose logs celery_image

# Common issues:
# - NVIDIA Container Toolkit not installed
# - Invalid REPLICATE_API_TOKEN
# - GPU out of memory (reduce celery_image concurrency)
```

### LLM Chat Not Responding

```bash
# Check text worker logs
docker-compose logs celery_text

# Common issues:
# - Channel not in LLM_WHITELIST_CHANNELS
# - Invalid API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
# - Timeout exceeded (increase TEXT_TIMEOUT)
```

### Database Corruption

```bash
# Backup and rebuild database
docker cp hollingsbot3-bot-1:/data/prompts.db ./prompts_backup.db
docker exec hollingsbot3-bot-1 rm /data/prompts.db
docker-compose restart bot
```

## License

[Specify your license here]

## Contributing

[Add contribution guidelines here]

## Credits

Built with:
- [discord.py](https://github.com/Rapptz/discord.py) - Discord API wrapper
- [Celery](https://docs.celeryproject.org/) - Distributed task queue
- [Replicate](https://replicate.com/) - Image generation API
- [Anthropic Claude](https://www.anthropic.com/) - LLM API
- [OpenAI GPT](https://openai.com/) - LLM API
