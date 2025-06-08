# hollingsbot3

This is a minimal Discord bot project using [discord.py](https://discordpy.readthedocs.io/).
It is set up to run inside Docker using docker-compose. Image generation tasks
are handled by a separate Celery worker communicating through Redis.

## Requirements
- Docker
- Docker Compose

## Setup
1. Create a `.env` file (already provided) and fill in the required tokens and IDs.
   The `GITHUB_TOKEN` must include the `repo` scope so the bot can merge pull requests.
2. Build and start the bot and worker:
   ```bash
   docker-compose up --build
   ```

This command launches the Discord bot, a Celery worker, and a Redis instance
used for task queueing.

### Celery worker reloading

Celery 5 removed the old `--autoreload` flag. The `docker-compose.yml` file now
starts the worker via [watchfiles](https://github.com/samuelcolvin/watchfiles),
which monitors Python files and restarts the worker when they change.

With local sources mounted into the container you can simply run:

```bash
docker-compose up
```

Whenever you edit `tasks.py` or other modules, the worker process will restart
and pick up the new code automatically. You can also invoke the command
directly:

```bash
watchfiles --filter python "celery -A tasks worker --loglevel=info" /app
```

If you prefer not to use auto reload you can still rebuild the containers
manually or run `watch_for_updates.py`.

### GPU acceleration

The Docker setup now uses a CUDA-enabled PyTorch image. Ensure your host has the
NVIDIA Container Toolkit installed so containers can access the GPU. The
`docker-compose.yml` file reserves a GPU for both the bot and the Celery worker.

### Automatic updates

Run `watch_for_updates.py` to poll the repository for new commits and
automatically rebuild the containers when updates are available:

```bash
python watch_for_updates.py
```

The script checks for changes every 60 seconds by default. Set the
`UPDATE_INTERVAL` environment variable to adjust the polling interval. By
default the script monitors the branch that is currently checked out. Override
this by setting the `WATCH_BRANCH` environment variable, e.g. `WATCH_BRANCH=main`.

The bot uses a modular Cog system located in the `cogs/` directory. A sample `ping` command is provided.

### Image generation configuration

Image prompts are routed based on the prefix defined in `image_gen_config.json`.
Each prefix maps to an API provider and model name:

```json
{
  "!": {"api": "replicate", "model": "black-forest-labs/flux-schnell"}
}
```

Additional prefixes can be added and will be handled by the configured image
generator class.

### Pull request notifications

The `PRManager` cog posts to a Discord webhook when pull requests are opened and
when they are merged via the bot. Set the `WEBHOOK_URL` environment variable to
override the default webhook, which currently points to:

```
https://discord.com/api/webhooks/1380252805494738974/wif1p4iK8_hZq41JwNCmHRcFj6E9cxyet1fNlTiLt-nN9dc2qyxclJcYZhoUnfRwG5p5
```
Persistent tracking of notified pull requests is stored in the SQLite database
(`prompts.db` by default). Set `PROMPT_DB_PATH` to change where this state is
kept.

### Starboard

Set `ENABLE_STARBOARD=1` and `STARBOARD_CHANNEL_ID` to repost bot messages to
another channel when they receive reactions. The optional
`STARBOARD_IGNORE_CHANNELS` variable accepts a comma-separated list of channel
IDs to exclude from reposting.

### Running tests

To run the test suite outside of Docker you must install the Python
dependencies locally. Failing to install them results in `ModuleNotFoundError`
for packages such as `discord.py`.

```bash
pip install -r requirements.txt
pytest
```
