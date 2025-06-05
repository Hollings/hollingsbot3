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

Set `STARBOARD_CHANNEL_ID` to the destination channel for reposts. The optional
`STARBOARD_IGNORE_CHANNELS` variable accepts a comma-separated list of channel
IDs to exclude from reposting.
