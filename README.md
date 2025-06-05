# hollingsbot3

This is a minimal Discord bot project using [discord.py](https://discordpy.readthedocs.io/).
It is set up to run inside Docker using docker-compose. Image generation tasks
are handled by a separate Celery worker communicating through Redis.

## Requirements
- Docker
- Docker Compose

## Setup
1. Create a `.env` file (already provided) and fill in the required tokens and IDs.
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
`UPDATE_INTERVAL` environment variable to adjust the polling interval.

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
