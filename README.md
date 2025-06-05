# hollingsbot3

This is a minimal Discord bot project using [discord.py](https://discordpy.readthedocs.io/).
It is set up to run inside Docker using docker-compose.

## Requirements
- Docker
- Docker Compose

## Setup
1. Create a `.env` file (already provided) and fill in the required tokens and IDs.
2. Build and start the bot:
   ```bash
   docker-compose up --build
   ```

The bot uses a modular Cog system located in the `cogs/` directory. A sample `ping` command is provided.
