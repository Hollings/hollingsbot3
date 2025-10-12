# Claude Code Development Notes

## Important Reminders

### Restarting the Bot
**The bot must be restarted after every code change.**

Use the following command to restart:
```bash
docker-compose down && docker-compose up -d
```

This ensures that code changes are picked up by the running container.
