import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Optional

from discord.ext import commands


class Admin(commands.Cog):
    """Operational commands anyone can run."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command(name="reset")
    async def reset(self, ctx: commands.Context):
        """Restart the project Docker containers (or just the bot if needed)."""
        # Acknowledge quickly so the message gets out before restart
        await ctx.send("Restarting containers… I may go offline briefly.")
        # Run restart asynchronously so we can return control to discord.py
        asyncio.create_task(self._restart_containers(ctx))

    async def _restart_containers(self, ctx: commands.Context) -> None:
        """Try docker compose restart; fallback to restarting this bot only."""
        # Optional override via env
        override = os.getenv("RESET_COMMAND", "").strip()
        compose_file = self._find_compose_file()

        async def _run(cmd: List[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd) if cwd else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_b, err_b = await proc.communicate()
            return proc.returncode or 0, out_b.decode(errors="ignore"), err_b.decode(errors="ignore")

        if override:
            # Run user-specified command string through the shell for flexibility
            proc = await asyncio.create_subprocess_shell(
                override,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_b, err_b = await proc.communicate()
            code = proc.returncode or 0
            if code == 0:
                return
            # If override fails, fall back to default behavior

        # Prefer Docker CLI v20+: `docker compose`
        docker_bin = shutil.which("docker")
        if docker_bin:
            cmd = [docker_bin, "compose"]
            if compose_file is not None:
                cmd += ["-f", str(compose_file)]
                cwd = compose_file.parent
            else:
                cwd = Path.cwd()
            cmd += ["restart"]
            code, _, _ = await _run(cmd, cwd=cwd)
            if code == 0:
                return

        # Fallback to legacy docker-compose
        docker_compose_bin = shutil.which("docker-compose")
        if docker_compose_bin:
            cmd = [docker_compose_bin]
            if compose_file is not None:
                cmd += ["-f", str(compose_file)]
                cwd = compose_file.parent
            else:
                cwd = Path.cwd()
            cmd += ["restart"]
            code, _, _ = await _run(cmd, cwd=cwd)
            if code == 0:
                return

        # If we get here, we couldn't control Docker from this process.
        # Fall back to restarting just the bot process (Compose restart policy will handle it if applicable).
        await ctx.send("Docker not accessible; restarting bot process only.")
        # Best-effort process restart; if running under Compose with a restart policy, container will come back.
        try:
            await self.bot.close()
        finally:
            os._exit(0)

    def _find_compose_file(self) -> Optional[Path]:
        """Search upwards for a docker-compose.yml file starting from CWD and module dir."""
        candidates = []
        try:
            candidates.append(Path.cwd())
        except Exception:  # noqa: BLE001
            pass
        candidates.append(Path(__file__).resolve().parents[3])  # repo root heuristic: src/hollingsbot/cogs/ -> repo root

        seen = set()
        for base in candidates:
            for p in [base, *base.parents]:
                if p in seen:
                    continue
                seen.add(p)
                for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
                    f = p / name
                    if f.exists() and f.is_file():
                        return f
        return None


async def setup(bot: commands.Bot):
    await bot.add_cog(Admin(bot))
