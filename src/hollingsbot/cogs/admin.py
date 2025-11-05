"""Admin cog for operational commands like container restart and credit management."""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands

from hollingsbot.cost_tracking import CostTracker
from hollingsbot.settings import get_admin_user_ids

_LOG = logging.getLogger(__name__)

# Constants for compose file discovery
_COMPOSE_FILE_NAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)
_REPO_ROOT_OFFSET = 3  # src/hollingsbot/cogs/ -> repo root


class Admin(commands.Cog):
    """Operational commands anyone can run, plus admin-only commands for credit management."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

        # Initialize cost tracker for admin commands
        db_path = os.getenv("PROMPT_DB_PATH", "prompts.db")
        daily_free_budget = float(os.getenv("DAILY_FREE_BUDGET", "0.50"))
        self._cost_tracker = CostTracker(db_path, daily_free_budget)

        # Load admin user IDs
        self._admin_ids = get_admin_user_ids()
        _LOG.info(f"Admin cog initialized with {len(self._admin_ids)} admin(s)")

    @commands.command(name="reset")
    async def reset(self, ctx: commands.Context) -> None:
        """Restart the project Docker containers (or just the bot if needed)."""
        await ctx.send("Restarting containers… I may go offline briefly.")
        # Run restart asynchronously to return control to discord.py
        asyncio.create_task(self._restart_containers(ctx))

    async def _restart_containers(self, ctx: commands.Context) -> None:
        """
        Attempt to restart Docker containers; fallback to bot process restart.

        Tries in order:
        1. Custom override command (RESET_COMMAND env var)
        2. Modern Docker CLI (docker compose)
        3. Legacy Docker Compose (docker-compose)
        4. Bot process restart only
        """
        # Try custom override command first
        if await self._try_override_command():
            _LOG.info("Container restart succeeded via override command")
            return

        # Try Docker Compose restart
        compose_file = self._find_compose_file()
        if await self._try_docker_compose_restart(compose_file):
            _LOG.info("Container restart succeeded via Docker Compose")
            return

        # Fallback to bot process restart
        _LOG.warning("Docker not accessible; falling back to bot process restart")
        await self._fallback_to_bot_restart(ctx)

    async def _try_override_command(self) -> bool:
        """
        Execute custom restart command from RESET_COMMAND env var if set.

        Returns:
            True if command succeeded, False otherwise
        """
        override = os.getenv("RESET_COMMAND", "").strip()
        if not override:
            return False

        _LOG.info(f"Attempting restart via override command: {override}")
        success = await self._run_shell_command(override)

        if not success:
            _LOG.warning("Override command failed; falling back to default behavior")

        return success

    async def _try_docker_compose_restart(self, compose_file: Optional[Path]) -> bool:
        """
        Attempt to restart containers using Docker Compose.

        Tries modern 'docker compose' first, then legacy 'docker-compose'.

        Args:
            compose_file: Path to compose file, or None to use current directory

        Returns:
            True if restart succeeded, False otherwise
        """
        # Try modern Docker CLI (v20+)
        docker_bin = shutil.which("docker")
        if docker_bin:
            _LOG.info(f"Attempting restart via 'docker compose' (found at {docker_bin})")
            if await self._execute_compose_restart(docker_bin, "compose", compose_file):
                return True

        # Try legacy docker-compose
        docker_compose_bin = shutil.which("docker-compose")
        if docker_compose_bin:
            _LOG.info(f"Attempting restart via 'docker-compose' (found at {docker_compose_bin})")
            if await self._execute_compose_restart(docker_compose_bin, None, compose_file):
                return True

        _LOG.warning("Neither 'docker' nor 'docker-compose' commands found in PATH")
        return False

    async def _execute_compose_restart(
        self,
        docker_bin: str,
        subcommand: Optional[str],
        compose_file: Optional[Path],
    ) -> bool:
        """
        Execute a Docker Compose restart command.

        Args:
            docker_bin: Path to docker binary
            subcommand: Subcommand to use (e.g., "compose" for modern Docker), or None for legacy
            compose_file: Path to compose file, or None to use current directory

        Returns:
            True if restart succeeded, False otherwise
        """
        cmd = [docker_bin]
        if subcommand:
            cmd.append(subcommand)

        if compose_file is not None:
            cmd.extend(["-f", str(compose_file)])
            cwd = compose_file.parent
        else:
            cwd = Path.cwd()

        cmd.append("restart")

        return await self._run_subprocess(cmd, cwd)

    async def _fallback_to_bot_restart(self, ctx: commands.Context) -> None:
        """
        Restart only the bot process as a last resort.

        If running under Docker Compose with a restart policy, the container
        will automatically restart.

        Args:
            ctx: Discord command context for sending notification
        """
        await ctx.send("Docker not accessible; restarting bot process only.")
        _LOG.info("Initiating bot process restart")

        try:
            await self.bot.close()
        except Exception as e:
            _LOG.error(f"Error closing bot gracefully: {e}")
        finally:
            _LOG.info("Exiting bot process")
            os._exit(0)

    async def _run_shell_command(self, command: str) -> bool:
        """
        Execute a shell command asynchronously.

        Args:
            command: Shell command string to execute

        Returns:
            True if command succeeded (exit code 0), False otherwise
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            exit_code = proc.returncode or 0

            if exit_code != 0:
                _LOG.warning(
                    f"Shell command failed with exit code {exit_code}. "
                    f"stderr: {stderr.decode(errors='ignore')[:200]}"
                )

            return exit_code == 0

        except Exception as e:
            _LOG.error(f"Failed to execute shell command: {e}")
            return False

    async def _run_subprocess(
        self,
        cmd: list[str],
        cwd: Optional[Path] = None,
    ) -> bool:
        """
        Execute a subprocess command asynchronously.

        Args:
            cmd: Command and arguments as a list
            cwd: Working directory for the command, or None for current directory

        Returns:
            True if command succeeded (exit code 0), False otherwise
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd) if cwd else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            exit_code = proc.returncode or 0

            if exit_code != 0:
                _LOG.warning(
                    f"Command {cmd[0]} failed with exit code {exit_code}. "
                    f"stderr: {stderr.decode(errors='ignore')[:200]}"
                )

            return exit_code == 0

        except Exception as e:
            _LOG.error(f"Failed to execute subprocess {cmd}: {e}")
            return False

    def _find_compose_file(self) -> Optional[Path]:
        """
        Search for a Docker Compose file in the project directory tree.

        Searches upward from the current working directory and the module's
        repository root for any standard compose file name.

        Returns:
            Path to the first compose file found, or None if not found
        """
        candidates = self._get_search_base_paths()
        seen = set()

        for base in candidates:
            for directory in [base, *base.parents]:
                if directory in seen:
                    continue
                seen.add(directory)

                compose_file = self._check_compose_files(directory)
                if compose_file:
                    _LOG.debug(f"Found compose file at {compose_file}")
                    return compose_file

        _LOG.debug("No compose file found in project tree")
        return None

    def _get_search_base_paths(self) -> list[Path]:
        """
        Get base paths to start searching for compose files.

        Returns:
            List of paths to search from (current working directory and repo root)
        """
        candidates = []

        # Add current working directory if accessible
        try:
            candidates.append(Path.cwd())
        except Exception as e:
            _LOG.debug(f"Could not access current working directory: {e}")

        # Add repository root (heuristic: go up from src/hollingsbot/cogs/)
        try:
            repo_root = Path(__file__).resolve().parents[_REPO_ROOT_OFFSET]
            candidates.append(repo_root)
        except Exception as e:
            _LOG.debug(f"Could not determine repository root: {e}")

        return candidates

    def _check_compose_files(self, directory: Path) -> Optional[Path]:
        """
        Check if any standard compose file exists in the given directory.

        Args:
            directory: Directory to check for compose files

        Returns:
            Path to the first compose file found, or None if none exist
        """
        for name in _COMPOSE_FILE_NAMES:
            compose_file = directory / name
            if compose_file.exists() and compose_file.is_file():
                return compose_file
        return None

    def _is_admin(self, user_id: int) -> bool:
        """Check if a user ID is in the admin list.

        Args:
            user_id: Discord user ID to check

        Returns:
            True if the user is an admin
        """
        return user_id in self._admin_ids

    @commands.command(name="grant")
    async def grant_credits(self, ctx: commands.Context, user: discord.User, amount: float) -> None:
        """Grant credits to a user (admin only).

        Usage: !grant @user 10.00

        Args:
            user: User mention to grant credits to
            amount: Amount of credits to grant (can be negative to deduct)
        """
        if not self._is_admin(ctx.author.id):
            await ctx.send("❌ This command is restricted to admins.")
            return

        try:
            self._cost_tracker.grant_credits(user.id, amount)
            action = "granted" if amount >= 0 else "deducted"
            await ctx.send(f"✅ Successfully {action} ${abs(amount):.2f} {'to' if amount >= 0 else 'from'} {user.mention}.")
            _LOG.info(f"Admin {ctx.author.id} {action} ${abs(amount):.2f} {'to' if amount >= 0 else 'from'} user {user.id}")
        except Exception as exc:
            _LOG.exception(f"Failed to grant credits: {exc}")
            await ctx.send(f"❌ Failed to grant credits: {exc}")

    @commands.command(name="balance")
    async def check_balance(self, ctx: commands.Context, user: discord.User | None = None) -> None:
        """Check a user's balance and usage (admin only, or own balance).

        Usage: !balance @user

        Args:
            user: User mention to check (optional, defaults to self)
        """
        # If no user specified, check own balance
        if user is None:
            user = ctx.author
        elif not self._is_admin(ctx.author.id):
            # If checking someone else's balance, must be admin
            await ctx.send("❌ Checking other users' balances is restricted to admins.")
            return

        try:
            status = self._cost_tracker.get_user_status(user.id)

            free_used = status["free_budget_used"]
            free_total = status["free_budget_total"]
            credits_used_today = status["credits_used_today"]
            generation_count = status["generation_count"]
            credit_balance = status["credit_balance"]
            reset_time = status["reset_time"]

            lines = [
                f"**Balance for {user.mention}:**",
                f"",
                f"**Today's usage** (resets in {reset_time}):",
                f"  • Free budget: ${free_used:.2f} / ${free_total:.2f} used",
            ]

            if credits_used_today > 0:
                lines.append(f"  • Credits spent today: ${credits_used_today:.2f}")

            lines.append(f"  • Total generations: {generation_count}")
            lines.append(f"\n**Credit balance:** ${credit_balance:.2f}")

            message = "\n".join(lines)
            await ctx.send(message)

        except Exception as exc:
            _LOG.exception(f"Failed to get balance for user {user.id}: {exc}")
            await ctx.send(f"❌ Failed to retrieve balance: {exc}")

    @commands.command(name="set_price")
    async def set_price(self, ctx: commands.Context, prefix: str, price: float) -> None:
        """Set the price for a model prefix (admin only).

        Usage: !set_price ! 0.005

        Args:
            prefix: Model prefix (e.g., "!", "$", "edit:")
            price: New price per image in dollars
        """
        if not self._is_admin(ctx.author.id):
            await ctx.send("❌ This command is restricted to admins.")
            return

        try:
            # Load config
            config_path = Path(__file__).resolve().parents[1] / "image_gen_config.json"
            with open(config_path, "r", encoding="utf8") as f:
                config = json.load(f)

            # Update price
            if prefix not in config:
                await ctx.send(f"❌ Prefix `{prefix}` not found in configuration.")
                return

            if not isinstance(config[prefix], dict):
                await ctx.send(f"❌ Prefix `{prefix}` is not a model configuration.")
                return

            old_price = config[prefix].get("price_per_image", "not set")
            config[prefix]["price_per_image"] = price

            # Save config
            with open(config_path, "w", encoding="utf8") as f:
                json.dump(config, f, indent=2)

            await ctx.send(
                f"✅ Updated price for `{prefix}` from ${old_price} to ${price:.3f} per image.\n"
                f"Note: Config will reload on next generation or bot restart."
            )
            _LOG.info(f"Admin {ctx.author.id} updated price for '{prefix}' to ${price:.3f}")

        except Exception as exc:
            _LOG.exception(f"Failed to set price: {exc}")
            await ctx.send(f"❌ Failed to set price: {exc}")

    @commands.command(name="set_budget")
    async def set_budget(self, ctx: commands.Context, budget: float) -> None:
        """Set the global daily free budget (admin only).

        Usage: !set_budget 1.00

        Args:
            budget: New daily free budget amount in dollars
        """
        if not self._is_admin(ctx.author.id):
            await ctx.send("❌ This command is restricted to admins.")
            return

        try:
            # Load config
            config_path = Path(__file__).resolve().parents[1] / "image_gen_config.json"
            with open(config_path, "r", encoding="utf8") as f:
                config = json.load(f)

            old_budget = config.get("daily_free_budget", "not set")
            config["daily_free_budget"] = budget

            # Save config
            with open(config_path, "w", encoding="utf8") as f:
                json.dump(config, f, indent=2)

            # Update cost tracker
            self._cost_tracker.daily_free_budget = budget

            await ctx.send(
                f"✅ Updated daily free budget from ${old_budget} to ${budget:.2f}.\n"
                f"Note: Config will reload on next generation or bot restart."
            )
            _LOG.info(f"Admin {ctx.author.id} updated daily free budget to ${budget:.2f}")

        except Exception as exc:
            _LOG.exception(f"Failed to set budget: {exc}")
            await ctx.send(f"❌ Failed to set budget: {exc}")


async def setup(bot: commands.Bot) -> None:
    """Load the Admin cog."""
    await bot.add_cog(Admin(bot))
