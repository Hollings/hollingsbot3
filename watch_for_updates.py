#!/usr/bin/env python3
"""Poll git for updates and rebuild containers when new commits appear."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

INTERVAL = int(os.environ.get("UPDATE_INTERVAL", "60"))
REPO_DIR = Path(__file__).resolve().parent


def run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"Command {' '.join(cmd)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> None:
    while True:
        # Fetch remote updates
        run(["git", "remote", "update"])
        local = run(["git", "rev-parse", "@"])
        remote = run(["git", "rev-parse", "@{u}"])
        base = run(["git", "merge-base", "@", "@{u}"])

        if local != remote and local == base:
            print("New commits found. Pulling and rebuilding containers...")
            run(["git", "pull", "--ff-only"])
            subprocess.run(["docker-compose", "up", "--build", "-d"], cwd=REPO_DIR)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
