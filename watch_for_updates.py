#!/usr/bin/env python3
"""Poll git for updates and rebuild containers when new commits appear."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

INTERVAL = int(os.environ.get("UPDATE_INTERVAL", "60"))
REPO_DIR = Path(__file__).resolve().parent

# Track only the current branch (defaults to "main")
BRANCH = os.environ.get("WATCH_BRANCH")
if BRANCH is None:
    BRANCH = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    ).stdout.strip() or "main"


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
        # Fetch updates for the branch only
        run(["git", "fetch", "origin", BRANCH])
        local = run(["git", "rev-parse", BRANCH])
        remote = run(["git", "rev-parse", f"origin/{BRANCH}"])
        base = run(["git", "merge-base", BRANCH, f"origin/{BRANCH}"])

        if local != remote and local == base:
            print("New commits found. Pulling and rebuilding containers...")
            run(["git", "pull", "origin", BRANCH, "--ff-only"])
            subprocess.run([
                "docker-compose",
                "up",
                "--build",
                "-d",
            ], cwd=REPO_DIR)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
