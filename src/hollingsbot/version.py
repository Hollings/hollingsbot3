"""Running-version info: which commit is live and when this process started.

The production (ARM) image has no ``git`` binary, but the repo - including
``.git`` - is bind-mounted at /app, so the commit SHA is read straight from the
git metadata files. ``git`` is only used, when available, to flag a dirty tree.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

__all__ = ["START_TIME", "presence_text", "running_sha"]

START_TIME = time.time()

_SHORT_SHA_LEN = 7
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_git_dir(start: Path) -> Path | None:
    """Locate the git directory for the checkout containing *start*."""
    for parent in (start, *start.parents):
        dot_git = parent / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            # Worktree/submodule: a file containing "gitdir: <path>"
            content = dot_git.read_text("utf8").strip()
            if content.startswith("gitdir:"):
                target = Path(content.split(":", 1)[1].strip())
                return target if target.is_absolute() else (parent / target).resolve()
    return None


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    """Resolve a ref like ``refs/heads/main`` via loose ref files, then packed-refs."""
    # Worktrees keep HEAD in their own git dir but refs in the common dir.
    common = git_dir
    commondir_file = git_dir / "commondir"
    if commondir_file.is_file():
        common = (git_dir / commondir_file.read_text("utf8").strip()).resolve()

    for base in (git_dir, common):
        loose = base / ref
        if loose.is_file():
            return loose.read_text("utf8").strip() or None

    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text("utf8").splitlines():
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha
    return None


def _head_sha(start: Path) -> str | None:
    """Full SHA of HEAD for the checkout containing *start*, without the git binary."""
    try:
        git_dir = _find_git_dir(start)
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text("utf8").strip()
        if head.startswith("ref:"):
            return _resolve_ref(git_dir, head.split(":", 1)[1].strip())
        return head or None  # detached HEAD
    except OSError:
        return None


def _is_dirty(cwd: Path) -> bool:
    """True if the working tree has uncommitted changes. False when git can't tell us."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def running_sha(start: Path | None = None) -> str:
    """Short SHA of the running commit; a trailing ``+`` marks a dirty tree."""
    root = start or _REPO_ROOT
    sha = _head_sha(root)
    if not sha:
        return "unknown"
    short = sha[:_SHORT_SHA_LEN]
    return f"{short}+" if _is_dirty(root) else short


def presence_text() -> str:
    """Presence string ``v<sha> · up <start time>``."""
    started = time.strftime("%b %d %H:%M", time.localtime(START_TIME))
    return f"v{running_sha()} · up {started}"
