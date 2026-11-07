"""Exercise deployment recovery with real Git and a fake Docker CLI."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_TEST_BASH = os.getenv("HOLLINGSBOT_TEST_BASH")
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" and not _TEST_BASH, reason="deployment script uses the Linux Bash toolchain"
)
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "auto_deploy.sh"


def test_failed_deploy_keeps_running_checkout_and_retries_unready_sha(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    remote = tmp_path / "remote.git"

    def git(*args):
        return subprocess.check_output(["git", *map(str, args)], cwd=repo, text=True).strip()

    git("init", "--bare", remote)
    git("init", "-b", "main")
    git("config", "user.name", "Deployment test")
    git("config", "user.email", "deployment@test.invalid")
    git("config", "core.autocrlf", "false")
    git("remote", "add", "origin", remote)
    app = repo / "app.txt"
    app.write_text("old\n", encoding="utf-8")
    compose = repo / "docker-compose.orange-pi.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    git("add", "app.txt", compose)
    git("commit", "-m", "old")
    old = git("rev-parse", "HEAD")
    app.write_text("new\n", encoding="utf-8")
    git("commit", "-am", "new")
    new = git("rev-parse", "HEAD")
    git("push", "origin", "main")
    git("reset", "--hard", old)
    (repo / ".env").write_text("EXAMPLE=not-a-real-secret\n", encoding="utf-8")
    compose.write_text("services: {}\n# Pi-local override\n", encoding="utf-8")
    marker = repo / ".git" / "hollingsbot-healthy-sha"
    marker.write_text(old + "\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_bytes(
        b"""#!/usr/bin/env bash
set -eu
case "$*" in
  *"config --format json"*) echo '{"name":"existing-project"}' ;;
  *"build bot celery_text celery_image"*)
    [[ "$*" == *"--env-file "* && "$*" == *"--project-name existing-project"* ]]
    [[ ! -s .env ]]
    grep -q 'Pi-local override' docker-compose.orange-pi.yml
    printf 'build:%s:%s\\n' "$(cat app.txt)" "$(cat "$HOLLINGSBOT_DEPLOY_REPO/app.txt")" >>"$DOCKER_CALLS"
    [[ "$BUILD_FAIL" == 0 ]] ;;
  *"up -d"*) printf '%s\\n' "$*" >>"$DOCKER_CALLS" ;;
  *"redis-cli ping"*) echo PONG ;;
  *"ps -q"*) echo "${*: -1}" ;;
  inspect*) echo 2026-01-01T00:00:00Z ;;
  logs*) [[ "$READY" == 1 ]] && echo 'Bot is ready. Logged in as test; worker ready.' ;;
  *) exit 98 ;;
esac
"""
    )
    docker.chmod(0o755)
    if _TEST_BASH and sys.platform != "linux":
        flock = bin_dir / "flock"
        flock.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
        flock.chmod(0o755)
    calls = tmp_path / "docker-calls"

    def deploy(*, build_fail=0, ready=1):
        env = {
            **os.environ,
            "DOCKER_CALLS": calls.as_posix(),
            "BUILD_FAIL": str(build_fail),
            "READY": str(ready),
            "HOLLINGSBOT_DEPLOY_REPO": repo.as_posix(),
            "HOLLINGSBOT_DEPLOY_LOG": (tmp_path / "deploy.log").as_posix(),
            "HOLLINGSBOT_DEPLOY_READY_TIMEOUT": "0",
        }
        result = subprocess.run(
            [
                _TEST_BASH or "bash",
                "-c",
                'bin="$1"; if command -v cygpath >/dev/null; then bin=$(cygpath -u "$bin"); fi; '
                'export PATH="$bin:$PATH"; exec bash "$2"',
                "test",
                bin_dir.as_posix(),
                _SCRIPT.as_posix(),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == (1 if build_fail or not ready else 0), result.stdout + result.stderr

    deploy(build_fail=1)
    assert git("rev-parse", "HEAD") == old
    assert calls.read_text(encoding="utf-8").splitlines() == ["build:new:old"]

    deploy(ready=0)
    assert git("rev-parse", "HEAD") == new
    assert marker.read_text(encoding="utf-8").strip() == old
    assert "Pi-local override" in compose.read_text(encoding="utf-8")

    deploy()
    assert marker.read_text(encoding="utf-8").strip() == new
    trace = calls.read_text(encoding="utf-8")
    assert "build:new:new" in trace
    assert "--no-deps --force-recreate bot celery_text celery_image" in trace
    deploy()
    assert calls.read_text(encoding="utf-8") == trace
