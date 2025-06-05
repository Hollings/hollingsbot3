import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from prompt_db import init_db, update_pr_status, get_seen_prs


def test_pr_status(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "db.sqlite"))
    init_db()
    update_pr_status(1, "open")
    update_pr_status(2, "closed")
    assert get_seen_prs() == {1: "open", 2: "closed"}
    update_pr_status(1, "merged")
    assert get_seen_prs()[1] == "merged"
