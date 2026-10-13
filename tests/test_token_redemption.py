"""Token exchanges cannot overspend or lose tokens on a credit write failure."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from hollingsbot import prompt_db


@pytest.fixture
def db(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_db, "DB_PATH", tmp_path / "tokens.db")
    prompt_db.init_db()
    with prompt_db._connect() as conn:
        conn.execute("INSERT INTO user_tokens (user_id, tokens) VALUES (1, 47)")


def test_exchange_credits_and_tokens_together(db):
    assert prompt_db.redeem_user_tokens(1, 47, 0.37) == (True, 0)
    with prompt_db._connect() as conn:
        assert conn.execute("SELECT balance FROM user_credits WHERE user_id = 1").fetchone()[0] == 0.37


def test_credit_failure_rolls_back_tokens(db):
    with prompt_db._connect() as conn:
        conn.execute("CREATE TRIGGER fail_credit BEFORE INSERT ON user_credits BEGIN SELECT RAISE(ABORT, 'fail'); END")
    with pytest.raises(Exception, match="fail"):
        prompt_db.redeem_user_tokens(1, 47, 0.37)
    assert prompt_db.get_user_token_balance(1) == 47


@pytest.mark.parametrize("redeem", [False, True])
def test_concurrent_spending_only_succeeds_once(db, redeem):
    def spend(_):
        if redeem:
            return prompt_db.redeem_user_tokens(1, 47, 0.37)
        return prompt_db.deduct_user_tokens(1, 47)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(spend, range(4)))
    assert sum(success for success, _ in results) == 1
    assert prompt_db.get_user_token_balance(1) == 0


@pytest.mark.parametrize("amount", [-1, 0, 1.5, True])
def test_invalid_token_amounts_leave_balance_unchanged(db, amount):
    with pytest.raises(ValueError):
        prompt_db.deduct_user_tokens(1, amount)
    with pytest.raises(ValueError):
        prompt_db.redeem_user_tokens(1, amount, 0.37)
    assert prompt_db.get_user_token_balance(1) == 47
