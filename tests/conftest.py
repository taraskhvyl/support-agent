import pytest

from support_agent import config


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "tickets.db")
    monkeypatch.setenv("SESSION_ID", "s1")
    monkeypatch.setenv("APPROVAL_SECRET", "k")
