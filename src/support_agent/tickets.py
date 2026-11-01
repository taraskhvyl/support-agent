"""SQLite ticket storage. Every query is parameterized; tickets are scoped to a session."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import NamedTuple

from . import config

_SCHEMA = """CREATE TABLE IF NOT EXISTS tickets(
    id INTEGER PRIMARY KEY, session TEXT, subject TEXT, description TEXT, status TEXT DEFAULT 'open')"""


class Ticket(NamedTuple):
    id: int
    subject: str
    description: str
    status: str


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(config.DB_PATH)) as db:
        db.execute(_SCHEMA)
        with db:  # commit on success, roll back on error
            yield db


def count(session: str) -> int:
    with _connect() as db:
        return db.execute("SELECT COUNT(*) FROM tickets WHERE session=?", (session,)).fetchone()[0]


def create(session: str, subject: str, description: str) -> int:
    with _connect() as db:
        cur = db.execute(
            "INSERT INTO tickets(session, subject, description) VALUES(?, ?, ?)",
            (session, subject, description),
        )
        return cur.lastrowid


def get(session: str, ticket_id: int) -> Ticket | None:
    with _connect() as db:
        row = db.execute(
            "SELECT id, subject, description, status FROM tickets WHERE id=? AND session=?",
            (ticket_id, session),
        ).fetchone()
    return Ticket(*row) if row else None
