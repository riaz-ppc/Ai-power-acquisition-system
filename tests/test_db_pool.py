import psycopg2
import pytest

from src import db


class FakeConn:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakePool:
    def __init__(self):
        self.conn = FakeConn()
        self.returned = []  # (conn, close) per putconn

    def getconn(self):
        return self.conn

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))


@pytest.fixture
def pool(monkeypatch):
    fake = FakePool()
    monkeypatch.setattr(db, "_get_pool", lambda: fake)
    monkeypatch.setattr(db, "DATABASE_URL", "postgresql://fake")
    return fake


def test_success_commits_and_returns_connection_to_pool(pool):
    with db.get_conn():
        pass
    assert pool.conn.commits == 1
    assert pool.returned == [(pool.conn, False)]


def test_app_error_rolls_back_and_keeps_connection(pool):
    with pytest.raises(ValueError):
        with db.get_conn():
            raise ValueError("bad query")
    assert pool.conn.rollbacks == 1
    assert pool.returned == [(pool.conn, False)]


def test_dropped_connection_is_evicted_not_reused(pool):
    with pytest.raises(psycopg2.OperationalError):
        with db.get_conn():
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
    assert pool.returned == [(pool.conn, True)]
    assert pool.conn.rollbacks == 0  # no rollback on a connection the server already dropped


def test_missing_database_url_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        with db.get_conn():
            pass
