"""Accounts, score tallies, and recorded human games.

Storage backends:
- **SQLite** (default, stdlib): ``big2/data/big2.db`` locally, or
  ``$BIG2_DB``.  If the location isn't writable (read-only serverless
  filesystems) it falls back to /tmp and flags itself non-persistent so
  the UI can warn that stats reset between deploys.
- **Postgres** when ``$DATABASE_URL`` is set (e.g. Vercel/Neon Postgres)
  and psycopg2 is installed — durable accounts on serverless.

Security model is deliberately simple for a hobby deployment: scrypt
password hashes with per-user salts, HMAC-signed bearer tokens keyed by
a server secret stored in the database.  No rate limiting, no email —
a username and password is the whole account, as requested.

Recorded games carry the full replay plus the time-stamped model lineup
(e.g. ``ppo@20260817-1755``), so scores stay attributable to specific
model versions and the replays double as training data:

    python -m big2.store --export replays.jsonl
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

DEFAULT_SQLITE = os.path.join(os.path.dirname(__file__), "data", "big2.db")

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY {autoinc},
        username TEXT UNIQUE NOT NULL,
        salt TEXT NOT NULL,
        pass_hash TEXT NOT NULL,
        created_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY {autoinc},
        user_id INTEGER NOT NULL,
        ts REAL NOT NULL,
        num_players INTEGER NOT NULL,
        lineup TEXT NOT NULL,
        rules TEXT NOT NULL,
        scoring TEXT NOT NULL,
        user_seat INTEGER NOT NULL,
        user_score INTEGER NOT NULL,
        won INTEGER NOT NULL,
        replay TEXT NOT NULL)""",
]


class Store:
    def __init__(self, url: Optional[str] = None):
        url = url or os.environ.get("DATABASE_URL") or ""
        self.persistent = True
        if url.startswith(("postgres://", "postgresql://")):
            import psycopg2  # optional dependency

            self._pg = True
            self._url = url
            self._connect = lambda: psycopg2.connect(url)
            self._ph = "%s"
        else:
            self._pg = False
            path = url or os.environ.get("BIG2_DB") or DEFAULT_SQLITE
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "a"):
                    pass
            except OSError:
                path = os.path.join("/tmp", "big2.db")
                self.persistent = False
            self._path = path
            self._connect = lambda: sqlite3.connect(self._path, timeout=10)
            self._ph = "?"
        self._init_schema()
        self._secret = self._get_secret()

    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        autoinc = "" if not self._pg else ""
        with self._connect() as con:
            cur = con.cursor()
            for stmt in _SCHEMA:
                sql = stmt.format(autoinc=autoinc)
                if self._pg:
                    sql = sql.replace(
                        "INTEGER PRIMARY KEY ", "BIGSERIAL PRIMARY KEY "
                    )
                cur.execute(sql)
            con.commit()

    def _get_secret(self) -> bytes:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"SELECT value FROM meta WHERE key = {self._ph}", ("secret",)
            )
            row = cur.fetchone()
            if row:
                return bytes.fromhex(row[0])
            secret = secrets.token_bytes(32)
            cur.execute(
                f"INSERT INTO meta (key, value) VALUES ({self._ph}, {self._ph})",
                ("secret", secret.hex()),
            )
            con.commit()
            return secret

    # ------------------------------------------------------------------
    # Accounts
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(password: str, salt: bytes) -> str:
        return hashlib.scrypt(
            password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32
        ).hex()

    def _token(self, username: str) -> str:
        sig = hmac.new(self._secret, username.encode(), "sha256").hexdigest()
        return f"{username}:{sig}"

    def auth(self, token: Optional[str]) -> Optional[Tuple[int, str]]:
        """token -> (user_id, username), or None."""
        if not token or ":" not in token:
            return None
        username, sig = token.rsplit(":", 1)
        expect = hmac.new(self._secret, username.encode(), "sha256").hexdigest()
        if not hmac.compare_digest(sig, expect):
            return None
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"SELECT id FROM users WHERE username = {self._ph}", (username,)
            )
            row = cur.fetchone()
        return (row[0], username) if row else None

    def register(self, username: str, password: str) -> str:
        username = username.strip()
        if not (2 <= len(username) <= 24) or not username.replace("_", "").isalnum():
            raise ValueError("username: 2-24 letters, digits, underscores")
        if len(password) < 4:
            raise ValueError("password must be at least 4 characters")
        salt = secrets.token_bytes(16)
        with self._connect() as con:
            cur = con.cursor()
            try:
                cur.execute(
                    f"INSERT INTO users (username, salt, pass_hash, created_at) "
                    f"VALUES ({self._ph}, {self._ph}, {self._ph}, {self._ph})",
                    (username, salt.hex(), self._hash(password, salt), time.time()),
                )
                con.commit()
            except (sqlite3.IntegrityError, Exception) as exc:
                if "unique" in str(exc).lower() or isinstance(
                    exc, sqlite3.IntegrityError
                ):
                    raise ValueError("username already taken")
                raise
        return self._token(username)

    def login(self, username: str, password: str) -> str:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"SELECT salt, pass_hash FROM users WHERE username = {self._ph}",
                (username.strip(),),
            )
            row = cur.fetchone()
        if not row or not hmac.compare_digest(
            row[1], self._hash(password, bytes.fromhex(row[0]))
        ):
            raise ValueError("wrong username or password")
        return self._token(username.strip())

    # ------------------------------------------------------------------
    # Games & stats
    # ------------------------------------------------------------------

    def record_game(self, user_id: int, payload: Dict) -> None:
        lineup = payload.get("lineup") or []
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"INSERT INTO games (user_id, ts, num_players, lineup, rules, "
                f"scoring, user_seat, user_score, won, replay) VALUES "
                f"({', '.join([self._ph] * 10)})",
                (
                    user_id,
                    time.time(),
                    int(payload.get("num_players", 4)),
                    json.dumps(sorted(lineup)),
                    json.dumps(payload.get("rules") or {}),
                    str(payload.get("scoring") or ""),
                    int(payload.get("user_seat", 0)),
                    int(payload.get("user_score", 0)),
                    1 if payload.get("won") else 0,
                    json.dumps(payload.get("replay") or {}),
                ),
            )
            con.commit()

    def stats(self, user_id: int) -> Dict:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                f"SELECT lineup, num_players, COUNT(*), SUM(user_score), "
                f"SUM(won), MAX(ts) FROM games WHERE user_id = {self._ph} "
                f"GROUP BY lineup, num_players ORDER BY MAX(ts) DESC",
                (user_id,),
            )
            rows = cur.fetchall()
            cur.execute(
                f"SELECT COUNT(*), COALESCE(SUM(user_score), 0), "
                f"COALESCE(SUM(won), 0) FROM games WHERE user_id = {self._ph}",
                (user_id,),
            )
            total = cur.fetchone()
        return {
            "games": int(total[0]),
            "total_score": int(total[1]),
            "wins": int(total[2]),
            "persistent": self.persistent,
            "lineups": [
                {
                    "lineup": json.loads(r[0]),
                    "num_players": int(r[1]),
                    "games": int(r[2]),
                    "total_score": int(r[3]),
                    "avg_score": round(r[3] / r[2], 2),
                    "wins": int(r[4]),
                    "last_ts": float(r[5]),
                }
                for r in rows
            ],
        }

    def export_replays(self, path: str) -> int:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT g.id, u.username, g.ts, g.lineup, g.rules, g.scoring, "
                "g.user_seat, g.user_score, g.won, g.replay "
                "FROM games g JOIN users u ON u.id = g.user_id ORDER BY g.ts"
            )
            n = 0
            with open(path, "w") as f:
                for row in cur.fetchall():
                    f.write(json.dumps({
                        "id": row[0], "username": row[1], "ts": row[2],
                        "lineup": json.loads(row[3]),
                        "rules": json.loads(row[4]), "scoring": row[5],
                        "user_seat": row[6], "user_score": row[7],
                        "won": bool(row[8]), "replay": json.loads(row[9]),
                    }) + "\n")
                    n += 1
        return n


_STORE: Optional[Store] = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", metavar="PATH",
                        help="dump recorded human games as JSONL")
    args = parser.parse_args()
    if args.export:
        n = get_store().export_replays(args.export)
        print(f"exported {n} recorded games -> {args.export}")


if __name__ == "__main__":
    main()
