"""Query Neon over HTTPS when the Postgres port is unreachable.

Sandboxed sessions route all traffic through a policy proxy that speaks
HTTP CONNECT.  Even with the database host allowlisted, that only opens
**443** — the Postgres wire port stays dark, and libpq cannot use an
HTTP proxy anyway (see big2/pgtunnel.py for the tunnel that handles the
case where 5432 *is* reachable).

Neon also exposes SQL over HTTPS: POST a statement to ``/sql`` on the
endpoint host with the connection string in a header.  That path goes
through the ordinary proxy like any other web request, so it works
wherever the domain is permitted.

    export DATABASE_URL='postgresql://...'
    python -m big2.neon_http --export replays.jsonl
    python -m big2.neon_http --sql "SELECT COUNT(*) FROM games"

The connection string is read from the environment and never written to
disk or logs.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

TIMEOUT = 60


class NeonHTTP:
    """Minimal SQL-over-HTTPS client for a Neon endpoint."""

    def __init__(self, url: Optional[str] = None):
        url = url or os.environ.get("DATABASE_URL") or os.environ.get(
            "POSTGRES_URL"
        )
        if not url:
            raise RuntimeError("set DATABASE_URL to the Neon connection string")
        parsed = urllib.parse.urlparse(url)
        if not parsed.hostname:
            raise ValueError("connection string has no host")
        self.conn = url
        self.endpoint = f"https://{parsed.hostname}/sql"

    def query(self, sql: str, params: Optional[Sequence[Any]] = None
              ) -> List[List[Any]]:
        """Rows as lists.  Raises with the server's message on error."""
        req = urllib.request.Request(
            self.endpoint,
            method="POST",
            data=json.dumps({"query": sql, "params": list(params or [])}).encode(),
            headers={
                "Content-Type": "application/json",
                "Neon-Connection-String": self.conn,
                "Neon-Raw-Text-Output": "true",
                "Neon-Array-Mode": "true",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.load(resp).get("rows", [])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"neon http {exc.code}: {detail}") from None

    # ------------------------------------------------------------------

    def leaderboard(self) -> List[Dict[str, Any]]:
        rows = self.query(
            "SELECT u.username, COUNT(g.id), COALESCE(SUM(g.won), 0), "
            "COALESCE(SUM(g.user_score), 0) "
            "FROM users u LEFT JOIN games g ON g.user_id = u.id "
            "GROUP BY u.username ORDER BY COUNT(g.id) DESC"
        )
        return [
            {"username": r[0], "games": int(r[1]), "wins": int(r[2]),
             "total_score": int(r[3])}
            for r in rows
        ]

    def by_lineup(self) -> List[Dict[str, Any]]:
        """How humans do against each model line — the number that says
        whether a shipped bot is actually holding up."""
        rows = self.query(
            "SELECT lineup, COUNT(*), COALESCE(SUM(user_score), 0), "
            "COALESCE(SUM(won), 0) FROM games GROUP BY lineup"
        )
        out = []
        for lineup, n, score, wins in rows:
            kinds = sorted({s.split("@")[0] for s in json.loads(lineup)})
            out.append({
                "models": kinds, "games": int(n),
                "human_score": int(score), "human_wins": int(wins),
                "human_per_game": int(score) / max(1, int(n)),
            })
        out.sort(key=lambda r: -r["games"])
        return out

    def export_replays(self, path: str, limit: Optional[int] = None) -> int:
        """Dump recorded games as JSONL in the shape big2.offline reads."""
        sql = (
            "SELECT g.id, u.username, g.ts, g.lineup, g.rules, g.scoring, "
            "g.user_seat, g.user_score, g.won, g.replay "
            "FROM games g JOIN users u ON u.id = g.user_id ORDER BY g.ts"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.query(sql)
        n = 0
        with open(path, "w") as f:
            for r in rows:
                replay = r[9]
                if isinstance(replay, str):
                    replay = json.loads(replay)
                f.write(json.dumps({
                    "id": r[0], "username": r[1], "ts": r[2],
                    "lineup": json.loads(r[3]) if isinstance(r[3], str) else r[3],
                    "rules": json.loads(r[4]) if isinstance(r[4], str) else r[4],
                    "scoring": r[5], "user_seat": int(r[6]),
                    "user_score": int(r[7]), "won": bool(int(r[8])),
                    "replay": replay,
                }) + "\n")
                n += 1
        return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", metavar="PATH")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sql")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    db = NeonHTTP()
    if args.sql:
        for row in db.query(args.sql):
            print(row)
    if args.report:
        print(f"{'tester':<20}{'games':>6}{'wins':>6}{'total':>8}{'avg':>8}")
        for t in db.leaderboard():
            n = t["games"]
            avg = f"{t['total_score'] / n:+.1f}" if n else "-"
            print(f"{t['username']:<20}{n:>6}{t['wins']:>6}"
                  f"{t['total_score']:>+8}{avg:>8}")
        print("\nhumans vs each model line:")
        for r in db.by_lineup():
            print(f"  vs {'/'.join(r['models']):<24} {r['games']:>4} games  "
                  f"human {r['human_score']:>+5} ({r['human_per_game']:+.2f}/game)")
    if args.export:
        n = db.export_replays(args.export, limit=args.limit)
        print(f"exported {n} recorded games -> {args.export}")


if __name__ == "__main__":
    main()
