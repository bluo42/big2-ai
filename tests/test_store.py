import os
import tempfile
import unittest

from big2.store import Store


class TestStore(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.store = Store(url=os.path.join(self._dir.name, "test.db"))

    def tearDown(self):
        self._dir.cleanup()

    def test_register_login_auth(self):
        token = self.store.register("alice", "hunter2")
        self.assertEqual(self.store.auth(token)[1], "alice")
        token2 = self.store.login("alice", "hunter2")
        self.assertEqual(self.store.auth(token2)[1], "alice")
        with self.assertRaises(ValueError):
            self.store.register("alice", "other")  # unique username
        with self.assertRaises(ValueError):
            self.store.login("alice", "wrong")
        self.assertIsNone(self.store.auth("alice:forgedsignature"))
        self.assertIsNone(self.store.auth(None))

    def test_username_and_password_rules(self):
        with self.assertRaises(ValueError):
            self.store.register("x", "goodpass")  # too short
        with self.assertRaises(ValueError):
            self.store.register("has space", "goodpass")
        with self.assertRaises(ValueError):
            self.store.register("fine_name", "abc")  # weak password

    def test_record_and_stats_grouping(self):
        token = self.store.register("bob", "secret1")
        uid = self.store.auth(token)[0]
        lineup_a = ["ppo@20260817-1755", "linear@20260817-0330", "evo@20260817-0645"]
        lineup_b = ["smart@builtin"]
        for score, won in ((10, True), (-4, False), (7, True)):
            self.store.record_game(uid, {
                "num_players": 4, "lineup": lineup_a, "rules": {},
                "scoring": "tiered", "user_seat": 0, "user_score": score,
                "won": won, "replay": {"actions": []},
            })
        self.store.record_game(uid, {
            "num_players": 2, "lineup": lineup_b, "rules": {},
            "scoring": "tiered", "user_seat": 0, "user_score": -6,
            "won": False, "replay": {"actions": []},
        })
        stats = self.store.stats(uid)
        self.assertEqual(stats["games"], 4)
        self.assertEqual(stats["wins"], 2)
        self.assertEqual(stats["total_score"], 7)
        self.assertEqual(len(stats["lineups"]), 2)
        by_np = {l["num_players"]: l for l in stats["lineups"]}
        self.assertEqual(by_np[4]["games"], 3)
        self.assertEqual(by_np[4]["total_score"], 13)
        self.assertEqual(sorted(by_np[4]["lineup"]), sorted(lineup_a))

    def test_export_replays(self):
        token = self.store.register("carol", "pass123")
        uid = self.store.auth(token)[0]
        self.store.record_game(uid, {
            "num_players": 4, "lineup": ["a"], "rules": {"pass_locks": False},
            "scoring": "tiered", "user_seat": 0, "user_score": 3, "won": True,
            "replay": {"actions": [{"p": 0, "cards": [0]}]},
        })
        out = os.path.join(self._dir.name, "replays.jsonl")
        n = self.store.export_replays(out)
        self.assertEqual(n, 1)
        import json

        row = json.loads(open(out).read().strip())
        self.assertEqual(row["username"], "carol")
        self.assertEqual(row["replay"]["actions"][0]["cards"], [0])


class TestLeaderboard(unittest.TestCase):
    def test_leaderboard_rows_and_totals(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(url=os.path.join(d, "lb.db"))
            t1 = store.register("erin", "pass123")
            store.register("frank", "pass123")  # never plays
            uid = store.auth(t1)[0]
            for score, won in ((8, True), (-3, False)):
                store.record_game(uid, {
                    "num_players": 4, "lineup": ["ppo@x"], "rules": {},
                    "scoring": "tiered", "user_seat": 0, "user_score": score,
                    "won": won, "replay": {},
                })
            board = store.leaderboard()
            # per-lineup rows: the client filters/aggregates from these
            self.assertEqual(len(board["rows"]), 1)
            self.assertEqual(board["rows"][0]["username"], "erin")
            self.assertEqual(board["rows"][0]["lineup"], ["ppo@x"])
            self.assertEqual(board["rows"][0]["games"], 2)
            self.assertEqual(board["rows"][0]["wins"], 1)
            self.assertEqual(len(board["testers"]), 2)
            top = board["testers"][0]
            self.assertEqual(top["username"], "erin")
            self.assertEqual(top["games"], 2)
            self.assertEqual(top["wins"], 1)
            self.assertEqual(top["total_score"], 5)
            idle = board["testers"][1]
            self.assertEqual(idle["games"], 0)
            self.assertIsNone(idle["last_ts"])
            self.assertEqual(board["totals"],
                             {"games": 2, "human_wins": 1, "human_score": 5})
            # the CLI report renders from the same data
            self.assertIn("erin", store.report())


class TestEndpoints(unittest.TestCase):
    def test_full_flow_through_flask(self):
        import big2.store as store_mod

        with tempfile.TemporaryDirectory() as d:
            old = store_mod._STORE
            store_mod._STORE = Store(url=os.path.join(d, "api.db"))
            try:
                from big2.server import app

                c = app.test_client()
                r = c.post("/api/register",
                           json={"username": "dave", "password": "pw1234"})
                self.assertEqual(r.status_code, 200)
                token = r.get_json()["token"]
                view = c.post("/api/new", json={"num_ai": 1, "ai": ["lowest"],
                                                "seed": 3}).get_json()
                self.assertIn("stamp", view["players"][1])
                r = c.post("/api/record-game", json={"token": token, "game": {
                    "num_players": 2, "lineup": [view["players"][1]["stamp"]],
                    "rules": view["rules"], "scoring": view["scoring"],
                    "user_seat": 0, "user_score": 5, "won": True,
                    "replay": {},
                }})
                self.assertEqual(r.status_code, 200)
                stats = c.post("/api/stats", json={"token": token}).get_json()
                self.assertEqual(stats["games"], 1)
                self.assertEqual(stats["username"], "dave")
                bad = c.post("/api/stats", json={"token": "x:y"})
                self.assertEqual(bad.status_code, 400)
            finally:
                store_mod._STORE = old

class TestProxyTunnel(unittest.TestCase):
    """The client-side piece libpq is missing: a CONNECT tunnel.

    Verified end-to-end against an allowed host in this environment;
    these cover the parts that need no network.
    """

    def test_dsn_keeps_the_real_host_for_tls_and_dials_the_tunnel(self):
        from big2.pgtunnel import tunnel_dsn

        url = ("postgresql://alice:secret@db.example.com:5432/appdb"
               "?sslmode=require&channel_binding=require")
        dsn, tun = tunnel_dsn(url)
        try:
            self.assertIn("host=db.example.com", dsn)   # SNI + cert target
            self.assertIn("hostaddr=127.0.0.1", dsn)    # where bytes go
            self.assertIn(f"port={tun.local_port}", dsn)
            self.assertIn("dbname=appdb", dsn)
            self.assertIn("user=alice", dsn)
            self.assertIn("password=secret", dsn)
            self.assertIn("sslmode=require", dsn)
            # channel binding depends on the TLS endpoint: dropped rather
            # than risking a SCRAM mismatch through the tunnel
            self.assertNotIn("channel_binding", dsn)
            self.assertNotEqual(tun.local_port, 0)
        finally:
            tun.close()

    def test_proxy_address_is_read_from_the_environment(self):
        import os

        from big2.pgtunnel import proxy_address

        old = os.environ.get("HTTPS_PROXY")
        try:
            os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9999"
            self.assertEqual(proxy_address(), ("127.0.0.1", 9999))
        finally:
            if old is None:
                os.environ.pop("HTTPS_PROXY", None)
            else:
                os.environ["HTTPS_PROXY"] = old

    def test_missing_proxy_is_reported_clearly(self):
        import os

        from big2.pgtunnel import ProxyTunnel

        saved = {k: os.environ.pop(k) for k in
                 ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
                 if k in os.environ}
        try:
            with self.assertRaises(RuntimeError):
                ProxyTunnel("db.example.com")
        finally:
            os.environ.update(saved)


if __name__ == "__main__":
    unittest.main()
