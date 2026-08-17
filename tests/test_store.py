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


if __name__ == "__main__":
    unittest.main()
