import random
import unittest

from big2 import webapi
from big2.game import Big2Game
from big2.strategies import PlayLowest


class TestSerialization(unittest.TestCase):
    def test_round_trip_mid_game(self):
        game = Big2Game(rng=random.Random(3))
        policy = PlayLowest()
        for _ in range(9):
            if game.game_over:
                break
            moves = game.legal_moves()
            game.step(policy.select(game, game.turn) if moves else None)
        blob = webapi.serialize_game(game, {0: None, 1: "smart", 2: "smart", 3: "smart"})
        clone = webapi.deserialize_game(blob)
        self.assertEqual(clone.turn, game.turn)
        self.assertEqual(clone.hands, game.hands)
        self.assertEqual(clone.passed, game.passed)
        self.assertEqual(clone.played_cards, game.played_cards)
        self.assertEqual(
            [m.cards for m in clone.legal_moves()],
            [m.cards for m in game.legal_moves()],
        )
        self.assertEqual(len(clone.history), len(game.history))
        # deserialized game keeps playing to a valid end
        scores = clone.play_out([PlayLowest()] * 4)
        self.assertEqual(sum(scores.values()), 0)


class TestHandlers(unittest.TestCase):
    def test_new_then_actions_to_completion(self):
        view = webapi.new_game(
            {"num_ai": 3, "ai": ["lowest", "lowest", "lowest"], "seed": 5}
        )
        steps = 0
        while view["phase"] == "playing":
            body = {"state": view["full_state"]}
            if view["legal_moves"]:
                body["cards"] = view["legal_moves"][0]["cards"]
            else:
                body["pass"] = True
            view = webapi.apply_action(body)
            steps += 1
            self.assertLess(steps, 300)
        self.assertEqual(sum(view["scores"].values()), 0)

    def test_illegal_action_rejected(self):
        view = webapi.new_game({"num_ai": 1, "ai": ["lowest"], "seed": 1})
        with self.assertRaises(ValueError):
            webapi.apply_action({"state": view["full_state"], "cards": [51, 50]})

    def test_hint_and_beliefs(self):
        view = webapi.new_game({"num_ai": 2, "ai": ["smart", "smart"], "seed": 2})
        h = webapi.hint({"state": view["full_state"]})
        self.assertIn("type", h)
        b = webapi.beliefs({"state": view["full_state"]})
        self.assertEqual(len(b["opponents"]), 2)

    def test_simulate_replay_integrity(self):
        out = webapi.simulate(
            {"agents": ["lowest", "dumper", "smart"], "games": 3, "seed": 9}
        )
        self.assertEqual(len(out["replays"]), 3)
        for rep in out["replays"]:
            hands = [list(h) for h in rep["initial_hands"]]
            for a in rep["actions"]:
                if a["cards"]:
                    for c in a["cards"]:
                        hands[a["p"]].remove(c)  # raises if inconsistent
            self.assertEqual(len(hands[rep["winner"]]), 0)
            self.assertEqual(sum(rep["scores"].values()), 0)

    def test_analyze_defaults_to_the_seats_own_model(self):
        """The explorer must show what the seated model is actually
        doing: the analyzing model resolves from the replay's player
        label unless the request overrides it."""
        sim = webapi.simulate({"agents": ["ppo11", "smart", "lowest",
                                          "dumper"], "games": 1, "seed": 4})
        rep = sim["replays"][0]
        hit = None
        for k in range(len(rep["actions"])):
            d = webapi.analyze({"replay": rep, "k": k})
            if d.get("over"):
                break
            if d.get("seat") == 0:
                hit = d
                break
        self.assertIsNotNone(hit)
        self.assertEqual(hit["model"], "ppo11")
        self.assertEqual(hit["model_label"], "WangBot_v1")
        self.assertTrue(hit["model_is_seats_own"])
        # an explicit override still wins, and says it is not the seat's
        # (linear is not seated at this table, so it can't be "own")
        d = webapi.analyze({"replay": rep, "k": 0, "model": "linear"})
        self.assertEqual(d["model"], "linear")
        self.assertFalse(d["model_is_seats_own"])

    def test_kind_resolver_reads_every_label_shape(self):
        cases = (("0:wangbot2", "wangbot2"), ("wangbot2@a1b2c3", "wangbot2"),
                 ("AI 2 (Khabib)", "khabib"), ("v2", "wangbot2"),
                 ("WangBot_v1", "ppo11"), ("smart", "smart"),
                 ("brandonluo", None), ("You", None), ("", None))
        for name, want in cases:
            self.assertEqual(webapi._kind_from_name(name), want, name)

    def test_hint_uses_the_table_model(self):
        view = webapi.new_game({"num_ai": 1, "ai": ["ppo11"], "seed": 3})
        state = view["full_state"]
        game, _ = webapi._load({"state": state})
        if game.turn == webapi.HUMAN and not game.game_over:
            out = webapi.hint({"state": state})
            self.assertEqual(out["model"], "WangBot_v1")

    def test_two_ppo_lines_are_distinct_kinds(self):
        self.assertIn("ppo11", webapi.AI_KINDS)
        # each line stamps under its own display name, so recorded games
        # stay attributable to the exact model version at the table
        self.assertTrue(webapi.model_stamp("ppo").startswith("ppo@"))
        self.assertTrue(webapi.model_stamp("ppo11").startswith("WangBot_v1@"))
        view = webapi.new_game({"num_ai": 1, "ai": ["ppo11"], "seed": 11})
        self.assertEqual(view["players"][1]["ai"], "ppo11")
        self.assertIn("WangBot_v1", view["players"][1]["name"])

    def test_assist_surface_gated_on_public_deploys(self):
        from big2.server import app

        c = app.test_client()
        old = app.config.get("BIG2_ADMIN")
        try:
            app.config["BIG2_ADMIN"] = False
            self.assertEqual(c.get("/admin").status_code, 404)
            self.assertEqual(c.post("/api/hint", json={}).status_code, 404)
            self.assertEqual(c.post("/api/beliefs", json={}).status_code, 404)
            self.assertEqual(c.post("/api/simulate", json={}).status_code, 404)
            self.assertEqual(c.post("/api/analyze", json={}).status_code, 404)
            # core play surface stays open
            self.assertEqual(
                c.post("/api/new", json={"num_ai": 1, "ai": ["lowest"]}).status_code,
                200,
            )
            app.config["BIG2_ADMIN"] = True
            self.assertEqual(c.get("/admin").status_code, 200)
            self.assertNotEqual(
                c.post("/api/analyze", json={}).status_code, 404)
        finally:
            app.config["BIG2_ADMIN"] = old

    def test_admin_account_unlocks_assist_surface(self):
        import os
        import tempfile

        import big2.store as store_mod
        from big2.server import app
        from big2.store import Store

        with tempfile.TemporaryDirectory() as d:
            old_store = store_mod._STORE
            store_mod._STORE = Store(url=os.path.join(d, "adm.db"))
            c = app.test_client()
            old = app.config.get("BIG2_ADMIN")
            try:
                app.config["BIG2_ADMIN"] = False
                admin_tok = store_mod._STORE.register("brandonluo", "pw1234")
                user_tok = store_mod._STORE.register("randomguy", "pw1234")
                # anonymous and ordinary accounts stay locked out
                self.assertEqual(c.get("/admin").status_code, 404)
                self.assertEqual(
                    c.get(f"/admin?token={user_tok}").status_code, 404
                )
                self.assertEqual(
                    c.post("/api/hint", json={"token": user_tok}).status_code,
                    404,
                )
                # ...but the leaderboard is public for everyone
                self.assertEqual(
                    c.post("/api/leaderboard", json={}).status_code, 200
                )
                # the admin account opens the full surface, any carrier
                self.assertEqual(
                    c.get(f"/admin?token={admin_tok}").status_code, 200
                )
                self.assertNotEqual(
                    c.post("/api/analyze", json={},
                           headers={"X-Big2-Token": admin_tok}).status_code,
                    404,
                )
                # Admins are kept OFF the board: they can face the table
                # up and read the bots' beliefs, so their scores are not
                # comparable to a player's.
                board = c.post("/api/leaderboard", json={"token": admin_tok})
                self.assertEqual(board.status_code, 200)
                self.assertEqual(
                    [t["username"] for t in board.get_json()["testers"]],
                    ["randomguy"],
                )
                # login/stats responses carry the admin flag for the UI
                login = c.post("/api/login", json={
                    "username": "brandonluo", "password": "pw1234"
                }).get_json()
                self.assertTrue(login["admin"])
                self.assertFalse(c.post("/api/login", json={
                    "username": "randomguy", "password": "pw1234"
                }).get_json()["admin"])
                stats = c.post("/api/stats",
                               json={"token": admin_tok}).get_json()
                self.assertTrue(stats["admin"])
            finally:
                app.config["BIG2_ADMIN"] = old
                store_mod._STORE = old_store


if __name__ == "__main__":
    unittest.main()
