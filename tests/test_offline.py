import json
import os
import random
import tempfile
import unittest

from big2.game import Big2Game, ScoringConfig
from big2.offline import (
    build_dataset,
    iter_decisions,
    load_replays,
    rebuild_game,
)
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayLowest, SmartHeuristic

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def synth_replay(seed, user_seat=0):
    """A finished game in the frontend's recorded-replay shape."""
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=4, rng=random.Random(seed))
    initial = [list(h) for h in g.hands]
    policies = [SmartHeuristic(), PlayLowest(), PlayLowest(), PlayLowest()]
    scores = g.play_out(policies)
    return {
        "user_seat": user_seat,
        "user_score": scores[user_seat],
        "replay": {
            "num_players": 4,
            "rules": {"allow_triples": False, "pass_locks": False,
                      "flush_rank_first": False},
            "scoring": "tiered",
            "start_card": g.start_card,
            "initial_hands": initial,
            "actions": [
                {"p": r.player,
                 "cards": list(r.combo.cards) if r.combo else None,
                 "te": r.trick_end}
                for r in g.history
            ],
            "scores": {str(p): s for p, s in scores.items()},
            "winner": g.winner,
        },
    }


class TestReplayReconstruction(unittest.TestCase):
    def test_rebuild_matches_the_original_deal(self):
        row = synth_replay(1)
        g = rebuild_game(row["replay"])
        self.assertEqual(g.hands, [sorted(h) for h in
                                   row["replay"]["initial_hands"]])
        self.assertFalse(g.game_over)
        self.assertTrue(g.first_play)
        self.assertIn(g.start_card, g.hands[g.turn])

    def test_decisions_replay_to_the_recorded_outcome(self):
        row = synth_replay(2)
        body = row["replay"]
        seen = 0
        game = None
        for game, p, cards in iter_decisions(body):
            self.assertEqual(p, game.turn)
            seen += 1
        self.assertGreater(seen, 10)
        # walking every recorded action reproduces the recorded result
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, body["winner"])
        self.assertEqual(
            {str(p): s for p, s in game.scores.items()}, body["scores"]
        )

    def test_load_replays_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.jsonl")
            with open(path, "w") as f:
                for s in (1, 2, 3):
                    f.write(json.dumps(synth_replay(s)) + "\n")
            rows = load_replays(path)
            self.assertEqual(len(rows), 3)


class TestDataset(unittest.TestCase):
    def test_weights_favour_the_bigger_wins(self):
        rows = [synth_replay(s) for s in range(12)]
        data = build_dataset(rows, seats="all", min_margin=1.0, beta=6.0)
        self.assertGreater(data["n"], 20)
        self.assertEqual(data["state"].shape[0], data["n"])
        self.assertEqual(data["acts"].shape[0], data["n"])
        self.assertEqual(data["chosen"].shape[0], data["n"])
        # every weight is a positive, bounded AWR factor
        self.assertTrue((data["weight"] > 0).all())
        self.assertTrue((data["weight"] <= 8.0 + 1e-6).all())
        # the chosen index always points at a real (unmasked) option
        for i in range(data["n"]):
            self.assertTrue(data["mask"][i, data["chosen"][i]])

    def test_human_seat_filter_selects_only_that_seat(self):
        rows = [synth_replay(s, user_seat=0) for s in range(10)]
        human = build_dataset(rows, seats="human", min_margin=-99)
        every = build_dataset(rows, seats="all", min_margin=-99)
        self.assertGreater(every["n"], human["n"])

    def test_min_margin_filters_out_losing_games(self):
        rows = [synth_replay(s) for s in range(10)]
        loose = build_dataset(rows, seats="all", min_margin=-99)
        strict = build_dataset(rows, seats="all", min_margin=8.0)
        self.assertGreater(loose["n"], strict.get("n", 0))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestAWRTraining(unittest.TestCase):
    def test_fine_tune_runs_and_saves(self):
        import torch

        from big2.neural import ACT_DIM, STATE_DIM, build_net
        from big2.offline import train_awr

        rows = [synth_replay(s) for s in range(8)]
        data = build_dataset(rows, seats="all", min_margin=1.0)
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "base.pt")
            net = build_net(d_model=32, heads=2, state_dim=STATE_DIM,
                            act_dim=ACT_DIM)
            torch.save({"state_dict": net.state_dict(), "d_model": 32,
                        "heads": 2}, base)
            out = os.path.join(d, "awr.pt")
            train_awr(data, base, out, epochs=2, minibatch=32, verbose=False)
            self.assertTrue(os.path.exists(out))
            saved = torch.load(out, map_location="cpu", weights_only=True)
            self.assertEqual(saved["meta"]["awr_decisions"], data["n"])
            self.assertEqual(saved["meta"]["note"], "awr-human")

    def test_fine_tune_moves_toward_the_demonstrated_moves(self):
        """The point of AWR: after fitting, the net assigns the recorded
        winning moves more probability than it did before."""
        import torch

        from big2.neural import ACT_DIM, STATE_DIM, build_net
        from big2.offline import train_awr

        rows = [synth_replay(s) for s in range(10)]
        data = build_dataset(rows, seats="all", min_margin=1.0)
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "base.pt")
            net = build_net(d_model=32, heads=2, state_dim=STATE_DIM,
                            act_dim=ACT_DIM)
            torch.save({"state_dict": net.state_dict(), "d_model": 32,
                        "heads": 2}, base)

            S = torch.from_numpy(data["state"])
            A = torch.from_numpy(data["acts"])
            M = torch.from_numpy(data["mask"])
            C = torch.from_numpy(data["chosen"])

            def mean_logp(model):
                with torch.no_grad():
                    logits, _, _ = model(S, A, M)
                    lp = torch.log_softmax(logits, dim=-1)
                    return float(lp.gather(1, C.unsqueeze(1)).mean())

            before = mean_logp(net)
            tuned = train_awr(data, base, os.path.join(d, "awr.pt"),
                              epochs=6, minibatch=32, lr=1e-3, verbose=False)
            self.assertGreater(mean_logp(tuned), before)


if __name__ == "__main__":
    unittest.main()
