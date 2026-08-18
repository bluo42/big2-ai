import random
import unittest

import numpy as np

from big2.game import Big2Game
from big2.humanlike import (
    human_decisions,
    split_by_game,
    style_stats_from_replays,
)
from big2.strategies import SmartHeuristic

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def synthetic_rows(n_games=6, username="tester", seed=0):
    """Replay rows in the export schema, from real self-play games."""
    rows = []
    for k in range(n_games):
        g = Big2Game(rng=random.Random(seed + k))
        initial = [list(h) for h in g.hands]
        actions = []
        pol = SmartHeuristic()
        while not g.game_over:
            p = g.turn
            m = pol.select(g, p)
            actions.append(
                {"p": p, "cards": [] if m is None else list(m.cards)}
            )
            g.step(m)
        rows.append({
            "username": username,
            "user_seat": k % 4,
            "replay": {
                "initial_hands": initial,
                "actions": actions,
                "num_players": 4,
            },
        })
    return rows


class TestDataset(unittest.TestCase):
    def test_only_the_named_players_multi_option_decisions(self):
        rows = synthetic_rows() + synthetic_rows(2, username="other", seed=50)
        items = human_decisions(rows, ["tester"])
        self.assertTrue(items)
        for it in items:
            self.assertEqual(it["user"], "tester")
            self.assertGreaterEqual(it["n_options"], 2)
            self.assertLess(it["chosen"], it["n_options"])
            self.assertEqual(it["acts"].shape[0], it["n_options"])

    def test_split_holds_out_whole_games(self):
        items = human_decisions(synthetic_rows(8), ["tester"])
        train, val = split_by_game(items, val_frac=0.25, seed=1)
        self.assertTrue(train and val)
        self.assertFalse({it["game"] for it in train}
                         & {it["game"] for it in val})

    def test_style_stats_are_sane(self):
        s = style_stats_from_replays(synthetic_rows(), ["tester"])
        self.assertGreaterEqual(s["pass_rate"], 0.0)
        self.assertLessEqual(s["pass_rate"], 1.0)
        self.assertGreater(s["followable"], 0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTraining(unittest.TestCase):
    def test_clone_learns_and_round_trips_as_a_policy(self):
        import torch

        from big2.humanlike import evaluate, train
        from big2.neural import PPOPolicy

        items = human_decisions(synthetic_rows(14), ["tester"])
        tr, va = split_by_game(items, seed=2)
        net = train(tr, va, d_model=48, epochs=8, batch=32,
                    patience=99, verbose=False)
        rep = evaluate(net, va)
        self.assertTrue(np.isfinite(rep["loss"]))
        # it must have learned *something* about SmartHeuristic's habits
        self.assertGreater(
            rep["top1"], np.mean([1.0 / it["n_options"] for it in va])
        )
        # a saved clone is an ordinary policy checkpoint
        import tempfile, os

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clone.pt")
            torch.save({"state_dict": net.state_dict(), "d_model": 48,
                        "heads": 4}, path)
            pol = PPOPolicy.load(path)
            g = Big2Game(rng=random.Random(3))
            move = pol.select(g, g.turn)
            legal = list(g.legal_moves(g.turn))
            self.assertIn(move, legal)


if __name__ == "__main__":
    unittest.main()
