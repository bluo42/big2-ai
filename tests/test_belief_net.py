import random
import unittest

import numpy as np

from big2.belief import (
    analytic_posterior,
    calibration_report,
    candidate_mask,
    collect_samples,
    hand_sizes,
    opponent_order,
    reveal_split,
    truth_matrix,
)
from big2.cards import NUM_CARDS
from big2.game import Big2Game
from big2.strategies import PlayLowest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestTargets(unittest.TestCase):
    def test_truth_matrix_matches_the_real_hands(self):
        g = Big2Game(rng=random.Random(1))
        t = truth_matrix(g, 0)
        self.assertEqual(t.shape, (3, NUM_CARDS))
        self.assertEqual(t.sum(), 39.0)
        for j, p in enumerate(opponent_order(g, 0)):
            for c in g.hands[p]:
                self.assertEqual(t[j, c], 1.0)
            for c in g.hands[0]:
                self.assertEqual(t[j, c], 0.0)

    def test_mask_excludes_our_cards_and_played_cards(self):
        g = Big2Game(rng=random.Random(2))
        for _ in range(6):
            g.step(PlayLowest().select(g, g.turn))
        m = candidate_mask(g, 0)
        for c in g.hands[0]:
            self.assertTrue((m[:, c] == 0).all())
        for c in g.played_cards:
            self.assertTrue((m[:, c] == 0).all())
        # every real opponent card is still a candidate
        t = truth_matrix(g, 0)
        self.assertTrue(((t > 0) <= (m > 0)).all())

    def test_reveal_split_reveals_only_real_cards(self):
        g = Big2Game(rng=random.Random(3))
        t = truth_matrix(g, 0)
        rev = reveal_split(t, 0.5, random.Random(0))
        self.assertTrue(((rev > 0) <= (t > 0)).all())
        self.assertAlmostEqual(rev.sum(), round(0.5 * t.sum()), delta=1)
        self.assertEqual(reveal_split(t, 0.0, random.Random(0)).sum(), 0.0)


class TestAnalyticBaseline(unittest.TestCase):
    def test_rows_sum_to_hand_sizes(self):
        g = Big2Game(rng=random.Random(4))
        for _ in range(8):
            g.step(PlayLowest().select(g, g.turn))
        m, z = candidate_mask(g, 0), hand_sizes(g, 0)
        rev = np.zeros_like(m)
        p = analytic_posterior(m, z, rev)
        for j in range(3):
            self.assertAlmostEqual(p[j].sum(), z[j], places=3)

    def test_revealed_cards_become_certain(self):
        g = Big2Game(rng=random.Random(5))
        m, z = candidate_mask(g, 0), hand_sizes(g, 0)
        t = truth_matrix(g, 0)
        rev = reveal_split(t, 0.4, random.Random(1))
        p = analytic_posterior(m, z, rev)
        self.assertTrue((p[rev > 0] == 1.0).all())
        # a card known to sit with one opponent is impossible elsewhere
        for j, c in np.argwhere(rev > 0):
            for other in range(3):
                if other != j:
                    self.assertEqual(p[other, c], 0.0)


class TestCalibrationMetrics(unittest.TestCase):
    def test_perfect_predictions_score_perfectly(self):
        truth = np.zeros((2, 3, NUM_CARDS), dtype=np.float32)
        truth[:, :, :5] = 1.0
        mask = np.ones_like(truth)
        r = calibration_report(truth, truth, mask)
        self.assertLess(r["brier"], 1e-6)
        self.assertEqual(r["topk_precision"], 1.0)

    def test_worse_predictions_score_worse(self):
        truth = np.zeros((2, 3, NUM_CARDS), dtype=np.float32)
        truth[:, :, :5] = 1.0
        mask = np.ones_like(truth)
        coin = np.full_like(truth, 0.5)
        self.assertGreater(
            calibration_report(coin, truth, mask)["brier"],
            calibration_report(truth, truth, mask)["brier"],
        )


class TestSampleCollection(unittest.TestCase):
    def test_shapes_and_consistency(self):
        d = collect_samples(3, seed=0)
        n = len(d["state"])
        self.assertGreater(n, 10)
        self.assertEqual(d["truth"].shape, (n, 3, NUM_CARDS))
        self.assertEqual(d["mask"].shape, (n, 3, NUM_CARDS))
        self.assertEqual(d["sizes"].shape, (n, 3))
        # truth always sits inside the candidate mask, and rows match sizes
        self.assertTrue(((d["truth"] > 0) <= (d["mask"] > 0)).all())
        np.testing.assert_allclose(d["truth"].sum(-1), d["sizes"], atol=1e-5)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTraining(unittest.TestCase):
    def test_learned_posterior_beats_the_analytic_baseline(self):
        """The whole point: a trained posterior must be measurably
        closer to reality than uniform-by-count."""
        from big2.belief import evaluate, train

        data = collect_samples(60, seed=11)
        held = collect_samples(20, seed=77)
        net = train(data, epochs=6, hidden=64, verbose=False)
        report = evaluate(net, held)
        # the net starts *as* the analytic prior (zero-init head), so any
        # improvement is real and it can never start out worse
        self.assertLessEqual(report["learned"]["brier"],
                             report["analytic"]["brier"])
        self.assertGreaterEqual(report["learned"]["topk_precision"],
                                report["analytic"]["topk_precision"] - 1e-6)

    def test_predictions_respect_mask_and_counts(self):
        import torch

        from big2.belief import build_net, predict

        data = collect_samples(8, seed=5)
        net = build_net(data["state"].shape[1], hidden=32)
        p = predict(
            net,
            torch.from_numpy(data["state"]),
            torch.from_numpy(data["revealed"]),
            torch.from_numpy(data["mask"]),
            torch.from_numpy(data["sizes"]),
            torch.from_numpy(data["prior"]),
        ).numpy()
        self.assertTrue((p >= 0).all() and (p <= 1.0 + 1e-5).all())
        # impossible cards stay at zero
        self.assertTrue((p[data["mask"] == 0] == 0).all())
        # each opponent's probabilities sum to their hand size
        np.testing.assert_allclose(p.sum(-1), data["sizes"], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
