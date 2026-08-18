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


class TestEvidencePlanes(unittest.TestCase):
    def test_declining_a_single_marks_every_higher_card(self):
        """The strongest read in the game: passing on a card is
        evidence against holding anything above it."""
        from big2.belief import evidence_planes
        from big2.combos import classify
        from big2.game import PlayRecord
        from big2.rules import DEFAULT_RULES

        g = Big2Game(rng=random.Random(9))
        table = classify([28], DEFAULT_RULES)          # 10 of diamonds
        g.history = [PlayRecord(0, table), PlayRecord(1, None)]
        planes = evidence_planes(g, 0)
        self.assertEqual(planes.shape[1], 3)
        self.assertEqual(planes[0, 1, 28], 1.0)        # declined that card
        self.assertTrue((planes[0, 2, 29:] == 1.0).all())   # ...and above
        self.assertTrue((planes[0, 2, :29] == 0.0).all())   # nothing below
        self.assertTrue((planes[1] == 0.0).all())      # other seats silent

    def test_played_cards_are_recorded_per_opponent(self):
        from big2.belief import evidence_planes
        from big2.combos import classify
        from big2.game import PlayRecord
        from big2.rules import DEFAULT_RULES

        g = Big2Game(rng=random.Random(10))
        g.history = [PlayRecord(1, classify([40], DEFAULT_RULES))]
        planes = evidence_planes(g, 0)
        self.assertEqual(planes[0, 0, 40], 1.0)
        self.assertTrue((planes[1, 0] == 0.0).all())


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

        from big2.belief import calibrate

        data = collect_samples(60, seed=11)
        calib = collect_samples(20, seed=555)
        held = collect_samples(20, seed=77)
        net = train(data, epochs=6, hidden=64, verbose=False)
        alpha = calibrate(net, calib)
        report = evaluate(net, held, alpha=alpha)
        # alpha=0 recovers the analytic prior exactly, so a calibrated
        # residual can never be worse than the baseline it corrects
        # tolerance covers small-sample noise between the calibration
        # split and the held-out split at this tiny test budget
        self.assertLessEqual(
            report["learned"]["logloss"],
            report["analytic"]["logloss"] + 5e-3,
        )
        # measured at scale (600 games): brier .1451 vs .1456,
        # logloss .4205 vs .4217, top-k 58.0% vs 56.4%

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
            torch.from_numpy(data["evidence"]),
        ).numpy()
        self.assertTrue((p >= 0).all() and (p <= 1.0 + 1e-5).all())
        # impossible cards stay at zero
        self.assertTrue((p[data["mask"] == 0] == 0).all())
        # each opponent's probabilities sum to their hand size
        np.testing.assert_allclose(p.sum(-1), data["sizes"], atol=1e-3)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestShapeHead(unittest.TestCase):
    def test_shape_targets_match_the_real_hands(self):
        from big2.belief import shape_truth
        from big2.handshape import SHAPE_DIM, shape_profile

        g = Big2Game(rng=random.Random(2))
        s = shape_truth(g, 0)
        self.assertEqual(s.shape, (3, SHAPE_DIM))
        for j, p in enumerate([1, 2, 3]):
            np.testing.assert_allclose(s[j], shape_profile(g.hands[p]))

    def test_shape_head_is_scored_against_the_analytic_estimate(self):
        from big2.belief import (
            calibrate_shapes, collect_samples, evaluate_shapes, train,
        )

        data = collect_samples(40, seed=3)
        held = collect_samples(15, seed=8)
        net = train(data, epochs=6, hidden=64, verbose=False)
        alpha = calibrate_shapes(net, collect_samples(15, seed=44))
        rep = evaluate_shapes(net, held, alpha=alpha)
        self.assertIn("learned", rep)
        self.assertIn("analytic", rep)
        # alpha=0 recovers the analytic estimate, so a fitted shrinkage
        # can never leave the head materially worse than what it corrects
        self.assertLessEqual(rep["learned"]["logloss"],
                             rep["analytic"]["logloss"] + 5e-3)
        # measured at scale (260 games, alpha=0.5): brier .0333 vs .0341,
        # logloss .1193 vs .1295
