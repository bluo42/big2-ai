import random
import unittest

import numpy as np

from big2.game import Big2Game
from big2.neural import ACT_DIM, BELIEF_SLOTS, belief_target, encode_decision
from big2.features import FEAT_DIM

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestFeatureAssembly(unittest.TestCase):
    def test_encode_decision_shapes(self):
        game = Big2Game(rng=random.Random(0))
        options, state, acts = encode_decision(game, game.turn)
        self.assertEqual(state.shape, (FEAT_DIM,))
        self.assertEqual(acts.shape, (len(options), ACT_DIM))
        self.assertGreaterEqual(len(options), 1)

    def test_belief_target_counts_opponent_cards(self):
        game = Big2Game(rng=random.Random(1))
        t = belief_target(game, game.turn)
        self.assertEqual(t.shape, (BELIEF_SLOTS,))
        self.assertEqual(t.sum(), 39.0)  # three hidden 13-card hands


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestNetAndPPO(unittest.TestCase):
    def test_forward_shapes_and_mask(self):
        import torch

        from big2.neural import build_net

        net = build_net(d_model=32, heads=2)
        B, A = 3, 7
        state = torch.randn(B, FEAT_DIM)
        acts = torch.randn(B, A, ACT_DIM)
        mask = torch.ones(B, A, dtype=torch.bool)
        mask[0, 4:] = False
        logits, value, belief = net(state, acts, mask)
        self.assertEqual(logits.shape, (B, A))
        self.assertEqual(value.shape, (B,))
        self.assertEqual(belief.shape, (B, BELIEF_SLOTS))
        self.assertTrue((logits[0, 4:] < -1e8).all())  # padded = impossible

    def test_policy_plays_full_game(self):
        from big2.neural import PPOPolicy, build_net
        from big2.strategies import PlayLowest

        policy = PPOPolicy(build_net(d_model=32, heads=2))
        game = Big2Game(rng=random.Random(2))
        scores = game.play_out([policy, PlayLowest(), PlayLowest(), PlayLowest()])
        self.assertEqual(sum(scores.values()), 0)

    def test_gae_terminal_only(self):
        from big2.neural import _gae

        adv, ret = _gae([0.1, 0.2, 0.3], final_return=1.0, lam=1.0)
        # lambda=1: every return equals the terminal outcome
        np.testing.assert_allclose(ret, [1.0, 1.0, 1.0], atol=1e-6)

    def test_tiny_training_iteration_with_snapshots(self):
        import os
        import tempfile

        from big2.neural import train_ppo

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "ppo_test.pt")
            net = train_ppo(
                iters=2, games_per_iter=4, workers=2, selfplay_prob=0.5,
                minibatch=64, probe_every_iters=0, verbose=False, seed=0,
                out=out, snapshot_every_iters=1, past_self_prob=0.9,
            )
            self.assertIsNotNone(net)
            snaps = os.listdir(os.path.join(d, "ppo_snapshots"))
            self.assertTrue(any(f.endswith(".pt") for f in snaps))


if __name__ == "__main__":
    unittest.main()
