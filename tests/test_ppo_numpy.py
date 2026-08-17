import os
import random
import tempfile
import unittest

import numpy as np

from big2.game import Big2Game
from big2.neural import encode_decision
from big2.strategies import PlayLowest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestNumpyEquivalence(unittest.TestCase):
    def _make_pair(self, d):
        import torch

        from big2.neural import build_net
        from big2.ppo_numpy import NumpyPPOPolicy, export_numpy

        net = build_net(d_model=64, heads=4)
        pt = os.path.join(d, "net.pt")
        torch.save({"state_dict": net.state_dict(), "d_model": 64, "heads": 4}, pt)
        npz = os.path.join(d, "net.npz")
        export_numpy(pt, npz)
        from big2.neural import PPOPolicy

        return PPOPolicy(net), NumpyPPOPolicy.load(npz)

    def test_logits_match_and_same_moves(self):
        import torch

        with tempfile.TemporaryDirectory() as d:
            torch_policy, np_policy = self._make_pair(d)
            rng = random.Random(0)
            checked = 0
            for seed in range(4):
                game = Big2Game(rng=random.Random(seed))
                driver = PlayLowest()
                while not game.game_over and checked < 40:
                    p = game.turn
                    options, state, acts = encode_decision(game, p)
                    if len(options) > 1:
                        with torch.no_grad():
                            t_logits, _, _ = torch_policy.net(
                                torch.from_numpy(state).unsqueeze(0),
                                torch.from_numpy(acts).unsqueeze(0),
                                torch.ones(1, len(options), dtype=torch.bool),
                            )
                        n_logits = np_policy._logits(
                            state.astype(np.float64), acts.astype(np.float64)
                        )
                        np.testing.assert_allclose(
                            t_logits[0].numpy(), n_logits, atol=1e-4
                        )
                        self.assertEqual(
                            int(t_logits[0].argmax()), int(np.argmax(n_logits))
                        )
                        checked += 1
                    game.step(driver.select(game, p))
            self.assertGreater(checked, 20)

    def test_numpy_policy_plays_full_game(self):
        with tempfile.TemporaryDirectory() as d:
            _, np_policy = self._make_pair(d)
            game = Big2Game(rng=random.Random(1))
            scores = game.play_out(
                [np_policy, PlayLowest(), PlayLowest(), PlayLowest()]
            )
            self.assertEqual(sum(scores.values()), 0)


if __name__ == "__main__":
    unittest.main()
