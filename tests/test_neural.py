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
        from big2.neural import STATE_DIM

        game = Big2Game(rng=random.Random(0))
        options, state, acts = encode_decision(game, game.turn)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(acts.shape, (len(options), ACT_DIM))
        options2, state2, _ = encode_decision(
            game, game.turn, include_profiles=False
        )
        self.assertEqual(state2.shape, (FEAT_DIM,))
        self.assertGreaterEqual(len(options), 1)

    def test_profile_book_ema_weights_recent_without_forgetting(self):
        from big2.profiles import PROFILE_DIM, OpponentProfileBook
        from big2.strategies import PlayLowest

        book = OpponentProfileBook(half_life_games=3)
        for seed in range(6):
            game = Big2Game(rng=random.Random(seed))
            game.play_out([PlayLowest()] * 4)
            book.observe_game(game, {1: "opp", 2: "other"})
        # nothing is ever eliminated: raw count keeps growing
        self.assertEqual(book.games_seen("opp"), 6)
        f = book.features("opp")
        self.assertEqual(f.shape, (PROFILE_DIM,))
        self.assertTrue(((f >= 0.0) & (f <= 1.0)).all())
        # confidence = 1 - decay^n: exactly 0.5 after one half-life
        book2 = OpponentProfileBook(half_life_games=3)
        for seed in range(3):
            game = Big2Game(rng=random.Random(seed))
            game.play_out([PlayLowest()] * 4)
            book2.observe_game(game, {1: "opp"})
        self.assertAlmostEqual(book2.features("opp")[0], 0.5, places=5)
        self.assertTrue((book.features("stranger") == 0).all())

    def test_belief_target_counts_opponent_cards(self):
        game = Big2Game(rng=random.Random(1))
        t = belief_target(game, game.turn)
        self.assertEqual(t.shape, (BELIEF_SLOTS,))
        self.assertEqual(t.sum(), 39.0)  # three hidden 13-card hands

    def test_danger_features_flag_near_winners(self):
        from big2.combos import classify
        from big2.neural import DANGER_DIM, danger_features
        from big2.rules import DEFAULT_RULES

        game = Big2Game(rng=random.Random(4))
        p = game.turn
        nxt = (p + 1) % game.num_players
        f = danger_features(game, p, None)
        self.assertEqual(f.shape, (DANGER_DIM,))
        # everyone still holds 13 cards: no alarms
        self.assertEqual(f[0], 0.0)
        self.assertEqual(f[2], 0.0)
        self.assertAlmostEqual(f[3], 1.0)
        self.assertEqual(f[4], 0.0)
        # next player down to one card: alarms fire
        game.hands[nxt] = game.hands[nxt][:1]
        f = danger_features(game, p, None)
        self.assertEqual(f[0], 1.0)
        self.assertEqual(f[2], 1.0)
        self.assertEqual(f[4], 1.0)
        self.assertEqual(f[6], 1.0)
        # a cheap single gifted to the near-winner scores hotter than a
        # top single
        lo = classify((min(game.hands[p]),), DEFAULT_RULES)
        hi = classify((max(game.hands[p]),), DEFAULT_RULES)
        f_lo = danger_features(game, p, lo)
        f_hi = danger_features(game, p, hi)
        self.assertGreater(f_lo[10], f_hi[10])
        self.assertGreater(f_lo[11], 0.0)
        self.assertEqual(f_lo[13], 1.0)  # our single: they can go out on it
        # encode_decision widens rows by exactly the danger block
        from big2.neural import ACT_DIM_V1

        _, _, acts = encode_decision(game, p)
        _, _, acts_v1 = encode_decision(game, p, include_danger=False)
        self.assertEqual(acts.shape[1], ACT_DIM)
        self.assertEqual(acts_v1.shape[1], ACT_DIM_V1)
        np.testing.assert_allclose(acts[:, :ACT_DIM_V1], acts_v1)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestNetAndPPO(unittest.TestCase):
    def test_forward_shapes_and_mask(self):
        import torch

        from big2.neural import STATE_DIM, build_net

        net = build_net(d_model=32, heads=2)
        B, A = 3, 7
        state = torch.randn(B, STATE_DIM)
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

    def test_widened_checkpoint_matches_old_net(self):
        import torch

        from big2.features import FEAT_DIM as OLD_DIM
        from big2.neural import (
            ACT_DIM_V1, STATE_DIM, build_net, widen_state_dict,
        )

        old = build_net(d_model=32, heads=2, state_dim=OLD_DIM,
                        act_dim=ACT_DIM_V1)
        wide = build_net(d_model=32, heads=2, state_dim=STATE_DIM,
                         act_dim=ACT_DIM)
        wide.load_state_dict(widen_state_dict(old.state_dict()))
        B, A = 2, 5
        state_old = torch.randn(B, OLD_DIM)
        state_wide = torch.cat(
            [state_old, torch.zeros(B, STATE_DIM - OLD_DIM)], dim=1
        )
        acts_old = torch.randn(B, A, ACT_DIM_V1)
        acts_wide = torch.cat(
            [acts_old, torch.zeros(B, A, ACT_DIM - ACT_DIM_V1)], dim=2
        )
        mask = torch.ones(B, A, dtype=torch.bool)
        with torch.no_grad():
            lo, vo, _ = old(state_old, acts_old, mask)
            lw, vw, _ = wide(state_wide, acts_wide, mask)
        torch.testing.assert_close(lo, lw, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(vo, vw, atol=1e-5, rtol=1e-5)

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
