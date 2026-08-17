import random
import unittest

from big2.features import (
    FEAT_DIM,
    FEAT_DIM_V3,
    encode_options,
    version_for_dim,
)
from big2.game import Big2Game
from big2.nn import MLPQ, NNPolicy
from big2.opponents import OpponentModel
from big2.strategies import PlayHighest, PlayLowest, RandomPolicy


class TestOpponentModel(unittest.TestCase):
    def _played_game(self, seed=0, steps=25):
        game = Big2Game(rng=random.Random(seed))
        policies = [PlayLowest(), PlayHighest(), RandomPolicy(seed),
                    PlayLowest()]
        for _ in range(steps):
            if game.game_over:
                break
            moves = game.legal_moves()
            game.step(policies[game.turn].select(game, game.turn)
                      if moves or game.can_pass() else None)
        return game

    def test_stats_bounded_and_consistent(self):
        game = self._played_game()
        model = OpponentModel(game, 0)
        for p in range(1, 4):
            for v in model.style(p):
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
            self.assertGreaterEqual(model.pass_honesty(p), 0.2)
            self.assertLessEqual(model.pass_honesty(p), 0.95)

    def test_honesty_tracks_pass_rate(self):
        game = self._played_game(seed=3, steps=30)
        model = OpponentModel(game, 0)
        rates = [(model.pass_rate(p), model.pass_honesty(p)) for p in range(1, 4)]
        rates.sort()
        # honesty is monotone in pass rate
        for (r1, h1), (r2, h2) in zip(rates, rates[1:]):
            if r2 > r1:
                self.assertGreaterEqual(h2, h1)


class TestEncodingVersions(unittest.TestCase):
    def test_version_dims(self):
        self.assertEqual(version_for_dim(FEAT_DIM_V3), 3)
        self.assertEqual(version_for_dim(FEAT_DIM), 4)
        with self.assertRaises(ValueError):
            version_for_dim(FEAT_DIM + 1)

    def test_v4_shapes_and_ranges(self):
        for n in (2, 4):
            game = Big2Game(num_players=n, rng=random.Random(1))
            options, feats = encode_options(game, game.turn, version=4)
            self.assertEqual(feats.shape, (len(options), FEAT_DIM))
            options3, feats3 = encode_options(game, game.turn, version=3)
            self.assertEqual(feats3.shape, (len(options3), FEAT_DIM_V3))
            # v3 block is a prefix-compatible slice of v4
            self.assertTrue((feats[:, :FEAT_DIM_V3] == feats3).all())

    def test_v3_and_v4_policies_both_play(self):
        for dim in (FEAT_DIM_V3, FEAT_DIM):
            policy = NNPolicy(MLPQ(dim, hidden=(16,), seed=1))
            game = Big2Game(rng=random.Random(2))
            scores = game.play_out([policy, PlayLowest(), PlayLowest(), PlayLowest()])
            self.assertEqual(sum(scores.values()), 0)


if __name__ == "__main__":
    unittest.main()
