import random
import unittest

import numpy as np

from big2.combos import ComboType, classify
from big2.game import Big2Game, ScoringConfig
from big2.handshape import ANY_FIVE, HOLDS_ABOVE, PAIR, SHAPE_DIM
from big2.lookahead import (
    lookahead_value,
    lookahead_values,
    opponent_profiles,
    survival_probability,
)
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayLowest


def make_state(hands, played=()):
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=len(hands), rng=random.Random(0))
    g.hands = [sorted(h) for h in hands]
    g.turn = 0
    g.first_play = False
    g.table_combo = None
    g.table_player = None
    g.passed = [False] * len(hands)
    g.history = []
    g.played_cards = list(played)
    g.winner = None
    g.scores = None
    return g


class TestSurvival(unittest.TestCase):
    def test_unbeatable_card_always_survives(self):
        g = make_state([[51, 4], [5], [8], [12]])
        prof = np.zeros((3, SHAPE_DIM), dtype=np.float32)
        # nobody holds anything above the 2 of spades, by construction
        s = survival_probability(g, 0, classify([51], DEFAULT_RULES), prof)
        self.assertAlmostEqual(s, 1.0)

    def test_probability_falls_as_opponents_get_stronger(self):
        g = make_state([[20, 4], [5], [8], [12]])
        weak = np.zeros((3, SHAPE_DIM), dtype=np.float32)
        strong = np.zeros((3, SHAPE_DIM), dtype=np.float32)
        for j in range(3):
            strong[j, HOLDS_ABOVE.start + 5] = 0.9
        move = classify([20], DEFAULT_RULES)   # rank 5 single
        self.assertGreater(
            survival_probability(g, 0, move, weak),
            survival_probability(g, 0, move, strong),
        )

    def test_independent_opponents_compound(self):
        g = make_state([[20, 4], [5], [8], [12]])
        prof = np.zeros((3, SHAPE_DIM), dtype=np.float32)
        for j in range(3):
            prof[j, HOLDS_ABOVE.start + 5] = 0.5
        s = survival_probability(g, 0, classify([20], DEFAULT_RULES), prof)
        self.assertAlmostEqual(s, 0.125, places=5)   # 0.5^3 survive


class TestProfiles(unittest.TestCase):
    def test_profiles_are_probabilities_of_the_right_shape(self):
        g = Big2Game(rng=random.Random(2))
        for _ in range(8):
            g.step(PlayLowest().select(g, g.turn))
        prof = opponent_profiles(g, 0, k=12, rng=random.Random(1))
        self.assertEqual(prof.shape, (3, SHAPE_DIM))
        self.assertTrue(((prof >= 0) & (prof <= 1.0 + 1e-6)).all())


class TestLookahead(unittest.TestCase):
    def test_small_positions_fall_through_to_the_exact_solver(self):
        """The boss spot: the coarse tree must still get it right,
        because it hands off to the solver."""
        g = make_state([[51, 4], [5], [8], [12]])
        prof = np.zeros((3, SHAPE_DIM), dtype=np.float32)
        boss = lookahead_value(g, 0, classify([51], DEFAULT_RULES), prof)
        weak = lookahead_value(g, 0, classify([4], DEFAULT_RULES), prof)
        self.assertGreater(boss, weak)

    def test_values_every_option_and_keys_like_the_solver(self):
        g = Big2Game(rng=random.Random(5))
        for _ in range(10):
            g.step(PlayLowest().select(g, g.turn))
        p = g.turn
        opts = list(g.legal_moves(p))
        if g.can_pass():
            opts.append(None)
        vals = lookahead_values(g, p, opts, depth=2, k_worlds=8,
                                rng=random.Random(0))
        self.assertEqual(len(vals), len(opts))
        for m in opts:
            key = None if m is None else tuple(m.cards)
            self.assertIn(key, vals)
            self.assertTrue(np.isfinite(vals[key]))

    def test_depth_is_a_budget_not_a_correctness_knob(self):
        g = Big2Game(rng=random.Random(7))
        for _ in range(12):
            g.step(PlayLowest().select(g, g.turn))
        p = g.turn
        opts = list(g.legal_moves(p))[:4]
        prof = opponent_profiles(g, p, k=8, rng=random.Random(0))
        for d in (1, 2, 3):
            vals = [lookahead_value(g, p, m, prof, depth=d) for m in opts]
            self.assertTrue(all(np.isfinite(v) for v in vals))

    def test_a_learned_value_head_is_used_when_supplied(self):
        g = Big2Game(rng=random.Random(9))
        for _ in range(10):
            g.step(PlayLowest().select(g, g.turn))
        # walk on until the player to move actually has a choice
        while not g.game_over and not g.legal_moves(g.turn):
            g.step(None)
        p = g.turn
        move = g.legal_moves(p)[0]
        prof = opponent_profiles(g, p, k=6, rng=random.Random(0))
        calls = []

        def fake_value(sim, who):
            calls.append(who)
            return 4.0

        v = lookahead_value(g, p, move, prof, depth=1, value_fn=fake_value)
        self.assertTrue(calls)
        self.assertAlmostEqual(v, 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
