import random
import unittest

from big2.combos import classify
from big2.game import Big2Game, ScoringConfig
from big2.planning import (
    PLAN_DIM,
    PLAN_STATE_DIM,
    PlanContext,
    beatable,
    boss_singles,
    plan_features,
    plan_state_features,
)
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayLowest


def make_state(hands, turn=0, played=()):
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=len(hands), rng=random.Random(0))
    g.hands = [sorted(h) for h in hands]
    g.turn = turn
    g.first_play = False
    g.table_combo = None
    g.table_player = None
    g.passed = [False] * len(hands)
    g.history = []
    g.played_cards = list(played)
    g.winner = None
    g.scores = None
    return g


class TestBeatable(unittest.TestCase):
    def test_singles_exact(self):
        g = make_state([[51, 4], [5], [8], [12]])
        pool = [5, 8, 12]
        self.assertFalse(beatable(g, classify([51], DEFAULT_RULES), pool))
        self.assertTrue(beatable(g, classify([4], DEFAULT_RULES), pool))

    def test_pairs_exact(self):
        g = make_state([[48, 49], [4, 5], [8], [12]])
        # our 2D-2C pair; pool holds 4D-4C, which cannot outrank twos
        self.assertFalse(
            beatable(g, classify([48, 49], DEFAULT_RULES), [4, 5, 8, 12])
        )
        # a lower pair of ours is beatable by the pool's higher pair
        self.assertTrue(
            beatable(g, classify([4, 5], DEFAULT_RULES), [48, 49, 8, 12])
        )

    def test_five_card_unknown_when_pool_is_large(self):
        g = Big2Game(rng=random.Random(2))
        moves = [m for m in g.legal_moves() if len(m) == 5]
        if moves:
            pool = [c for c in range(52) if c not in g.hands[g.turn]]
            self.assertIsNone(beatable(g, moves[0], pool))


class TestPlanFeatures(unittest.TestCase):
    def test_shapes(self):
        g = make_state([[51, 4], [5], [8], [12]])
        ctx = PlanContext(g, 0)
        self.assertEqual(plan_state_features(ctx).shape, (PLAN_STATE_DIM,))
        self.assertEqual(
            plan_features(ctx, classify([51], DEFAULT_RULES)).shape, (PLAN_DIM,)
        )
        self.assertEqual(plan_features(ctx, None).shape, (PLAN_DIM,))

    def test_boss_and_waste_flags(self):
        """2S is boss and keeps the lead; the 4D is answerable and
        spending it while holding a boss is the flagged mistake."""
        g = make_state([[51, 4], [5], [8], [12]])
        ctx = PlanContext(g, 0)
        boss = plan_features(ctx, classify([51], DEFAULT_RULES))
        weak = plan_features(ctx, classify([4], DEFAULT_RULES))
        self.assertEqual(boss[2], 1.0)      # boss
        self.assertEqual(boss[6], 1.0)      # boss, keeps the lead
        self.assertEqual(boss[7], 0.0)      # wastes nothing
        self.assertEqual(weak[2], 0.0)      # answerable
        self.assertEqual(weak[7], 1.0)      # wastes control: the bug
        self.assertEqual(weak[13], 1.0)     # ...gifted to a 1-card player

    def test_winning_move_flags(self):
        g = make_state([[51], [5], [8], [12]])
        ctx = PlanContext(g, 0)
        f = plan_features(ctx, classify([51], DEFAULT_RULES))
        self.assertEqual(f[4], 1.0)   # empties the hand
        self.assertEqual(f[5], 1.0)   # and cannot be answered
        self.assertEqual(f[9], 1.0)   # zero units left

    def test_run_out_state_flag(self):
        """Hand of nothing but boss units while leading = run it out."""
        g = make_state([[50, 51], [4], [8], [12]], played=list(range(13, 48)))
        ctx = PlanContext(g, 0)
        s = plan_state_features(ctx)
        self.assertEqual(s[0], 1.0)   # every unit is boss
        self.assertEqual(s[3], 1.0)   # can run out from the lead

    def test_boss_singles_helper(self):
        g = make_state([[51, 4], [5], [8], [12]])
        self.assertEqual(boss_singles(g, 0), [51])

    def test_features_are_finite_in_real_games(self):
        import numpy as np

        g = Big2Game(rng=random.Random(7))
        steps = 0
        while not g.game_over and steps < 40:
            p = g.turn
            ctx = PlanContext(g, p)
            opts = list(g.legal_moves())
            if g.can_pass():
                opts.append(None)
            for m in opts:
                f = plan_features(ctx, m)
                self.assertTrue(np.isfinite(f).all())
                self.assertTrue((np.abs(f) <= 1.0 + 1e-6).all())
            self.assertTrue(np.isfinite(plan_state_features(ctx)).all())
            g.step(PlayLowest().select(g, p))
            steps += 1


if __name__ == "__main__":
    unittest.main()
