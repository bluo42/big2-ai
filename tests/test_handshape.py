import random
import unittest

import numpy as np

from big2.combos import ComboType
from big2.game import Big2Game
from big2.handshape import (
    ANY_BOMB,
    ANY_FIVE,
    ANY_PAIR,
    ANY_TRIPLE,
    FLUSH,
    FOUR_FLUSH,
    FULL_HOUSE_ANY,
    FULL_HOUSE_RANK,
    HOLDS_ABOVE,
    PAIR,
    QUAD,
    SHAPE_DIM,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPLE,
    beat_probability,
    profile_from_worlds,
    shape_profile,
)

# card index = rank * 4 + suit; suits 0..3 = D C H S
def card(r, s):
    return r * 4 + s


class TestShapeProfile(unittest.TestCase):
    def test_empty_hand_is_all_zeros(self):
        f = shape_profile([])
        self.assertEqual(f.shape, (SHAPE_DIM,))
        self.assertEqual(f.sum(), 0.0)

    def test_pairs_triples_and_quads_by_rank(self):
        # four 7s (rank 4) plus a lone 9 (rank 6)
        hand = [card(4, s) for s in range(4)] + [card(6, 0)]
        f = shape_profile(hand)
        self.assertEqual(f[PAIR.start + 4], 1.0)
        self.assertEqual(f[TRIPLE.start + 4], 1.0)
        self.assertEqual(f[QUAD.start + 4], 1.0)
        self.assertEqual(f[PAIR.start + 6], 0.0)   # a single is not a pair
        self.assertEqual(f[ANY_PAIR], 1.0)
        self.assertEqual(f[ANY_TRIPLE], 1.0)
        self.assertEqual(f[ANY_BOMB], 1.0)

    def test_flush_and_four_flush_are_distinguished(self):
        four = [card(r, 2) for r in range(4)]            # 4 hearts
        f = shape_profile(four)
        self.assertEqual(f[FOUR_FLUSH.start + 2], 1.0)
        self.assertEqual(f[FLUSH.start + 2], 0.0)
        five = four + [card(9, 2)]                        # 5 hearts
        g = shape_profile(five)
        self.assertEqual(g[FLUSH.start + 2], 1.0)
        self.assertEqual(g[FOUR_FLUSH.start + 2], 0.0)
        self.assertEqual(g[ANY_FIVE], 1.0)

    def test_straight_is_recorded_at_its_top_rank(self):
        # 3,4,5,6,7 across mixed suits -> straight topped at rank 4
        hand = [card(0, 0), card(1, 1), card(2, 2), card(3, 3), card(4, 0)]
        f = shape_profile(hand)
        self.assertEqual(f[STRAIGHT.start + 4], 1.0)
        self.assertEqual(f[STRAIGHT_FLUSH.start:STRAIGHT_FLUSH.stop].sum(), 0.0)
        self.assertEqual(f[ANY_FIVE], 1.0)

    def test_straight_flush_sets_both_straight_and_suit(self):
        hand = [card(r, 3) for r in range(5)]   # 3-7 all spades
        f = shape_profile(hand)
        self.assertEqual(f[STRAIGHT_FLUSH.start + 3], 1.0)
        self.assertEqual(f[STRAIGHT.start + 4], 1.0)
        self.assertEqual(f[FLUSH.start + 3], 1.0)

    def test_full_house_records_its_triple_rank(self):
        hand = [card(5, 0), card(5, 1), card(5, 2), card(8, 0), card(8, 1)]
        f = shape_profile(hand)
        self.assertEqual(f[FULL_HOUSE_ANY], 1.0)
        self.assertEqual(f[FULL_HOUSE_RANK.start + 5], 1.0)   # triple of 8s
        self.assertEqual(f[FULL_HOUSE_RANK.start + 8], 0.0)   # not the pair

    def test_holds_above_is_the_beat_a_single_curve(self):
        hand = [card(7, 0)]          # a single of rank 7
        f = shape_profile(hand)
        for r in range(7):
            self.assertEqual(f[HOLDS_ABOVE.start + r], 1.0)   # beats these
        for r in range(7, 13):
            self.assertEqual(f[HOLDS_ABOVE.start + r], 0.0)   # loses to these

    def test_profile_matches_a_real_dealt_hand(self):
        g = Big2Game(rng=random.Random(3))
        f = shape_profile(g.hands[0])
        self.assertTrue(np.isfinite(f).all())
        self.assertTrue(((f == 0) | (f == 1)).all())


class TestAnalyticProfile(unittest.TestCase):
    def test_weighted_average_over_worlds(self):
        pair_hand = [card(4, 0), card(4, 1)]
        no_pair = [card(4, 0), card(9, 1)]
        worlds = [({1: pair_hand}, 1.0), ({1: no_pair}, 1.0)]
        prof = profile_from_worlds(worlds, [1])
        self.assertAlmostEqual(prof[0][PAIR.start + 4], 0.5)
        self.assertAlmostEqual(prof[0][ANY_PAIR], 0.5)
        self.assertTrue((prof[1] == 0).all())   # unseated rows stay empty

    def test_weights_shift_the_estimate(self):
        pair_hand = [card(4, 0), card(4, 1)]
        no_pair = [card(4, 0), card(9, 1)]
        prof = profile_from_worlds(
            [({1: pair_hand}, 3.0), ({1: no_pair}, 1.0)], [1]
        )
        self.assertAlmostEqual(prof[0][PAIR.start + 4], 0.75)


class TestBeatProbability(unittest.TestCase):
    def test_single_reads_straight_off_the_curve(self):
        prof = np.zeros(SHAPE_DIM, dtype=np.float32)
        prof[HOLDS_ABOVE.start + 6] = 0.8
        self.assertAlmostEqual(
            beat_probability(prof, ComboType.SINGLE, 6), 0.8, places=5
        )

    def test_pair_combines_every_higher_rank(self):
        prof = np.zeros(SHAPE_DIM, dtype=np.float32)
        prof[PAIR.start + 7] = 0.5
        prof[PAIR.start + 9] = 0.5
        # 1 - (1-.5)(1-.5) = .75 across the two higher pairs
        self.assertAlmostEqual(
            beat_probability(prof, ComboType.PAIR, 5), 0.75, places=5
        )
        # nothing above rank 9 can answer
        self.assertAlmostEqual(
            beat_probability(prof, ComboType.PAIR, 9), 0.0, places=5
        )

    def test_five_card_uses_the_aggregate(self):
        prof = np.zeros(SHAPE_DIM, dtype=np.float32)
        prof[ANY_FIVE] = 0.4
        self.assertAlmostEqual(
            beat_probability(prof, ComboType.FLUSH, 8), 0.4, places=5
        )


if __name__ == "__main__":
    unittest.main()
