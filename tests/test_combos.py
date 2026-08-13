import unittest

from big2.cards import parse_card
from big2.combos import ComboType, beats, classify, generate_moves
from big2.rules import RuleConfig


def cards(*names):
    return tuple(parse_card(n) for n in names)


def combo(*names):
    c = classify(cards(*names))
    assert c is not None, f"expected {names} to classify"
    return c


class TestClassification(unittest.TestCase):
    def test_single_and_pair(self):
        self.assertEqual(combo("3d").type, ComboType.SINGLE)
        self.assertEqual(combo("9h", "9s").type, ComboType.PAIR)
        self.assertIsNone(classify(cards("9h", "8s")))

    def test_no_lone_triple(self):
        self.assertIsNone(classify(cards("9h", "9s", "9d")))

    def test_five_card_types(self):
        self.assertEqual(combo("3d", "4c", "5h", "6s", "7d").type, ComboType.STRAIGHT)
        self.assertEqual(combo("3h", "6h", "9h", "Jh", "Ah").type, ComboType.FLUSH)
        self.assertEqual(
            combo("8d", "8c", "8h", "Ks", "Kd").type, ComboType.FULL_HOUSE
        )
        self.assertEqual(
            combo("8d", "8c", "8h", "8s", "3d").type, ComboType.FOUR_OF_A_KIND
        )
        self.assertEqual(
            combo("3h", "4h", "5h", "6h", "7h").type, ComboType.STRAIGHT_FLUSH
        )
        self.assertIsNone(classify(cards("3d", "4c", "5h", "6s", "8d")))

    def test_no_twos_or_wraparound_in_straights(self):
        self.assertIsNone(classify(cards("Jd", "Qc", "Kh", "As", "2d")))
        self.assertIsNone(classify(cards("Ad", "2c", "3h", "4s", "5d")))
        self.assertIsNone(classify(cards("Kd", "Ac", "3h", "4s", "5d")))
        # 10-J-Q-K-A is the top straight
        self.assertEqual(
            combo("10d", "Jc", "Qh", "Ks", "Ad").type, ComboType.STRAIGHT
        )


class TestComparisons(unittest.TestCase):
    def test_single_order(self):
        self.assertTrue(beats(combo("3c"), combo("3d")))  # clubs > diamonds
        self.assertTrue(beats(combo("2d"), combo("As")))  # 2 is the top rank
        self.assertFalse(beats(combo("3d"), combo("3c")))

    def test_pair_rank_then_suit(self):
        self.assertTrue(beats(combo("9d", "9s"), combo("9c", "9h")))  # spade tops
        self.assertTrue(beats(combo("10d", "10c"), combo("9h", "9s")))

    def test_straight_high_card_then_suit(self):
        low = combo("3d", "4c", "5h", "6s", "7c")
        high_suit = combo("3c", "4d", "5d", "6d", "7s")  # same top rank, spade 7
        higher = combo("4d", "5c", "6h", "7d", "8d")
        self.assertTrue(beats(high_suit, low))
        self.assertTrue(beats(higher, high_suit))

    def test_flush_suit_dominates_rank(self):
        club_flush = combo("3c", "5c", "9c", "Kc", "Ac")  # ace-high clubs
        heart_flush = combo("4h", "6h", "8h", "10h", "Jh")  # jack-high hearts
        self.assertTrue(beats(heart_flush, club_flush))
        # same suit -> highest card decides
        big_hearts = combo("3h", "5h", "9h", "Kh", "Ah")
        self.assertTrue(beats(big_hearts, heart_flush))

    def test_full_house_by_triple(self):
        kings_over_threes = combo("Kd", "Kc", "Kh", "3d", "3c")
        nines_over_aces = combo("9d", "9c", "9h", "Ad", "Ac")
        self.assertTrue(beats(kings_over_threes, nines_over_aces))

    def test_five_card_hierarchy(self):
        straight = combo("10d", "Jc", "Qh", "Ks", "Ad")
        flush = combo("3c", "5c", "9c", "Kc", "Ac")
        full = combo("4d", "4c", "4h", "5s", "5d")
        quads = combo("4d", "4c", "4h", "4s", "5d")
        sflush = combo("3h", "4h", "5h", "6h", "7h")
        self.assertTrue(beats(flush, straight))
        self.assertTrue(beats(full, flush))
        self.assertTrue(beats(quads, full))
        self.assertTrue(beats(sflush, quads))
        self.assertFalse(beats(straight, flush))

    def test_size_classes_never_cross(self):
        self.assertFalse(beats(combo("2s"), combo("3d", "3c")))
        self.assertFalse(
            beats(combo("4d", "4c", "4h", "4s", "5d"), combo("2s"))
        )


class TestRuleVariants(unittest.TestCase):
    TRIPLES = RuleConfig(allow_triples=True)

    def test_lone_triples_when_enabled(self):
        triple = classify(cards("9d", "9c", "9h"), self.TRIPLES)
        self.assertIsNotNone(triple)
        self.assertEqual(triple.type, ComboType.TRIPLE)
        self.assertIsNone(classify(cards("9d", "9c", "8h"), self.TRIPLES))
        self.assertIsNone(classify(cards("9d", "9c", "9h")))  # default rules

    def test_triple_comparison_and_isolation(self):
        low = classify(cards("9d", "9c", "9h"), self.TRIPLES)
        high = classify(cards("Kd", "Kc", "Ks"), self.TRIPLES)
        self.assertTrue(beats(high, low))
        self.assertFalse(beats(low, high))
        # triples never cross size classes
        self.assertFalse(beats(high, combo("2s", "2h")))

    def test_triple_move_generation(self):
        hand = cards("9d", "9c", "9h", "9s", "Kd")
        default_moves = generate_moves(hand)
        self.assertFalse(any(len(m) == 3 for m in default_moves))
        triple_moves = [
            m for m in generate_moves(hand, rules=self.TRIPLES) if len(m) == 3
        ]
        self.assertEqual(len(triple_moves), 4)  # C(4,3) ways to pick the 9s
        table = classify(cards("8d", "8c", "8h"), self.TRIPLES)
        beating = generate_moves(hand, table, self.TRIPLES)
        self.assertTrue(all(len(m) == 3 for m in beating))
        self.assertEqual(len(beating), 4)

    def test_flush_rank_first_variant(self):
        rank_first = RuleConfig(flush_rank_first=True)
        club_ace = cards("3c", "5c", "9c", "Kc", "Ac")
        heart_jack = cards("4h", "6h", "8h", "10h", "Jh")
        # house rule: suit dominates
        self.assertTrue(beats(classify(heart_jack), classify(club_ace)))
        # variant: rank dominates
        self.assertTrue(
            beats(classify(club_ace, rank_first), classify(heart_jack, rank_first))
        )


class TestMoveGeneration(unittest.TestCase):
    def test_generation_counts(self):
        hand = cards("3d", "3c", "3h", "4d", "5s", "6c", "7h", "8d")
        moves = generate_moves(hand)
        singles = [m for m in moves if len(m) == 1]
        pairs = [m for m in moves if len(m) == 2]
        fives = [m for m in moves if len(m) == 5]
        self.assertEqual(len(singles), 8)
        self.assertEqual(len(pairs), 3)  # 3 ways to pair the three 3s
        # straights 3-7 (3 choices of 3) and 4-8: 3 + 1
        self.assertEqual(len(fives), 4)
        self.assertTrue(all(m.type == ComboType.STRAIGHT for m in fives))

    def test_must_beat_filter(self):
        hand = cards("3d", "9c", "9h", "Kd", "2s")
        table = combo("Jc")
        moves = generate_moves(hand, table)
        self.assertEqual(
            {m.cards for m in moves}, {cards("Kd"), cards("2s")}
        )
        table_pair = combo("9d", "9s")
        moves = generate_moves(hand, table_pair)
        self.assertEqual(moves, [])  # 9c9h loses to 9d9s (spade)

    def test_sorted_weakest_first(self):
        hand = cards("3d", "5c", "5h", "2s", "2d")
        moves = generate_moves(hand)
        for a, b in zip(moves, moves[1:]):
            if len(a) == len(b):
                self.assertLess(a.sort_key, b.sort_key)
            else:
                self.assertLess(len(a), len(b))


if __name__ == "__main__":
    unittest.main()
