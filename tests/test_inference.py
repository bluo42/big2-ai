import random
import unittest

from big2.cards import NUM_CARDS
from big2.combos import classify
from big2.game import Big2Game, PlayRecord, ScoringConfig
from big2.inference import OVERPLAY_DIM, InferenceState, overplay_features
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayLowest


def rec(player, cards, trick_end=False):
    combo = None if cards is None else classify(cards, DEFAULT_RULES)
    return PlayRecord(player, combo, trick_end)


def game_with_history(hands, history, turn=0, played=()):
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=4, rng=random.Random(0))
    g.hands = [sorted(h) for h in hands]
    g.turn = turn
    g.first_play = False
    g.table_combo = None
    g.table_player = None
    g.passed = [False] * 4
    g.history = history
    g.played_cards = list(played)
    g.winner = None
    g.scores = None
    return g


class TestOverplayFeatures(unittest.TestCase):
    def test_detects_big_jumps_versus_minimum_answers(self):
        # seat 1 answers a 3D (card 0) with the 2S (card 51): huge jump.
        # seat 2 answers a 3D with the 3C (card 1): minimum winner.
        hist = [
            rec(0, [4]), rec(1, [51], True),
            rec(0, [0]), rec(2, [1], True),
        ]
        g = game_with_history([[8], [12], [16], [20]], hist)
        f = overplay_features(g, 0)
        self.assertEqual(len(f[1]), OVERPLAY_DIM)
        self.assertEqual(f[1][0], 1.0)      # seat 1 always overplays
        self.assertGreater(f[1][1], 0.5)    # by a large rank gap
        self.assertEqual(f[2][0], 0.0)      # seat 2 never overplays
        self.assertEqual(f[2][2], 1.0)      # ...it plays the minimum winner

    def test_confidence_grows_with_observed_answers(self):
        few = game_with_history(
            [[8], [12], [16], [20]], [rec(0, [4]), rec(1, [51], True)]
        )
        self.assertLess(overplay_features(few, 0)[1][5], 0.2)
        many_hist = []
        for _ in range(4):
            many_hist += [rec(0, [4]), rec(1, [51], True)]
        many = game_with_history([[8], [12], [16], [20]], many_hist)
        self.assertEqual(overplay_features(many, 0)[1][5], 0.5)

    def test_no_history_is_all_zeros(self):
        g = Big2Game(rng=random.Random(1))
        f = overplay_features(g, 0)
        self.assertEqual(len(f), 3)
        for row in f.values():
            self.assertEqual(row, [0.0] * OVERPLAY_DIM)


class TestInferenceState(unittest.TestCase):
    def test_overplay_downweights_worlds_with_a_cheap_loose_winner(self):
        """Seat 1 answered a 10D (card 28) with the 2S.  A world where
        they held a loose KD (a cheap winner they skipped) is less likely
        than one where every card they held was too low to answer."""
        hist = [rec(0, [28]), rec(1, [51], True)]
        g = game_with_history([[8, 12], [40, 44], [24, 32], [36, 48]], hist,
                              played=[28, 51])
        inf = InferenceState(g, 0, skip_honesty=0.4, rng=random.Random(0))
        skipped = {1: [40, 44], 2: [24, 32], 3: [36, 48]}   # loose KD/AD
        clean = {1: [4, 20], 2: [24, 32], 3: [36, 48]}      # nothing wins
        self.assertEqual(inf._world_weight(clean), 1.0)
        self.assertLess(inf._world_weight(skipped), inf._world_weight(clean))

    def test_skip_honesty_one_disables_the_signal(self):
        hist = [rec(0, [28]), rec(1, [51], True)]
        g = game_with_history([[8, 12], [40, 44], [24, 32], [36, 48]], hist,
                              played=[28, 51])
        inf = InferenceState(g, 0, skip_honesty=1.0, rng=random.Random(0))
        skipped = {1: [40, 44], 2: [24, 32], 3: [36, 48]}
        self.assertEqual(inf._skip_weight(skipped), 1.0)

    def test_cheap_winner_locked_in_a_pair_is_not_evidence(self):
        """Skipping a cheap card that would break a pair is good play,
        not information — that world keeps full weight."""
        hist = [rec(0, [28]), rec(1, [51], True)]
        g = game_with_history([[8, 12], [40, 41], [24, 32], [36, 48]], hist,
                              played=[28, 51])
        inf = InferenceState(g, 0, skip_honesty=0.4, rng=random.Random(0))
        paired = {1: [40, 41], 2: [24, 32], 3: [36, 48]}  # KD-KC pair
        loose = {1: [40, 44], 2: [24, 32], 3: [36, 48]}
        self.assertEqual(inf._skip_weight(paired), 1.0)
        self.assertLess(inf._skip_weight(loose), 1.0)

    def test_card_marginals_are_probabilities_summing_to_hand_size(self):
        g = Big2Game(rng=random.Random(3))
        for _ in range(8):
            g.step(PlayLowest().select(g, g.turn))
        inf = InferenceState(g, 0, rng=random.Random(0))
        marg = inf.card_marginals(k=60)
        for p, row in marg.items():
            self.assertEqual(len(row), NUM_CARDS)
            self.assertTrue(all(0.0 <= v <= 1.0 + 1e-9 for v in row))
            self.assertAlmostEqual(sum(row), len(g.hands[p]), places=4)
            # our own cards are never in an opponent's posterior
            for c in g.hands[0]:
                self.assertEqual(row[c], 0.0)

    def test_worlds_for_search_are_positive_and_sorted(self):
        g = Big2Game(rng=random.Random(5))
        for _ in range(10):
            g.step(PlayLowest().select(g, g.turn))
        inf = InferenceState(g, 0, rng=random.Random(1))
        worlds = inf.worlds_for_search(k=30, top=8)
        self.assertLessEqual(len(worlds), 8)
        self.assertTrue(all(w > 0 for _, w in worlds))
        weights = [w for _, w in worlds]
        self.assertEqual(weights, sorted(weights, reverse=True))
        for world, _ in worlds:
            for p, hand in world.items():
                self.assertEqual(len(hand), len(g.hands[p]))


if __name__ == "__main__":
    unittest.main()
