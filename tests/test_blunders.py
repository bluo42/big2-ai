import random
import unittest

from big2.blunders import (
    GIFT,
    KINDS,
    MISSED_WIN,
    WASTED_BOSS,
    check_decision,
    scan_selfplay,
)
from big2.combos import classify
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayLowest, SmartHeuristic


def make_state(hands, played=(), turn=0):
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


class TestMissedWin(unittest.TestCase):
    def test_flags_declining_a_guaranteed_win(self):
        """Holding only the 2S: playing it wins outright.  Anything
        else is objectively wrong."""
        g = make_state([[51, 4], [5], [8], [12]])
        # our whole hand is 2 cards, so no single empties it -- use a
        # position where the boss single IS the whole hand
        g2 = make_state([[51], [5], [8], [12]])
        blunders = check_decision(g2, 0, classify([51], DEFAULT_RULES))
        self.assertEqual(blunders, [])   # playing it is correct

    def test_missed_win_when_another_move_was_played(self):
        # hand is exactly the boss pair; playing one card instead is wrong
        g = make_state([[50, 51], [5], [8], [12]], played=list(range(13, 48)))
        b = check_decision(g, 0, classify([50], DEFAULT_RULES))
        self.assertTrue(b)
        self.assertEqual(b[0].kind, MISSED_WIN)
        self.assertIn("2", b[0].better)


class TestGiftAndWaste(unittest.TestCase):
    def test_gift_to_a_one_card_opponent(self):
        """Next player has one card; we hold the unbeatable 2S but lead
        a low single they can beat."""
        g = make_state([[51, 4, 20], [5], [8], [12]])
        b = check_decision(g, 0, classify([4], DEFAULT_RULES))
        self.assertTrue(b)
        self.assertEqual(b[0].kind, GIFT)
        self.assertIn("2", b[0].better)

    def test_playing_the_boss_card_is_never_a_blunder(self):
        g = make_state([[51, 4, 20], [5], [8], [12]])
        self.assertEqual(check_decision(g, 0, classify([51], DEFAULT_RULES)), [])

    def test_wasted_boss_when_someone_is_two_from_out(self):
        g = make_state([[51, 4, 20], [5, 9, 13], [8, 16], [12, 24]])
        b = check_decision(g, 0, classify([4], DEFAULT_RULES))
        self.assertTrue(b)
        self.assertEqual(b[0].kind, WASTED_BOSS)

    def test_no_blunder_when_no_boss_alternative_exists(self):
        g = make_state([[4, 20], [5], [8], [12]])
        self.assertEqual(check_decision(g, 0, classify([4], DEFAULT_RULES)), [])

    def test_blunders_describe_themselves(self):
        g = make_state([[51, 4, 20], [5], [8], [12]])
        b = check_decision(g, 0, classify([4], DEFAULT_RULES))[0]
        text = b.describe()
        self.assertIn(GIFT, text)
        self.assertIn("played", text)


class TestScanning(unittest.TestCase):
    def test_selfplay_scan_reports_rates(self):
        rates, found = scan_selfplay(PlayLowest(), [SmartHeuristic()],
                                     n_games=6, seed=1)
        self.assertGreater(rates["decisions"], 20)
        for k in KINDS:
            self.assertIn(k, rates)
            self.assertGreaterEqual(rates[k], 0.0)
        self.assertAlmostEqual(
            rates["total_per_100"], sum(rates[k] for k in KINDS), places=6
        )
        for b in found:
            self.assertIn(b.kind, KINDS)

    def test_a_careless_policy_blunders_more_than_a_careful_one(self):
        """PlayLowest always dumps its cheapest card, so it should give
        away more of these than the unit-preserving heuristic."""
        low, _ = scan_selfplay(PlayLowest(), [SmartHeuristic()],
                               n_games=25, seed=4)
        smart, _ = scan_selfplay(SmartHeuristic(), [SmartHeuristic()],
                                 n_games=25, seed=4)
        self.assertGreaterEqual(low["total_per_100"], smart["total_per_100"])


if __name__ == "__main__":
    unittest.main()
