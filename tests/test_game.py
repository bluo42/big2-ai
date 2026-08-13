import random
import unittest

from big2.cards import THREE_OF_DIAMONDS, parse_card
from big2.game import NUM_PLAYERS, Big2Game, ScoringConfig
from big2.strategies import (
    FiveCardDumper,
    PlayHighest,
    PlayLowest,
    RandomPolicy,
    SmartHeuristic,
)


class TestScoring(unittest.TestCase):
    def c(self, *names):
        return [parse_card(n) for n in names]

    def test_base_payment(self):
        cfg = ScoringConfig(two_modifier=False, big_hand_modifier=False)
        self.assertEqual(cfg.payment(self.c("3d", "9h", "Kd")), 3)
        self.assertEqual(cfg.payment([]), 0)

    def test_two_modifier_doubles(self):
        cfg = ScoringConfig(two_modifier=True, big_hand_modifier=False)
        self.assertEqual(cfg.payment(self.c("3d", "9h", "2s")), 6)
        self.assertEqual(cfg.payment(self.c("3d", "9h", "Kd")), 3)

    def test_per_two_stacks(self):
        cfg = ScoringConfig(two_modifier=True, per_two=True, big_hand_modifier=False)
        self.assertEqual(cfg.payment(self.c("3d", "2h", "2s")), 9)  # 3 * (1+2)

    def test_big_hand_modifier(self):
        cfg = ScoringConfig(two_modifier=False, big_hand_modifier=True)
        ten = self.c("3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "Jd", "Qd")
        self.assertEqual(cfg.payment(ten), 20)
        self.assertEqual(cfg.payment(ten[:9]), 9)

    def test_modifiers_stack(self):
        cfg = ScoringConfig(two_modifier=True, big_hand_modifier=True)
        eleven = self.c(
            "3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "Jd", "Qd", "2s"
        )
        self.assertEqual(cfg.payment(eleven), 33)  # 11 * 3


class TestGameFlow(unittest.TestCase):
    def test_first_play_contains_three_of_diamonds(self):
        game = Big2Game(rng=random.Random(7))
        self.assertIn(THREE_OF_DIAMONDS, game.hands[game.turn])
        for move in game.legal_moves():
            self.assertIn(THREE_OF_DIAMONDS, move.cards)

    def test_leader_cannot_pass(self):
        game = Big2Game(rng=random.Random(7))
        self.assertFalse(game.can_pass())
        with self.assertRaises(ValueError):
            game.step(None)

    def test_pass_locks_out_for_trick(self):
        game = Big2Game(rng=random.Random(7))
        leader = game.turn
        game.step(game.legal_moves()[0])
        passer = game.turn
        game.step(None)  # passer sits out the rest of the trick
        while game.table_combo is not None and not game.game_over:
            self.assertNotEqual(game.turn, passer)
            moves = game.legal_moves()
            game.step(moves[0] if moves else None)

    def test_trick_winner_leads_next(self):
        game = Big2Game(rng=random.Random(11))
        first = game.legal_moves()[0]
        game.step(first)
        winner = (game.turn - 1) % NUM_PLAYERS
        for _ in range(NUM_PLAYERS - 1):
            game.step(None)
        self.assertEqual(game.turn, winner)
        self.assertIsNone(game.table_combo)
        self.assertEqual(game.passed, [False] * NUM_PLAYERS)

    def test_random_playouts_terminate_and_zero_sum(self):
        for seed in range(30):
            game = Big2Game(rng=random.Random(seed))
            policies = [RandomPolicy(seed + p) for p in range(NUM_PLAYERS)]
            scores = game.play_out(policies)
            self.assertEqual(len(game.hands[game.winner]), 0)
            self.assertEqual(sum(scores.values()), 0)
            self.assertGreaterEqual(scores[game.winner], 0)

    def test_all_baselines_play_full_games(self):
        lineups = [
            [PlayLowest(), PlayHighest(), FiveCardDumper(), SmartHeuristic()],
            [SmartHeuristic(), SmartHeuristic(), SmartHeuristic(), SmartHeuristic()],
        ]
        for lineup in lineups:
            for seed in range(10):
                game = Big2Game(rng=random.Random(seed))
                scores = game.play_out(lineup)
                self.assertEqual(sum(scores.values()), 0)


if __name__ == "__main__":
    unittest.main()
