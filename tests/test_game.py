import random
import unittest

from big2.cards import THREE_OF_DIAMONDS, parse_card
from big2.game import NUM_PLAYERS, Big2Game, ScoringConfig
from big2.rules import RuleConfig
from big2.strategies import (
    FiveCardDumper,
    PlayHighest,
    PlayLowest,
    RandomPolicy,
    SmartHeuristic,
)


def cardlist(*names):
    return sorted(parse_card(n) for n in names)


class TestScoring(unittest.TestCase):
    def c(self, *names):
        return [parse_card(n) for n in names]

    def hand_of(self, n):
        """First n cards of a diamond-heavy 13-card hand (no 2s)."""
        names = ["3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d",
                 "Jd", "Qd", "Kd", "Ad", "3c"]
        return self.c(*names[:n])

    def test_base_payment(self):
        cfg = ScoringConfig(big_hand_double=False, full_hand_triple=False)
        self.assertEqual(cfg.payment(self.c("3d", "9h", "Kd")), 3)
        self.assertEqual(cfg.payment(self.hand_of(13)), 13)
        self.assertEqual(cfg.payment([]), 0)

    def test_tiered_multipliers_default(self):
        cfg = ScoringConfig()  # house rule
        self.assertEqual(cfg.payment(self.hand_of(9)), 9)    # 1x
        self.assertEqual(cfg.payment(self.hand_of(10)), 20)  # 2x
        self.assertEqual(cfg.payment(self.hand_of(11)), 22)  # 2x
        self.assertEqual(cfg.payment(self.hand_of(12)), 24)  # 2x
        self.assertEqual(cfg.payment(self.hand_of(13)), 39)  # 3x

    def test_two_modifier_off_by_default(self):
        self.assertEqual(ScoringConfig().payment(self.c("3d", "9h", "2s")), 3)

    def test_two_modifier_doubles(self):
        cfg = ScoringConfig(
            big_hand_double=False, full_hand_triple=False, two_modifier=True
        )
        self.assertEqual(cfg.payment(self.c("3d", "9h", "2s")), 6)
        self.assertEqual(cfg.payment(self.c("3d", "9h", "Kd")), 3)

    def test_per_two_stacks(self):
        cfg = ScoringConfig(
            big_hand_double=False, full_hand_triple=False,
            two_modifier=True, per_two=True,
        )
        self.assertEqual(cfg.payment(self.c("3d", "2h", "2s")), 9)  # 3 * (1+2)

    def test_tiered_plus_two_stack(self):
        cfg = ScoringConfig(two_modifier=True)
        eleven = self.hand_of(10) + self.c("2s")
        self.assertEqual(cfg.payment(eleven), 33)  # 11 * (2 + 1)


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
        # lock-out is now the variant, not the default
        game = Big2Game(rules=RuleConfig(pass_locks=True), rng=random.Random(7))
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


class TestVariants(unittest.TestCase):
    def _mid_trick_game(self, rules):
        """P0 leads 4d, P1 passes, P2 plays 5d, P3 passes, P0 passes."""
        game = Big2Game(rules=rules, rng=random.Random(0))
        game.hands = [
            cardlist("4d", "3h"),
            cardlist("6d", "6h"),
            cardlist("5d", "9h"),
            cardlist("3s", "8h"),
        ]
        game.turn = 0
        game.first_play = False
        game.step(game.legal_moves()[0])  # P0: 4d
        game.step(None)  # P1 passes
        game.step([m for m in game.legal_moves() if len(m) == 1][0])  # P2: 5d
        game.step(None)  # P3 passes
        game.step(None)  # P0 passes
        return game

    def test_pass_lock_skips_passer(self):
        game = self._mid_trick_game(RuleConfig(pass_locks=True))
        # P1 is locked out, so the trick ends and P2 leads fresh.
        self.assertEqual(game.turn, 2)
        self.assertIsNone(game.table_combo)

    def test_soft_pass_lets_passer_back_in(self):
        game = self._mid_trick_game(RuleConfig(pass_locks=False))
        # P1 passed earlier but P2's play reopened the trick.
        self.assertEqual(game.turn, 1)
        self.assertIsNotNone(game.table_combo)
        self.assertTrue(game.legal_moves())  # 6d/6h beat the 5d

    def test_triples_playable_in_game(self):
        rules = RuleConfig(allow_triples=True)
        for seed in range(10):
            game = Big2Game(rules=rules, rng=random.Random(seed))
            scores = game.play_out([RandomPolicy(seed + p) for p in range(4)])
            self.assertEqual(sum(scores.values()), 0)

    def test_fewer_players(self):
        for n in (2, 3):
            for seed in range(10):
                game = Big2Game(num_players=n, rng=random.Random(seed))
                self.assertEqual(len(game.hands), n)
                self.assertTrue(all(len(h) == 13 for h in game.hands))
                start = game.start_card
                self.assertEqual(start, min(min(h) for h in game.hands))
                self.assertIn(start, game.hands[game.turn])
                for move in game.legal_moves():
                    self.assertIn(start, move.cards)
                scores = game.play_out([RandomPolicy(seed + p) for p in range(n)])
                self.assertEqual(sum(scores.values()), 0)


if __name__ == "__main__":
    unittest.main()
