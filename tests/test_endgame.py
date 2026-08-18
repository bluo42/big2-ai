import random
import unittest

from big2.combos import classify
from big2.endgame import (
    boss_chain,
    boss_move,
    move_key,
    pimc_move_values,
    solve,
    solve_move_values,
    unbeatable_probability,
)
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES


def make_state(hands, turn=0, table=None, played=()):
    """A hand-built position: hands is a list of card lists."""
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=len(hands), rng=random.Random(0))
    g.hands = [sorted(h) for h in hands]
    g.turn = turn
    g.first_play = False
    g.table_combo = None if table is None else classify(table, DEFAULT_RULES)
    g.table_player = None if table is None else (turn - 1) % len(hands)
    g.passed = [False] * len(hands)
    g.history = []
    g.played_cards = list(played)
    g.winner = None
    g.scores = None
    return g


class TestBossDetection(unittest.TestCase):
    def test_two_of_spades_is_always_boss(self):
        g = make_state([[51, 4], [5], [8], [12]])
        self.assertTrue(boss_move(g, 0, classify([51], DEFAULT_RULES)))
        self.assertFalse(boss_move(g, 0, classify([4], DEFAULT_RULES)))

    def test_boss_when_every_higher_card_is_played(self):
        # all cards above 4 (index 4..51) already played: our 4 is boss
        g = make_state([[4], [0], [1], [2]], played=list(range(5, 52)))
        self.assertTrue(boss_move(g, 0, classify([4], DEFAULT_RULES)))

    def test_boss_chain_counts_unbeatable_units(self):
        g = make_state([[51, 4], [5], [8], [12]])
        boss, total = boss_chain(g, 0)
        self.assertEqual(total, 2)      # two loose singles
        self.assertEqual(boss, 1)       # only the 2 of spades is boss

    def test_unbeatable_probability_exact_for_boss(self):
        g = make_state([[51, 4], [5], [8], [12]])
        self.assertEqual(
            unbeatable_probability(g, 0, classify([51], DEFAULT_RULES)), 1.0
        )

    def test_unbeatable_probability_from_worlds(self):
        # opponent 1 holds the only answer; worlds split 50/50 on who has it
        # our 4D is card 4; only cards above it can answer (5..51)
        g = make_state([[4, 20], [5], [0], [1]])
        move = classify([4], DEFAULT_RULES)
        worlds = [
            ({1: [5], 2: [0], 3: [1]}, 1.0),   # 5 beats our 4: answerable
            ({1: [3], 2: [0], 3: [1]}, 1.0),   # all threes: nobody answers
        ]
        self.assertAlmostEqual(
            unbeatable_probability(g, 0, move, worlds), 0.5
        )


class TestExactSolver(unittest.TestCase):
    def test_solver_plays_the_boss_card_first(self):
        """The champion's endgame bug, as a solvable position.

        We hold 2S (unbeatable) and a low 4D; player 1 holds a 4C that
        beats the 4D and would go out on it.  Leading the 2S wins the
        whole game (+3); leading the 4D loses it (-1).
        """
        g = make_state([[51, 4], [5], [8], [12]])
        vals, exact = solve_move_values(g, 0)
        self.assertTrue(exact)
        self.assertEqual(vals[move_key(classify([51], DEFAULT_RULES))], 3.0)
        self.assertEqual(vals[move_key(classify([4], DEFAULT_RULES))], -1.0)
        best = max(vals, key=vals.get)
        self.assertEqual(best, (51,))

    def test_values_are_zero_sum(self):
        g = make_state([[51, 4], [5], [8], [12]])
        vec = solve(g)
        self.assertAlmostEqual(sum(vec), 0.0)

    def test_solver_matches_played_out_game(self):
        """Solved value is achievable: following the solver's own moves
        reproduces its predicted score."""
        g = make_state([[51, 4], [5], [8], [12]])
        predicted = solve(g)[0]
        while not g.game_over:
            p = g.turn
            vals, _ = solve_move_values(g, p)
            best = max(vals, key=vals.get)
            move = next(
                (m for m in g.legal_moves() if move_key(m) == best), None
            )
            g.step(move)
        self.assertEqual(float(g.scores[0]), predicted)


class TestPIMC(unittest.TestCase):
    def test_pimc_averages_exact_values_over_worlds(self):
        g = make_state([[51, 4], [5], [8], [12]])
        worlds = [({1: [5], 2: [8], 3: [12]}, 1.0)]
        vals = pimc_move_values(g, 0, worlds)
        # single world == perfect information: matches the exact solve
        self.assertEqual(vals[(51,)], 3.0)
        self.assertEqual(vals[(4,)], -1.0)

    def test_pimc_blends_two_worlds(self):
        g = make_state([[4, 20], [5], [8], [12]])
        worlds = [
            ({1: [5], 2: [8], 3: [12]}, 1.0),
            ({1: [1], 2: [8], 3: [12]}, 1.0),
        ]
        vals = pimc_move_values(g, 0, worlds)
        self.assertEqual(len(vals), 2)
        for v in vals.values():
            self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main()
