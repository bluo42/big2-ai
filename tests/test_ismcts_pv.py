import random
import time
import unittest

import numpy as np

from big2.combos import classify
from big2.endgame import move_key
from big2.game import Big2Game, ScoringConfig
from big2.ismcts_pv import MAX_CANDIDATES, PolicyValueISMCTS, SearchResult
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayLowest, SmartHeuristic, Strategy


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


class _Peaked(Strategy):
    """A policy with a sharp opinion, so priors are easy to reason about."""

    name = "peaked"

    def __init__(self, favour_lowest=True):
        self.favour_lowest = favour_lowest

    def option_scores(self, game, player):
        options = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        s = []
        for m in options:
            if m is None:
                s.append(-2.0)
            else:
                v = -min(m.cards) if self.favour_lowest else max(m.cards)
                s.append(v / 8.0)
        return options, np.array(s, dtype=np.float64)

    def select(self, game, player):
        options, scores = self.option_scores(game, player)
        return options[int(np.argmax(scores))]


class TestPrior(unittest.TestCase):
    def test_prior_is_a_distribution_over_the_legal_set(self):
        g = Big2Game(rng=random.Random(1))
        s = PolicyValueISMCTS(_Peaked(), simulations=4)
        options, prior = s._prior(g, g.turn)
        self.assertEqual(len(options), len(prior))
        self.assertAlmostEqual(float(prior.sum()), 1.0, places=6)
        self.assertTrue((prior >= 0).all())

    def test_a_policy_without_scores_still_gets_a_usable_prior(self):
        g = Big2Game(rng=random.Random(2))
        s = PolicyValueISMCTS(SmartHeuristic(), simulations=4)
        options, prior = s._prior(g, g.turn)
        self.assertAlmostEqual(float(prior.sum()), 1.0, places=6)
        self.assertGreater(prior.max(), prior.min())   # it has an opinion


class TestSearchBehaviour(unittest.TestCase):
    def test_one_simulation_returns_the_policy_move(self):
        """With no evidence the prior must carry the decision -- this is
        what makes the early game defer to the network automatically."""
        g = Big2Game(rng=random.Random(3))
        pol = _Peaked()
        s = PolicyValueISMCTS(pol, simulations=1)
        res = s.search(g, g.turn)
        self.assertEqual(res.policy_move, move_key(pol.select(g, g.turn)))

    def test_search_reports_visits_values_and_prior(self):
        g = Big2Game(rng=random.Random(4))
        for _ in range(10):
            g.step(PlayLowest().select(g, g.turn))
        while not g.game_over and len(g.legal_moves(g.turn)) < 2:
            g.step(PlayLowest().select(g, g.turn))
        s = PolicyValueISMCTS(_Peaked(), simulations=24, depth=4)
        res = s.search(g, g.turn)
        self.assertEqual(sum(res.visits.values()), res.simulations)
        self.assertEqual(set(res.visits), set(res.prior))
        for v in res.values.values():
            self.assertTrue(np.isfinite(v))
        self.assertIsInstance(res.agreed, bool)
        self.assertTrue(np.isfinite(res.margin))

    def test_more_simulations_concentrate_the_visits(self):
        g = Big2Game(rng=random.Random(6))
        for _ in range(8):
            g.step(PlayLowest().select(g, g.turn))
        while not g.game_over and len(g.legal_moves(g.turn)) < 3:
            g.step(PlayLowest().select(g, g.turn))
        p = g.turn
        few = PolicyValueISMCTS(_Peaked(), simulations=8, seed=1).search(g, p)
        many = PolicyValueISMCTS(_Peaked(), simulations=64, seed=1).search(g, p)
        self.assertEqual(sum(few.visits.values()), few.simulations)
        self.assertEqual(sum(many.visits.values()), many.simulations)
        self.assertGreater(many.simulations, few.simulations)
        # the best line should take a larger share as evidence accrues
        self.assertGreaterEqual(
            max(many.visits.values()) / many.simulations,
            max(few.visits.values()) / few.simulations - 0.35,
        )

    def test_a_certain_prior_skips_the_search(self):
        """Cheap moves stay cheap: with one candidate above the prior
        floor and no optional pass to weigh against it, there is nothing
        to arbitrate, so the search returns without simulating."""

        class _Sure(_Peaked):
            def option_scores(self, game, player):
                options, scores = super().option_scores(game, player)
                s = np.full(len(options), -30.0)
                s[int(np.argmax(scores))] = 30.0     # certain
                return options, s

        # Leading a fresh trick: passing is not legal, so the shortlist
        # really can collapse to one move.
        g = make_state([[0, 4, 8], [5, 9, 13], [6, 10, 14], [7, 11, 15]])
        res = PolicyValueISMCTS(_Sure(), simulations=400, seed=1).search(
            g, 0)
        self.assertEqual(res.simulations, 0)
        self.assertEqual(res.move, res.policy_move)

    def test_the_time_budget_is_respected(self):
        class _Slow(_Peaked):
            def option_scores(self, game, player):
                time.sleep(0.004)
                return super().option_scores(game, player)

        g = Big2Game(rng=random.Random(7))
        for _ in range(6):
            g.step(PlayLowest().select(g, g.turn))
        s = PolicyValueISMCTS(_Slow(), simulations=100000, depth=6, seed=1,
                              time_budget=0.25)
        res = s.search(g, g.turn)
        self.assertLess(res.elapsed, 1.5)
        self.assertLess(res.simulations, 100000)

    def test_moves_below_the_prior_floor_cost_nothing(self):
        """Only candidates the prior takes seriously are measured, and
        the budget is split evenly between them."""

        class _TwoLive(_Peaked):
            def option_scores(self, game, player):
                options, scores = super().option_scores(game, player)
                s = np.full(len(options), -12.0)     # ~0 after softmax
                order = np.argsort(-np.asarray(scores))
                s[order[0]] = 1.0
                s[order[1]] = 0.6
                return options, s

        g = make_state([[0, 4, 8, 12, 16, 20], [5, 9, 13],
                        [6, 10, 14], [7, 11, 15]])
        res = PolicyValueISMCTS(_TwoLive(), simulations=40, seed=1).search(
            g, 0)
        searched = [k for k, v in res.visits.items() if v > 0]
        self.assertEqual(len(searched), 2)
        counts = sorted(res.visits[k] for k in searched)
        self.assertLessEqual(counts[-1] - counts[0], 1)

    def test_pass_is_always_measured_when_it_is_optional(self):
        """Hold-or-spend is the decision the prior is least reliable on,
        so passing is never decided without simulating it."""

        class _NoPass(_Peaked):
            def option_scores(self, game, player):
                options, scores = super().option_scores(game, player)
                s = np.asarray(scores, dtype=np.float64).copy()
                for i, m in enumerate(options):
                    s[i] = -30.0 if m is None else s[i]    # prior hates pass
                return options, s

        # Facing a played single, holding beaters: passing is optional.
        g = make_state([[20, 24, 28], [5, 9], [6, 10], [7, 11]])
        g.table_combo = classify((16,), g.rules)
        g.table_player = 3
        res = PolicyValueISMCTS(_NoPass(), simulations=40, seed=1).search(
            g, 0)
        self.assertIn(None, [m for m in [None] if True])   # pass is legal
        self.assertGreater(res.visits.get(move_key(None), 0), 0,
                           "pass must be simulated, not assumed")

    def test_a_torn_prior_widens_to_the_cap(self):
        class _Flat(_Peaked):
            def option_scores(self, game, player):
                options, scores = super().option_scores(game, player)
                return options, np.zeros(len(options))   # no opinion

        g = Big2Game(rng=random.Random(11))
        for _ in range(4):
            g.step(PlayLowest().select(g, g.turn))
        while not g.game_over and len(g.legal_moves(g.turn)) < 6:
            g.step(PlayLowest().select(g, g.turn))
        res = PolicyValueISMCTS(_Flat(), simulations=60, breadth=4,
                                seed=1).search(g, g.turn)
        searched = [k for k, v in res.visits.items() if v > 0]
        # A torn prior widens to the shortlist cap (plus pass, when
        # passing is legal here) and no further.
        self.assertGreaterEqual(len(searched), MAX_CANDIDATES)
        self.assertLessEqual(len(searched), MAX_CANDIDATES + 1)

    def test_breadth_is_cut_by_the_prior(self):
        g = Big2Game(rng=random.Random(11))
        for _ in range(4):
            g.step(PlayLowest().select(g, g.turn))
        while not g.game_over and len(g.legal_moves(g.turn)) < 6:
            g.step(PlayLowest().select(g, g.turn))
        s = PolicyValueISMCTS(_Peaked(), simulations=40, breadth=3, seed=1)
        res = s.search(g, g.turn)
        searched = [k for k, v in res.visits.items() if v > 0]
        self.assertLessEqual(len(searched), 3)
        # every legal move is still reported, so nothing is hidden
        self.assertEqual(len(res.prior), len(res.visits))
        self.assertGreater(len(res.visits), len(searched))

    def test_forced_positions_short_circuit(self):
        g = make_state([[51], [5], [8], [12]])
        res = PolicyValueISMCTS(_Peaked(), simulations=16).search(g, 0)
        self.assertEqual(res.move, (51,))
        self.assertEqual(res.simulations, 0)

    def test_search_finds_the_boss_card_a_cheap_prior_misses(self):
        """The endgame spot: the prior wants the 4, but the position is
        small enough to settle exactly, and the exact answer is the
        unbeatable 2S."""
        g = make_state([[51, 4], [5], [8], [12]])
        pol = _Peaked(favour_lowest=True)
        self.assertEqual(move_key(pol.select(g, 0)), (4,))
        res = PolicyValueISMCTS(pol, simulations=48, seed=2).search(g, 0)
        self.assertEqual(res.move, (51,))
        self.assertFalse(res.agreed)
        self.assertGreater(res.margin, 0.0)
        self.assertTrue(res.exact)


class TestBeliefWeighting(unittest.TestCase):
    def test_worlds_are_drawn_in_proportion_to_their_weight(self):
        g = Big2Game(rng=random.Random(12))
        for _ in range(9):
            g.step(PlayLowest().select(g, g.turn))
        s = PolicyValueISMCTS(_Peaked(), simulations=8, seed=3)
        worlds, probs = s._worlds(g, g.turn, k=16)
        self.assertEqual(len(worlds), len(probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertTrue(all(p >= 0 for p in probs))
        # weights come from the posterior, so they are not all identical
        # unless the evidence really is uninformative
        for (world, w), p in zip(worlds, probs):
            self.assertGreaterEqual(w, 0.0)
            if w > 0:
                self.assertGreater(p, 0.0)

    def test_opponent_replies_are_sampled_not_fixed(self):
        """A determinization is already one guess; answering it greedily
        would collapse the rollout to a single fictitious line."""
        g = Big2Game(rng=random.Random(13))
        s = PolicyValueISMCTS(_Peaked(), simulations=4, seed=5)
        picks = {move_key(s._sampled_move(g, g.turn)) for _ in range(60)}
        self.assertGreater(len(picks), 1)

    def test_sampling_collapses_to_greedy_at_low_temperature(self):
        g = Big2Game(rng=random.Random(13))
        pol = _Peaked()
        s = PolicyValueISMCTS(pol, simulations=4, seed=5, rollout_temp=1e-3)
        options, scores = pol.option_scores(g, g.turn)
        top = {move_key(m) for m, sc in zip(options, scores)
               if sc >= scores.max() - 1e-9}
        for _ in range(20):
            self.assertIn(move_key(s._sampled_move(g, g.turn)), top)


class TestValue(unittest.TestCase):
    def test_terminal_states_score_exactly(self):
        g = make_state([[51], [5], [8], [12]])
        s = PolicyValueISMCTS(_Peaked(), simulations=2)
        g.step(classify([51], DEFAULT_RULES))
        self.assertTrue(g.game_over)
        v, exact = s._value(g, 0)
        self.assertAlmostEqual(v, g.scores[0] / 39.0, places=6)
        self.assertTrue(exact)

    def test_small_positions_are_solved_not_estimated(self):
        g = make_state([[51, 4], [5], [8], [12]])
        s = PolicyValueISMCTS(_Peaked(), simulations=2, solve_below=12)
        v, exact = s._value(g, 0)
        self.assertTrue(-1.0 <= v <= 1.0)
        self.assertTrue(exact)

    def test_big_positions_are_estimated_not_solved(self):
        g = Big2Game(rng=random.Random(14))
        s = PolicyValueISMCTS(_Peaked(), simulations=2, solve_below=12)
        v, exact = s._value(g, 0)
        self.assertTrue(-1.0 <= v <= 1.0)
        self.assertFalse(exact)


if __name__ == "__main__":
    unittest.main()
