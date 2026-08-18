import random
import time
import unittest

import numpy as np

from big2.agent import OVERRIDE_MARGIN, SOLVE_CARDS, Decision, IntegratedAgent
from big2.combos import classify
from big2.endgame import move_key, remaining_cards
from big2.game import Big2Game, ScoringConfig
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


class _CheapFirst(Strategy):
    """A prior that always dumps its lowest card -- the mistake."""

    name = "cheap"

    def option_scores(self, game, player):
        options = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        prior = [0.0 if m is None else -min(m.cards) / 10.0 for m in options]
        return options, np.array(prior, dtype=np.float64)

    def select(self, game, player):
        options, prior = self.option_scores(game, player)
        return options[int(np.argmax(prior))]


class TestSolverOverride(unittest.TestCase):
    """The endgame is the one place a line can be *verified*, and that is
    the only place the agent is allowed to stop deferring to evidence."""

    def test_endgame_uses_the_exact_solver(self):
        g = make_state([[51, 4], [5], [8], [12]])
        agent = IntegratedAgent(_CheapFirst())
        dec = agent.explain(g, 0)
        self.assertLessEqual(dec.cards_left, SOLVE_CARDS)
        self.assertEqual(dec.source, "solver")
        self.assertEqual(dec.move, (51,))          # fixes the cheap prior
        self.assertTrue(dec.exact)

    def test_provable_win_short_circuits_to_the_solver(self):
        """Holding 2H 2S: the pair empties the hand and nothing beats a
        pair of twos, so it wins on the spot -- even though playing
        either card singly is also legal."""
        g = make_state([[50, 51], [5], [8], [12]])
        agent = IntegratedAgent(_CheapFirst())
        dec = agent.explain(g, 0)
        self.assertEqual(dec.source, "solver")
        self.assertEqual(dec.move, (50, 51))

    def test_forced_moves_are_labelled(self):
        g = make_state([[51], [5], [8], [12]])
        agent = IntegratedAgent(_CheapFirst(), use_solver=False,
                                use_search=False)
        self.assertEqual(agent.explain(g, 0).source, "forced")


class TestNoStageSwitch(unittest.TestCase):
    """There is no threshold at which a different decision-maker takes
    over.  Search runs everywhere; what changes with the hand is how much
    the evidence it gathers is worth against the prior."""

    def test_the_search_runs_in_the_opening(self):
        g = Big2Game(rng=random.Random(3))
        g.step(PlayLowest().select(g, g.turn))
        while not g.game_over and len(g.legal_moves(g.turn)) < 2:
            g.step(PlayLowest().select(g, g.turn))
        dec = IntegratedAgent(_CheapFirst(), simulations=24).explain(g, g.turn)
        self.assertGreater(dec.cards_left, 40)
        self.assertGreater(dec.simulations, 0)     # it searched
        self.assertTrue(dec.prior)                 # ...guided by the net
        self.assertGreater(dec.worlds, 0)          # ...over sampled hands

    def test_the_search_runs_in_the_midgame(self):
        g = Big2Game(rng=random.Random(4))
        while not g.game_over and remaining_cards(g) > 34:
            g.step(PlayLowest().select(g, g.turn))
        dec = IntegratedAgent(_CheapFirst(), simulations=24).explain(g, g.turn)
        self.assertIn(dec.source, ("policy", "search", "forced"))
        if dec.source != "forced":
            self.assertGreater(dec.simulations, 0)

    def test_a_thin_margin_leaves_the_move_with_the_policy(self):
        """A search that cannot show its pick is better does not get it."""
        g = Big2Game(rng=random.Random(4))
        while not g.game_over and remaining_cards(g) > 34:
            g.step(PlayLowest().select(g, g.turn))
        p = g.turn
        strict = IntegratedAgent(_CheapFirst(), simulations=24,
                                 override_margin=1e6).explain(g, p)
        self.assertEqual(strict.source, "policy")
        self.assertEqual(strict.move, strict.policy_move)

    def test_a_wide_margin_hands_the_move_to_the_search(self):
        g = Big2Game(rng=random.Random(4))
        while not g.game_over and remaining_cards(g) > 34:
            g.step(PlayLowest().select(g, g.turn))
        p = g.turn
        loose = IntegratedAgent(_CheapFirst(), simulations=48,
                                override_margin=-1e6).explain(g, p)
        if not loose.agreed_with_policy:
            self.assertEqual(loose.source, "search")


class TestBudget(unittest.TestCase):
    def test_a_decision_respects_its_time_budget(self):
        g = Big2Game(rng=random.Random(8))
        for _ in range(6):
            g.step(PlayLowest().select(g, g.turn))
        agent = IntegratedAgent(SmartHeuristic(), simulations=100000,
                                time_budget=0.3)
        t0 = time.monotonic()
        agent.explain(g, g.turn)
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_a_whole_game_stays_within_the_budget_per_move(self):
        agent = IntegratedAgent(SmartHeuristic(), simulations=32,
                                time_budget=1.0)
        g = Big2Game(rng=random.Random(11))
        t0 = time.monotonic()
        moves = 0
        while not g.game_over:
            p = g.turn
            g.step(agent.select(g, p) if p == 0
                   else PlayLowest().select(g, p))
            moves += 1
        self.assertGreater(moves, 4)
        self.assertLess(time.monotonic() - t0, 1.0 * moves)


class TestExplanation(unittest.TestCase):
    def test_decision_serialises_for_the_ui(self):
        g = Big2Game(rng=random.Random(5))
        g.step(PlayLowest().select(g, g.turn))
        while not g.game_over and len(g.legal_moves(g.turn)) < 2:
            g.step(PlayLowest().select(g, g.turn))
        d = IntegratedAgent(_CheapFirst(), simulations=16).explain(
            g, g.turn).as_dict()
        for field in ("move", "policy_move", "source", "margin", "cards_left",
                      "simulations", "worlds", "elapsed_ms", "exact",
                      "candidates"):
            self.assertIn(field, d)
        self.assertTrue(d["candidates"])
        for c in d["candidates"]:
            self.assertIn("prior", c)
            self.assertIn("visits", c)
            self.assertIn("value", c)

    def test_priors_are_a_distribution(self):
        g = Big2Game(rng=random.Random(5))
        g.step(PlayLowest().select(g, g.turn))
        while not g.game_over and len(g.legal_moves(g.turn)) < 2:
            g.step(PlayLowest().select(g, g.turn))
        dec = IntegratedAgent(_CheapFirst(), simulations=16).explain(g, g.turn)
        self.assertAlmostEqual(sum(dec.prior.values()), 1.0, places=5)


class TestPlay(unittest.TestCase):
    def test_plays_complete_games(self):
        agent = IntegratedAgent(SmartHeuristic(), simulations=8, depth=2)
        g = Big2Game(rng=random.Random(11))
        scores = g.play_out([agent, PlayLowest(), PlayLowest(), PlayLowest()])
        self.assertEqual(sum(scores.values()), 0)

    def test_select_matches_explain(self):
        g = make_state([[51, 4], [5], [8], [12]])
        agent = IntegratedAgent(_CheapFirst())
        self.assertEqual(move_key(agent.select(g, 0)), agent.explain(g, 0).move)

    def test_disabling_search_falls_back_to_the_policy(self):
        g = make_state([[51, 4], [5], [8], [12]])
        agent = IntegratedAgent(_CheapFirst(), use_solver=False,
                                use_search=False)
        self.assertEqual(move_key(agent.select(g, 0)),
                         move_key(_CheapFirst().select(g, 0)))


if __name__ == "__main__":
    unittest.main()
