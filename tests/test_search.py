import random
import unittest

import numpy as np

from big2.combos import classify
from big2.endgame import move_key
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES
from big2.search import (
    SEARCH_FULL,
    SEARCH_START,
    SearchAugmentedPolicy,
    search_distribution,
    search_weight,
)
from big2.strategies import PlayLowest, Strategy


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


class _WeakPrior(Strategy):
    """Stands in for the champion's mistake: prefers the cheapest card."""

    name = "weak"

    def option_scores(self, game, player):
        options = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        prior = []
        for m in options:
            prior.append(0.0 if m is None else -min(m.cards) / 10.0)
        return options, np.array(prior, dtype=np.float64)

    def select(self, game, player):
        options, prior = self.option_scores(game, player)
        return options[int(np.argmax(prior))]


class TestSearchWeight(unittest.TestCase):
    def test_ramp_endpoints_and_middle(self):
        big = make_state([list(range(13)), list(range(13, 26)),
                          list(range(26, 39)), list(range(39, 52))])
        self.assertEqual(search_weight(big), 0.0)
        tiny = make_state([[51, 4], [5], [8], [12]])
        self.assertEqual(search_weight(tiny), 1.0)
        mid_cards = (SEARCH_START + SEARCH_FULL) // 2
        per = mid_cards // 4 or 1
        mid = make_state([list(range(per)), list(range(13, 13 + per)),
                          list(range(26, 26 + per)),
                          list(range(39, 39 + per))])
        w = search_weight(mid)
        self.assertGreater(w, 0.0)
        self.assertLess(w, 1.0)


class TestSearchFixesEndgame(unittest.TestCase):
    def test_search_overrides_a_prior_that_wastes_the_boss_card(self):
        """The exact misplay: a cheap-first prior gives away the game;
        with the endgame tree the same agent leads the unbeatable 2S."""
        g = make_state([[51, 4], [5], [8], [12]])
        weak = _WeakPrior()
        self.assertEqual(weak.select(g, 0).cards, (4,))     # the mistake

        searcher = SearchAugmentedPolicy(weak, worlds=12, top_worlds=6)
        self.assertEqual(searcher.select(g, 0).cards, (51,))  # fixed

    def test_provable_win_short_circuits(self):
        g = make_state([[51], [5], [8], [12]])
        searcher = SearchAugmentedPolicy(_WeakPrior())
        self.assertEqual(searcher.select(g, 0).cards, (51,))

    def test_distribution_prefers_the_higher_ev_move(self):
        g = make_state([[51, 4], [5], [8], [12]])
        options = list(g.legal_moves(0))
        prior = np.zeros(len(options))
        probs, evs = search_distribution(
            g, 0, options, prior, worlds=12, top_worlds=6,
            rng=random.Random(0),
        )
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        best = options[int(np.argmax(probs))]
        self.assertEqual(best.cards, (51,))
        self.assertGreater(evs[move_key(classify([51], DEFAULT_RULES))],
                           evs[move_key(classify([4], DEFAULT_RULES))])

    def test_early_game_defers_entirely_to_the_prior(self):
        g = Big2Game(rng=random.Random(4))
        weak = _WeakPrior()
        searcher = SearchAugmentedPolicy(weak)
        self.assertEqual(
            move_key(searcher.select(g, g.turn)),
            move_key(weak.select(g, g.turn)),
        )

    def test_plays_full_games_without_error(self):
        searcher = SearchAugmentedPolicy(_WeakPrior(), worlds=8, top_worlds=4)
        g = Big2Game(rng=random.Random(11))
        scores = g.play_out([searcher, PlayLowest(), PlayLowest(), PlayLowest()])
        self.assertEqual(sum(scores.values()), 0)

    def test_search_finds_the_optimal_move_more_often_than_its_prior(self):
        """Benchmark on solved positions: deal random small endgames,
        compute the true best move by exact perfect-information solve,
        and count agreement.  The prior is deliberately weak; search
        should recover the optimum far more often."""
        from big2.endgame import solve_move_values

        base = _WeakPrior()
        searcher = SearchAugmentedPolicy(base, worlds=12, top_worlds=6)
        rng = random.Random(17)
        hits_search = hits_prior = graded = 0
        for _ in range(40):
            deck = list(range(52))
            rng.shuffle(deck)
            hands = [sorted(deck[i * 3:(i + 1) * 3]) for i in range(4)]
            g = make_state(hands, played=sorted(deck[12:]))
            truth, exact = solve_move_values(g, 0)
            if not exact or len(truth) < 2:
                continue
            best = max(truth.values())
            # only grade positions where the choice actually matters
            if best == min(truth.values()):
                continue
            graded += 1
            optimal = {k for k, v in truth.items() if v == best}
            hits_search += move_key(searcher.select(g, 0)) in optimal
            hits_prior += move_key(base.select(g, 0)) in optimal
        self.assertGreater(graded, 10)
        # measured on 69 solved spots: prior 35%, smart 41%, search 78%
        self.assertGreater(hits_search, hits_prior)
        self.assertGreater(hits_search / graded, 0.6)


if __name__ == "__main__":
    unittest.main()
