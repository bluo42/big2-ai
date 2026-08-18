import random
import unittest

from big2.combos import classify
from big2.critique import (
    Node,
    critical_nodes,
    losing_games,
    move_ev,
    rollout_ev,
    summarize,
)
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayHighest, PlayLowest, SmartHeuristic, Strategy


def synth_replay(seed):
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=4, rng=random.Random(seed))
    initial = [list(h) for h in g.hands]
    scores = g.play_out([SmartHeuristic(), PlayLowest(), PlayLowest(),
                         PlayHighest()])
    return {
        "user_seat": 0,
        "replay": {
            "num_players": 4,
            "rules": {"allow_triples": False, "pass_locks": False,
                      "flush_rank_first": False},
            "scoring": "tiered",
            "start_card": g.start_card,
            "initial_hands": initial,
            "actions": [
                {"p": r.player,
                 "cards": list(r.combo.cards) if r.combo else None,
                 "te": r.trick_end}
                for r in g.history
            ],
            "scores": {str(p): s for p, s in scores.items()},
            "winner": g.winner,
        },
    }


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


class _Fixed(Strategy):
    """Plays a named card if it can, else the lowest."""

    def __init__(self, card, name="fixed"):
        self.card = card
        self.name = name

    def select(self, game, player):
        for m in game.legal_moves(player):
            if tuple(m.cards) == (self.card,):
                return m
        return PlayLowest().select(game, player)


class TestGameSelection(unittest.TestCase):
    def test_picks_out_the_badly_lost_games(self):
        rows = [synth_replay(s) for s in range(12)]
        heavy = losing_games(rows, margin=8.0)
        light = losing_games(rows, margin=0.5)
        self.assertLessEqual(len(heavy), len(light))
        from big2.offline import replay_outcomes

        for _i, body, seat in heavy:
            self.assertLessEqual(replay_outcomes(body)[seat], -8.0)

    def test_seat_filter_examines_that_seat_only(self):
        rows = [synth_replay(s) for s in range(12)]
        for _i, _body, seat in losing_games(rows, seat=2, margin=0.0):
            self.assertEqual(seat, 2)


class TestNodeEV(unittest.TestCase):
    def test_small_positions_are_settled_exactly(self):
        """The boss-card spot: exact solving must call it, not sampling."""
        g = make_state([[51, 4], [5], [8], [12]])
        boss, e1 = move_ev(g, 0, classify([51], DEFAULT_RULES),
                           [SmartHeuristic()])
        weak, e2 = move_ev(g, 0, classify([4], DEFAULT_RULES),
                           [SmartHeuristic()])
        self.assertTrue(e1 and e2)          # exact, not rollout
        self.assertEqual(boss, 3.0)
        self.assertEqual(weak, -1.0)

    def test_rollouts_used_when_the_position_is_large(self):
        g = Big2Game(rng=random.Random(4))
        move = g.legal_moves()[0]
        ev, exact = move_ev(g, g.turn, move, [SmartHeuristic()], rollouts=4)
        self.assertFalse(exact)
        self.assertIsInstance(ev, float)

    def test_rollout_ev_is_deterministic_for_a_seed(self):
        g = Big2Game(rng=random.Random(6))
        move = g.legal_moves()[0]
        a = rollout_ev(g, g.turn, move, [PlayLowest()], n=5, seed=3)
        b = rollout_ev(g, g.turn, move, [PlayLowest()], n=5, seed=3)
        self.assertEqual(a, b)


class TestCriticalNodes(unittest.TestCase):
    def test_only_divergent_decisions_are_reported(self):
        rows = [synth_replay(s) for s in range(6)]
        same = PlayLowest()
        nodes = critical_nodes(rows, same, same, margin=0.0, rollouts=3)
        self.assertEqual(nodes, [])  # a model never diverges from itself

    def test_finds_and_ranks_disagreements(self):
        rows = [synth_replay(s) for s in range(8)]
        nodes = critical_nodes(
            rows, PlayLowest(), PlayHighest(), margin=0.0, rollouts=4,
            max_games=3,
        )
        self.assertTrue(nodes)
        crits = [n.criticality for n in nodes]
        self.assertEqual(crits, sorted(crits, reverse=True))
        for n in nodes:
            self.assertNotEqual(n.old_move, n.new_move)
            self.assertIn(n.verdict, ("new better", "old better", "tie"))
            self.assertIn("EV", n.describe())

    def test_min_criticality_filters_noise(self):
        rows = [synth_replay(s) for s in range(8)]
        loose = critical_nodes(rows, PlayLowest(), PlayHighest(),
                               margin=0.0, rollouts=4, max_games=3)
        tight = critical_nodes(rows, PlayLowest(), PlayHighest(),
                               margin=0.0, rollouts=4, max_games=3,
                               min_criticality=2.0)
        self.assertLessEqual(len(tight), len(loose))
        for n in tight:
            self.assertGreaterEqual(n.criticality, 2.0)

    def test_summary_counts_who_was_right(self):
        nodes = [
            Node(0, 1, 0, (1,), (1, 1, 1, 1), (1,), (2,), 1.0, 3.0, True),
            Node(0, 2, 0, (1,), (1, 1, 1, 1), (1,), (2,), 2.0, 1.0, False),
        ]
        s = summarize(nodes)
        self.assertEqual(s["nodes"], 2)
        self.assertEqual(s["new_better"], 1)
        self.assertEqual(s["old_better"], 1)
        self.assertEqual(s["exactly_solved"], 1)
        self.assertAlmostEqual(s["mean_ev_gain"], 0.5)
        self.assertAlmostEqual(s["max_criticality"], 2.0)
        self.assertEqual(summarize([])["nodes"], 0)


if __name__ == "__main__":
    unittest.main()
