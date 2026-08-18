import random
import unittest

from big2.endgame import move_key
from big2.faceup import (
    InfoPoint,
    best_under_information,
    information_point,
    reveal_worlds,
    summarize,
)
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayLowest, SmartHeuristic


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


class TestRevealWorlds(unittest.TestCase):
    def test_full_reveal_is_the_true_deal(self):
        g = Big2Game(rng=random.Random(1))
        worlds = reveal_worlds(g, 0, 1.0)
        self.assertEqual(len(worlds), 1)
        world, w = worlds[0]
        self.assertEqual(w, 1.0)
        for p in (1, 2, 3):
            self.assertEqual(sorted(world[p]), sorted(g.hands[p]))

    def test_partial_reveal_keeps_hand_sizes_and_revealed_cards(self):
        g = Big2Game(rng=random.Random(2))
        for _ in range(6):
            g.step(PlayLowest().select(g, g.turn))
        worlds = reveal_worlds(g, 0, 0.5, k=6, rng=random.Random(3))
        self.assertTrue(worlds)
        for world, _w in worlds:
            for p in (1, 2, 3):
                self.assertEqual(len(world[p]), len(g.hands[p]))
            # nobody is dealt a card we hold or that has been played
            seen = set(g.played_cards) | set(g.hands[0])
            for cards in world.values():
                self.assertFalse(seen & set(cards))

    def test_zero_reveal_still_produces_plausible_deals(self):
        g = Big2Game(rng=random.Random(4))
        worlds = reveal_worlds(g, 0, 0.0, k=5, rng=random.Random(5))
        self.assertTrue(worlds)
        for world, _w in worlds:
            for p in (1, 2, 3):
                self.assertEqual(len(world[p]), len(g.hands[p]))


class TestBestUnderInformation(unittest.TestCase):
    def test_face_up_finds_the_winning_move(self):
        """With the deal exposed, the boss card is provably right."""
        g = make_state([[51, 4], [5], [8], [12]])
        worlds = reveal_worlds(g, 0, 1.0)
        best, evs = best_under_information(g, 0, worlds)
        self.assertEqual(best, (51,))
        self.assertGreater(evs[(51,)], evs[(4,)])

    def test_no_worlds_gives_no_answer(self):
        g = make_state([[51, 4], [5], [8], [12]])
        best, evs = best_under_information(g, 0, [])
        self.assertIsNone(best)
        self.assertEqual(evs, {})


class TestInformationPoint(unittest.TestCase):
    def test_flags_a_position_where_sight_changes_the_move(self):
        """A cheap-first policy misplays the boss spot; seeing the deal
        picks the 2S, so this counts as information worth having."""
        g = make_state([[51, 4], [5], [8], [12]])
        pt = information_point(g, 0, PlayLowest(), [SmartHeuristic()],
                               rollouts=4, rng=random.Random(0))
        self.assertIsNotNone(pt)
        self.assertEqual(pt.faceup_move, (51,))
        self.assertEqual(pt.model_move, (4,))
        self.assertTrue(pt.changed)
        self.assertGreater(pt.value_of_information, 0.0)

    def test_agreement_means_no_information_value(self):
        g = make_state([[51, 4], [5], [8], [12]])
        pt = information_point(g, 0, _AlwaysBoss(), [SmartHeuristic()],
                               rollouts=4, rng=random.Random(0))
        self.assertFalse(pt.changed)
        self.assertEqual(pt.value_of_information, 0.0)

    def test_single_option_positions_are_skipped(self):
        g = make_state([[51], [5], [8], [12]])
        self.assertIsNone(
            information_point(g, 0, PlayLowest(), [SmartHeuristic()])
        )


class _AlwaysBoss(SmartHeuristic):
    name = "boss"

    def select(self, game, player):
        moves = game.legal_moves(player)
        return max(moves, key=lambda m: max(m.cards)) if moves else None


class TestSummary(unittest.TestCase):
    def test_aggregates_the_two_headline_numbers(self):
        pts = [
            InfoPoint(0, 0, (1, 1, 1, 1), (1,), (2,), 0.0, 3.0, (2,), None),
            InfoPoint(1, 0, (1, 1, 1, 1), (5,), (5,), 1.0, 1.0, (5,), None),
        ]
        s = summarize(pts)
        self.assertEqual(s["points"], 2)
        self.assertAlmostEqual(s["faceup_changes_move"], 0.5)
        self.assertAlmostEqual(s["partial_changes_move"], 0.5)
        self.assertAlmostEqual(s["mean_value_of_information"], 1.5)
        self.assertAlmostEqual(s["max_value_of_information"], 3.0)
        self.assertEqual(summarize([])["points"], 0)


if __name__ == "__main__":
    unittest.main()
