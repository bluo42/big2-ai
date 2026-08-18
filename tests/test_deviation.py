import random
import unittest

from big2.deviation import move_class, phase_of, winning_deviations
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES
from big2.strategies import PlayHighest, PlayLowest, SmartHeuristic


def synth_replay(seed, policies=None):
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=4, rng=random.Random(seed))
    initial = [list(h) for h in g.hands]
    scores = g.play_out(policies or [PlayHighest()] + [PlayLowest()] * 3)
    return {"username": "tester", "user_seat": 0, "replay": {
        "num_players": 4,
        "rules": {"allow_triples": False, "pass_locks": False,
                  "flush_rank_first": False},
        "scoring": "tiered", "start_card": g.start_card,
        "initial_hands": initial,
        "actions": [{"p": r.player,
                     "cards": list(r.combo.cards) if r.combo else None,
                     "te": r.trick_end} for r in g.history],
        "scores": {str(p): s for p, s in scores.items()},
        "winner": g.winner}}


class TestHelpers(unittest.TestCase):
    def test_move_class_names(self):
        from big2.combos import classify

        self.assertEqual(move_class(None), "pass")
        self.assertEqual(move_class(classify([0], DEFAULT_RULES)), "single")
        self.assertEqual(move_class(classify([0, 1], DEFAULT_RULES)), "pair")

    def test_phase_tracks_cards_remaining(self):
        g = Big2Game(rng=random.Random(1))
        self.assertEqual(phase_of(g), "early (>36 left)")
        g.hands = [h[:5] for h in g.hands]        # 20 cards left
        self.assertEqual(phase_of(g), "late (<=20)")
        g.hands = [h[:7] for h in Big2Game(rng=random.Random(1)).hands]
        self.assertEqual(phase_of(g), "mid (20-36)")


class TestWinningDeviations(unittest.TestCase):
    def test_no_deviations_when_the_model_is_the_recorded_player(self):
        """A model compared against its own recorded play must find
        nothing to disagree with."""
        rows = [synth_replay(s, [PlayLowest()] * 4) for s in range(3)]
        out = winning_deviations(rows, PlayLowest(), rollouts=2)
        self.assertEqual(out["diverged"], 0)
        self.assertEqual(out["better"], 0)

    def test_finds_and_describes_disagreements(self):
        rows = [synth_replay(s) for s in range(4)]   # seat 0 plays highest
        out = winning_deviations(rows, PlayLowest(), rollouts=3,
                                 opponents=[SmartHeuristic()], max_games=3)
        self.assertGreater(out["diverged"], 0)
        self.assertIn("swaps", out)
        self.assertIn("phases", out)
        for c in out["cases"]:
            self.assertGreaterEqual(c["gain"], 0.25)
            self.assertIn("hand", c)
        gains = [c["gain"] for c in out["cases"]]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_min_gain_filters_marginal_cases(self):
        rows = [synth_replay(s) for s in range(4)]
        loose = winning_deviations(rows, PlayLowest(), rollouts=3,
                                   min_gain=0.0, max_games=3)
        tight = winning_deviations(rows, PlayLowest(), rollouts=3,
                                   min_gain=5.0, max_games=3)
        self.assertLessEqual(tight["better"], loose["better"])


if __name__ == "__main__":
    unittest.main()
