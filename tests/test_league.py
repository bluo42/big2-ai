import unittest

import numpy as np

from big2.dmc import DIM, DMCPolicy
from big2.league import League, run_league, train_dmc_vs_field
from big2.strategies import PlayLowest, SmartHeuristic


class TestLeague(unittest.TestCase):
    def test_matches_update_ratings(self):
        league = League(seed=0)
        for name in ("a", "b", "c", "d", "e"):
            league.add(name, PlayLowest(), "scripted")
        league.play_matches(10)
        played = sum(m.games for m in league.population)
        self.assertEqual(played, 40)  # 10 games x 4 seats
        total = sum(m.total for m in league.population)
        self.assertAlmostEqual(total, 0.0)  # zero-sum across the field

    def test_dmc_vs_field_updates_weights(self):
        league = League(seed=1)
        league.add("smart", SmartHeuristic(), "scripted")
        league.add("lowest", PlayLowest(), "scripted")
        league.add("lowest2", PlayLowest(), "scripted")
        member = league.add("dmc", DMCPolicy(), "dmc", trainable=True)
        w0 = np.zeros(DIM, dtype=np.float32)
        w1 = train_dmc_vs_field(w0, league, member, episodes=5)
        self.assertEqual(w1.shape, (DIM,))
        self.assertGreater(np.abs(w1).sum(), 0.0)
        self.assertEqual(np.abs(w0).sum(), 0.0)  # input untouched

    def test_generation_smoke(self):
        league = run_league(
            generations=1, games_per_gen=12, dmc_episodes=8,
            cem_iters=1, cem_pop=4, cem_games=4, seed=0,
            warm_start_cem=None, verbose=False,
        )
        names = {m.name for m in league.population}
        self.assertIn("dmc-g1", names)
        self.assertIn("cem-g1", names)
        # anchors survive pruning
        for anchor in ("smart", "dumper", "lowest", "decomp"):
            self.assertIn(anchor, names)


if __name__ == "__main__":
    unittest.main()
