import os
import tempfile
import unittest

from big2.evolve import Genome, playoff, run_island, sample_genome
import random


class TestEvolve(unittest.TestCase):
    CFG = {
        "seed": 0,
        "islands": 2,
        "pop_size": 3,
        "games_per_island": 120,
        "phase_2p_games": 80,
        "games_per_round": 60,
        "min_window_games": 10,
        "evolve_margin": 0.1,
        "max_checkpoints": 2,
    }

    def test_genome_mutation_bounds(self):
        rng = random.Random(0)
        g = sample_genome(rng)
        for _ in range(50):
            g = g.mutate(rng)
            self.assertGreaterEqual(g.lr, 1e-4)
            self.assertLessEqual(g.lr, 5e-2)
            self.assertGreaterEqual(g.eps, 0.02)
            self.assertLessEqual(g.eps, 0.30)
            self.assertIn(len(g.hidden), (1, 2, 3, 4))

    def test_islands_and_playoff(self):
        with tempfile.TemporaryDirectory() as d:
            # run two tiny islands sequentially in-process
            run_island(0, self.CFG, d)
            run_island(1, self.CFG, d)
            self.assertTrue(os.path.exists(os.path.join(d, "island0_final.npz")))
            self.assertTrue(os.path.exists(os.path.join(d, "island1_final.npz")))
            winner_path, meta = playoff(d, n_games=24, seed=0)
            self.assertTrue(winner_path.endswith("_final.npz"))
            self.assertIn("genome", meta)


if __name__ == "__main__":
    unittest.main()
