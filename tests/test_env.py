import unittest

import numpy as np

from big2.env import MAX_ACTIONS, OBS_SIZE, Big2Env
from big2.strategies import PlayLowest, RandomPolicy, SmartHeuristic


class TestEnv(unittest.TestCase):
    def make_env(self, seed=0):
        return Big2Env(
            opponents=[PlayLowest(), SmartHeuristic(), RandomPolicy(seed)],
            seed=seed,
        )

    def test_reset_shapes(self):
        env = self.make_env()
        obs, info = env.reset(seed=42)
        self.assertEqual(obs["obs"].shape, (OBS_SIZE,))
        self.assertEqual(obs["action_mask"].shape, (MAX_ACTIONS,))
        self.assertGreater(obs["action_mask"].sum(), 0)
        self.assertEqual(len(info["legal_moves"]), obs["action_mask"][1:].sum())

    def test_random_valid_episodes(self):
        env = self.make_env()
        rng = np.random.default_rng(0)
        for ep in range(5):
            obs, info = env.reset(seed=100 + ep)
            terminated = False
            steps = 0
            final_reward = None
            while not terminated:
                mask = obs["action_mask"]
                action = int(rng.choice(np.flatnonzero(mask)))
                obs, reward, terminated, truncated, info = env.step(action)
                final_reward = reward
                steps += 1
                self.assertLess(steps, 500)
            self.assertIn("scores", info)
            self.assertEqual(final_reward, info["scores"][info["agent_seat"]])
            self.assertEqual(sum(info["scores"].values()), 0)

    def test_invalid_action_raises(self):
        env = self.make_env()
        obs, _ = env.reset(seed=7)
        bad = int(np.flatnonzero(obs["action_mask"] == 0)[-1])
        with self.assertRaises(ValueError):
            env.step(bad)

    def test_mask_pass_only_when_following(self):
        env = self.make_env()
        for ep in range(3):
            obs, info = env.reset(seed=200 + ep)
            terminated = False
            while not terminated:
                mask = obs["action_mask"]
                if env.game.leading:
                    self.assertEqual(mask[0], 0)
                else:
                    self.assertEqual(mask[0], 1)
                action = int(np.flatnonzero(mask)[0])
                obs, _, terminated, _, info = env.step(action)


if __name__ == "__main__":
    unittest.main()
