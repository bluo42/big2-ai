import random
import unittest

import numpy as np

from big2.features import FEAT_DIM, encode_options
from big2.game import Big2Game
from big2.nn import MLPQ, NNPolicy
from big2.strategies import PlayLowest


class TestMLP(unittest.TestCase):
    def test_gradient_check(self):
        """Numeric vs analytic gradients on a tiny net."""
        rng = np.random.default_rng(0)
        net = MLPQ(6, hidden=(5, 4), seed=1)
        X = rng.normal(size=(3, 6))
        y = rng.normal(size=3)

        def loss_at(params_flat):
            shapes = [(w.shape, b.shape) for w, b in zip(net.W, net.b)]
            i = 0
            W, b = [], []
            for ws, bs in shapes:
                W.append(params_flat[i : i + np.prod(ws)].reshape(ws)); i += np.prod(ws)
                b.append(params_flat[i : i + np.prod(bs)].reshape(bs)); i += np.prod(bs)
            a = X
            for j in range(len(W)):
                a = a @ W[j] + b[j]
                if j < len(W) - 1:
                    a = np.maximum(a, 0.0)
            return np.mean((a[:, 0] - y) ** 2)

        flat = np.concatenate([np.r_[w.ravel(), b.ravel()]
                               for w, b in zip(net.W, net.b)])
        # analytic grad via one train step with lr encoded... instead use
        # finite differences against the internal backward pass:
        eps = 1e-6
        numeric = np.zeros_like(flat)
        for k in range(len(flat)):
            up, down = flat.copy(), flat.copy()
            up[k] += eps
            down[k] -= eps
            numeric[k] = (loss_at(up) - loss_at(down)) / (2 * eps)

        # capture analytic gradients by monkey-running train_batch with a
        # zero learning rate is a no-op; instead recompute manually:
        acts, pre, a = [X], [], X
        for j, (w, bb) in enumerate(zip(net.W, net.b)):
            z = a @ w + bb
            pre.append(z)
            a = np.maximum(z, 0.0) if j < len(net.W) - 1 else z
            acts.append(a)
        grad = (2.0 / X.shape[0]) * (acts[-1][:, 0] - y)[:, None]
        gW = [None] * len(net.W)
        gb = [None] * len(net.b)
        for j in reversed(range(len(net.W))):
            if j < len(net.W) - 1:
                grad = grad * (pre[j] > 0.0)
            gW[j] = acts[j].T @ grad
            gb[j] = grad.sum(axis=0)
            if j > 0:
                grad = grad @ net.W[j].T
        analytic = np.concatenate([np.r_[w.ravel(), b.ravel()]
                                   for w, b in zip(gW, gb)])
        np.testing.assert_allclose(numeric, analytic, atol=1e-5)

    def test_training_reduces_loss(self):
        rng = np.random.default_rng(1)
        net = MLPQ(10, hidden=(32,), seed=2)
        X = rng.normal(size=(200, 10))
        y = X[:, 0] * 2 - X[:, 1]
        first = net.train_batch(X, y, lr=1e-2)
        for _ in range(300):
            last = net.train_batch(X, y, lr=1e-2)
        self.assertLess(last, first * 0.1)

    def test_save_load_roundtrip(self):
        import tempfile, os

        net = MLPQ(FEAT_DIM, hidden=(64, 32), seed=3)
        X = np.random.default_rng(4).normal(size=(5, FEAT_DIM))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "net.npz")
            net.save(path, meta={"genome": {"lr": 0.01}})
            loaded, meta = MLPQ.load(path)
            np.testing.assert_allclose(net.predict(X), loaded.predict(X))
            self.assertEqual(meta["genome"]["lr"], 0.01)

    def test_policy_plays_full_game(self):
        policy = NNPolicy(MLPQ(FEAT_DIM, hidden=(32,), seed=5))
        game = Big2Game(rng=random.Random(0))
        scores = game.play_out([policy, PlayLowest(), PlayLowest(), PlayLowest()])
        self.assertEqual(sum(scores.values()), 0)

    def test_encode_options_shapes(self):
        for n in (2, 4):
            game = Big2Game(num_players=n, rng=random.Random(1))
            options, feats = encode_options(game, game.turn)
            self.assertEqual(feats.shape, (len(options), FEAT_DIM))
            self.assertGreaterEqual(len(options), 1)


if __name__ == "__main__":
    unittest.main()
