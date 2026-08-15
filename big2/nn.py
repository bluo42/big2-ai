"""Small numpy MLP for Q(s, a) regression, plus the NNPolicy wrapper.

The evolutionary trainer (big2/evolve.py) varies depth and width per
agent, so the network is architecture-parameterized: ``hidden`` may be
(64,), (128, 64), (256, 128, 64), ...  ReLU activations, linear output,
Adam updates on minibatch MSE.  Deliberately dependency-free — at this
scale (inputs ~250, batches ~30) numpy matmuls are faster than framework
overhead, and the encoding carries over unchanged to a torch upgrade.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class MLPQ:
    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 64),
                 seed: int = 0):
        self.in_dim = in_dim
        self.hidden = tuple(int(h) for h in hidden)
        rng = np.random.default_rng(seed)
        sizes = [in_dim, *self.hidden, 1]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)  # He init for ReLU
            self.W.append(rng.normal(0.0, scale, (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        # Adam state
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Q values for a batch of encoded (state, action) rows -> (B,)."""
        a = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            a = a @ w + b
            if i < len(self.W) - 1:
                a = np.maximum(a, 0.0)
        return a[:, 0]

    def train_batch(self, X: np.ndarray, y: np.ndarray, lr: float,
                    beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8) -> float:
        """One Adam step on MSE; returns the batch loss."""
        acts = [X]
        pre: List[np.ndarray] = []
        a = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            pre.append(z)
            a = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(a)

        q = acts[-1][:, 0]
        err = q - y
        loss = float(np.mean(err**2))
        B = X.shape[0]
        grad = (2.0 / B) * err[:, None]  # dL/d(output)

        gW = [np.zeros_like(w) for w in self.W]
        gb = [np.zeros_like(b) for b in self.b]
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                grad = grad * (pre[i] > 0.0)
            gW[i] = acts[i].T @ grad
            gb[i] = grad.sum(axis=0)
            if i > 0:
                grad = grad @ self.W[i].T

        self._t += 1
        t = self._t
        for i in range(len(self.W)):
            for g, p, m, v in (
                (gW[i], self.W[i], self._mW[i], self._vW[i]),
                (gb[i], self.b[i], self._mb[i], self._vb[i]),
            ):
                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * g * g
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return loss

    # ------------------------------------------------------------------

    def clone(self) -> "MLPQ":
        other = MLPQ(self.in_dim, self.hidden, seed=0)
        other.W = [w.copy() for w in self.W]
        other.b = [b.copy() for b in self.b]
        return other

    def copy_weights_from(self, other: "MLPQ") -> None:
        """Adopt another net's parameters (architectures must match).
        Adam moments reset — the copier starts a fresh optimization."""
        if other.hidden != self.hidden or other.in_dim != self.in_dim:
            raise ValueError("architecture mismatch")
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def save(self, path: str, meta: Optional[Dict] = None) -> None:
        arrays = {"in_dim": np.array(self.in_dim), "hidden": np.array(self.hidden)}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        arrays["meta"] = np.array(json.dumps(meta or {}))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> Tuple["MLPQ", Dict]:
        data = np.load(path, allow_pickle=False)
        net = cls(int(data["in_dim"]), tuple(int(h) for h in data["hidden"]))
        net.W = [data[f"W{i}"] for i in range(len(net.W))]
        net.b = [data[f"b{i}"] for i in range(len(net.b))]
        meta = json.loads(str(data["meta"]))
        return net, meta


class NNPolicy:
    """Greedy argmax over MLP Q values; a drop-in Strategy."""

    name = "evo"

    def __init__(self, net: MLPQ):
        self.net = net

    def select(self, game, player):
        from big2.features import DecisionContext, encode_options

        options, feats = encode_options(game, player)
        if len(options) == 1:
            return options[0]
        return options[int(np.argmax(self.net.predict(feats)))]

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def load(cls, path: str) -> "NNPolicy":
        net, _ = MLPQ.load(path)
        return cls(net)
