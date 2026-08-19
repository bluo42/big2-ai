"""Torch-free inference for the PPO set-attention champion.

Serverless deploys can't carry torch (~2 GB), but the trained net is
tiny — two MLPs, one attention block, a linear head — so this module
reimplements the exact forward pass in numpy.  ``export_numpy`` converts
a trained checkpoint once; ``NumpyPPOPolicy`` then plays byte-identical
moves with no torch dependency (equivalence is unit-tested to 1e-4).

    python -m big2.ppo_numpy               # export the champion
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from big2.combos import Combo
from big2.game import Big2Game
from big2.neural import encode_decision

DEFAULT_NPZ = "big2/policies/ppo_attn_np.npz"

_PARAM_KEYS = [
    "state_mlp.0.weight", "state_mlp.0.bias",
    "state_mlp.2.weight", "state_mlp.2.bias",
    "act_mlp.0.weight", "act_mlp.0.bias",
    "act_mlp.2.weight", "act_mlp.2.bias",
    "attn.in_proj_weight", "attn.in_proj_bias",
    "attn.out_proj.weight", "attn.out_proj.bias",
    "norm.weight", "norm.bias",
    "policy_head.weight", "policy_head.bias",
]


# Second attention block (the 2026 chain nets): pre-norm residual,
# exported only when the checkpoint carries it.
_ATTN2_KEYS = [
    "attn2.in_proj_weight", "attn2.in_proj_bias",
    "attn2.out_proj.weight", "attn2.out_proj.bias",
    "norm2.weight", "norm2.bias",
]


def export_numpy(pt_path: str, npz_path: str = DEFAULT_NPZ) -> None:
    import torch

    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    sd = payload["state_dict"]
    keys = list(_PARAM_KEYS) + [k for k in _ATTN2_KEYS if k in sd]
    arrays = {k.replace(".", "__"): sd[k].numpy() for k in keys}
    arrays["d_model"] = np.array(payload.get("d_model", 192))
    arrays["heads"] = np.array(payload.get("heads", 4))
    np.savez(npz_path, **arrays)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class NumpyPPOPolicy:
    """Greedy argmax inference identical to neural.PPOPolicy."""

    name = "ppo"

    def __init__(self, params: Dict[str, np.ndarray], d_model: int, heads: int):
        self.p = {k: v.astype(np.float64) for k, v in params.items()}
        self.d = d_model
        self.heads = heads

    @classmethod
    def load(cls, path: str = DEFAULT_NPZ) -> "NumpyPPOPolicy":
        data = np.load(path)
        params = {
            k.replace("__", "."): data[k]
            for k in data.files
            if k not in ("d_model", "heads")
        }
        return cls(params, int(data["d_model"]), int(data["heads"]))

    def _attend(self, h: np.ndarray, prefix: str) -> np.ndarray:
        p = self.p
        d, nh = self.d, self.heads
        dh = d // nh
        qkv = h @ p[f"{prefix}.in_proj_weight"].T + p[f"{prefix}.in_proj_bias"]
        q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
        A = h.shape[0]
        q = q.reshape(A, nh, dh).transpose(1, 0, 2) / np.sqrt(dh)
        k = k.reshape(A, nh, dh).transpose(1, 0, 2)
        v = v.reshape(A, nh, dh).transpose(1, 0, 2)
        attn = _softmax(q @ k.transpose(0, 2, 1)) @ v  # (heads, A, dh)
        attn = attn.transpose(1, 0, 2).reshape(A, d)
        return attn @ p[f"{prefix}.out_proj.weight"].T \
            + p[f"{prefix}.out_proj.bias"]

    @staticmethod
    def _layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def _logits(self, state: np.ndarray, acts: np.ndarray) -> np.ndarray:
        p = self.p
        s = _relu(state @ p["state_mlp.0.weight"].T + p["state_mlp.0.bias"])
        s = _relu(s @ p["state_mlp.2.weight"].T + p["state_mlp.2.bias"])
        a = _relu(acts @ p["act_mlp.0.weight"].T + p["act_mlp.0.bias"])
        a = a @ p["act_mlp.2.weight"].T + p["act_mlp.2.bias"]
        h = a + s  # (A, d): action embeddings conditioned on the state

        x = self._layernorm(h + self._attend(h, "attn"),
                            p["norm.weight"], p["norm.bias"])
        if "attn2.in_proj_weight" in p:
            # Pre-norm residual second block, exactly as in Big2Net.
            n2 = self._layernorm(x, p["norm2.weight"], p["norm2.bias"])
            x = x + self._attend(n2, "attn2")

        return (x @ p["policy_head.weight"].T + p["policy_head.bias"])[:, 0]

    def option_scores(self, game: Big2Game, player: int):
        """(options, logits) — the same interface as neural.PPOPolicy,
        so the IS-MCTS agent can use this port as its prior."""
        from big2.features import FEAT_DIM
        from big2.neural import (
            ACT_DIM, ACT_DIM_BEAT, ACT_DIM_V11, STATE_DIM,
        )

        in_dim = self.p["state_mlp.0.weight"].shape[1]
        a_dim = self.p["act_mlp.0.weight"].shape[1]
        options, state, acts = encode_decision(
            game, player, include_profiles=(in_dim != FEAT_DIM),
            include_danger=(a_dim >= ACT_DIM_V11),
            include_plan=(a_dim >= ACT_DIM and in_dim >= STATE_DIM),
            include_beat=(a_dim >= ACT_DIM_BEAT),
        )
        return options, self._logits(state, acts)

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        options, logits = self.option_scores(game, player)
        if len(options) == 1:
            return options[0]
        return options[int(np.argmax(logits))]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="big2/policies/ppo_attn.pt")
    parser.add_argument("--out", default=DEFAULT_NPZ)
    args = parser.parse_args()
    export_numpy(args.checkpoint, args.out)
    print(f"exported {args.checkpoint} -> {args.out}")


if __name__ == "__main__":
    main()
