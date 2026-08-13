"""Gymnasium environment for Big 2.

Single-agent view: the agent controls one seat, the other three seats are
driven by fixed opponent policies (any Strategy).  The action space is a
fixed-size Discrete space over slots into the current legal-move list:

    action 0          = PASS (legal only when following)
    action 1..N       = the N legal combos, sorted weakest-first
    action N+1..MAX   = padding (always illegal)

An action mask is exposed both in the observation dict and via
``action_masks()`` (compatible with sb3-contrib's MaskablePPO).  Because
moves are sorted weakest-first, slot semantics are stable in a useful
way: the lowest-numbered legal combo slot is always the weakest play and
the highest is the strongest.

Observation (Dict):
    "obs":  float32 vector, see _build_obs for layout.
    "action_mask": int8 vector of length MAX_ACTIONS.

Reward: 0 on every step until the game ends; the terminal reward is the
agent's score under the configured ScoringConfig (positive if the agent
won, negative cards-payment otherwise).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from big2.cards import NUM_CARDS, Card
from big2.combos import Combo, ComboType
from big2.game import CARDS_PER_PLAYER, NUM_PLAYERS, Big2Game, ScoringConfig
from big2.strategies import PlayLowest, Strategy

# Generous upper bound on simultaneous legal moves (flush-heavy hands can
# exceed 1300 five-card combos); slot 0 is reserved for PASS.
MAX_ACTIONS = 2048

OBS_SIZE = (
    NUM_CARDS  # my hand
    + NUM_CARDS  # combo on the table
    + (len(ComboType) + 1)  # table combo type one-hot (+1 for "none")
    + NUM_CARDS  # all cards played so far
    + (NUM_PLAYERS - 1)  # opponent card counts / 13, in seat order after me
    + (NUM_PLAYERS - 1)  # opponent passed flags
    + NUM_PLAYERS  # who owns the table combo, relative one-hot (+none)
    + 1  # am I leading
)


class Big2Env(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponents: Optional[Sequence[Strategy]] = None,
        scoring: Optional[ScoringConfig] = None,
        agent_seat: Optional[int] = None,
        invalid_action: str = "error",  # or "fallback"
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.opponent_pool = list(opponents) if opponents else [
            PlayLowest(), PlayLowest(), PlayLowest()
        ]
        if len(self.opponent_pool) != NUM_PLAYERS - 1:
            raise ValueError(f"need exactly {NUM_PLAYERS - 1} opponents")
        self.scoring = scoring or ScoringConfig()
        self.fixed_agent_seat = agent_seat
        self.invalid_action = invalid_action
        self.rng = random.Random(seed)
        self.game = Big2Game(scoring=self.scoring, rng=random.Random(self.rng.random()))

        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(MAX_ACTIONS),
            }
        )
        self.agent_seat = 0
        self.seat_policies: Dict[int, Strategy] = {}
        self._legal: List[Combo] = []

    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.agent_seat = (
            self.fixed_agent_seat
            if self.fixed_agent_seat is not None
            else self.rng.randrange(NUM_PLAYERS)
        )
        opp_seats = [p for p in range(NUM_PLAYERS) if p != self.agent_seat]
        self.seat_policies = dict(zip(opp_seats, self.opponent_pool))
        self.game.reset(seed=self.rng.randrange(2**31))
        self._run_opponents()
        obs = self._build_obs()
        return obs, self._info()

    def step(self, action: int):
        move = self._decode_action(int(action))
        self.game.step(move)
        self._run_opponents()
        terminated = self.game.game_over
        reward = 0.0
        if terminated:
            reward = float(self.game.scores[self.agent_seat])
        return self._build_obs(), reward, terminated, False, self._info()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        if not self.game.game_over and self.game.turn == self.agent_seat:
            if self.game.can_pass():
                mask[0] = 1
            mask[1 : 1 + len(self._legal)] = 1
        return mask

    def render(self):
        g = self.game
        lines = [
            f"turn: seat {g.turn} (agent seat {self.agent_seat})",
            f"table: {g.table_combo} by seat {g.table_player}",
        ]
        for p in range(NUM_PLAYERS):
            tag = "*" if p == self.agent_seat else " "
            passed = " [passed]" if g.passed[p] else ""
            lines.append(f"{tag}seat {p}: {len(g.hands[p])} cards{passed}")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _decode_action(self, action: int) -> Optional[Combo]:
        mask = self.action_masks()
        if action >= MAX_ACTIONS or not mask[action]:
            if self.invalid_action == "fallback":
                action = 1 if len(self._legal) > 0 else 0
            else:
                raise ValueError(f"invalid action {action}")
        return None if action == 0 else self._legal[action - 1]

    def _run_opponents(self) -> None:
        while not self.game.game_over and self.game.turn != self.agent_seat:
            policy = self.seat_policies[self.game.turn]
            self.game.step(policy.select(self.game, self.game.turn))
        self._legal = (
            self.game.legal_moves(self.agent_seat) if not self.game.game_over else []
        )

    def _build_obs(self) -> Dict[str, np.ndarray]:
        g = self.game
        me = self.agent_seat
        vec = np.zeros(OBS_SIZE, dtype=np.float32)
        i = 0

        for c in g.hands[me]:
            vec[i + c] = 1.0
        i += NUM_CARDS

        if g.table_combo is not None:
            for c in g.table_combo.cards:
                vec[i + c] = 1.0
        i += NUM_CARDS

        type_idx = 0 if g.table_combo is None else int(g.table_combo.type) + 1
        vec[i + type_idx] = 1.0
        i += len(ComboType) + 1

        for c in g.played_cards:
            vec[i + c] = 1.0
        i += NUM_CARDS

        others = [(me + k) % NUM_PLAYERS for k in range(1, NUM_PLAYERS)]
        for j, p in enumerate(others):
            vec[i + j] = len(g.hands[p]) / CARDS_PER_PLAYER
        i += NUM_PLAYERS - 1
        for j, p in enumerate(others):
            vec[i + j] = 1.0 if g.passed[p] else 0.0
        i += NUM_PLAYERS - 1

        if g.table_player is None:
            vec[i] = 1.0
        else:
            rel = (g.table_player - me) % NUM_PLAYERS  # 1..3
            vec[i + rel] = 1.0
        i += NUM_PLAYERS

        vec[i] = 1.0 if g.leading else 0.0

        return {"obs": vec, "action_mask": self.action_masks()}

    def _info(self) -> Dict:
        info: Dict = {
            "legal_moves": [m.cards for m in self._legal],
            "agent_seat": self.agent_seat,
        }
        if self.game.game_over:
            info["scores"] = dict(self.game.scores)
            info["winner"] = self.game.winner
        return info
