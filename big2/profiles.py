"""Cross-game opponent profiles: guesses about how each opponent plays,
continuously updated for as long as that opponent keeps sitting down.

The within-match OpponentModel (big2/opponents.py) starts from zero
every deal.  This book persists *across* games as an exponential moving
average: every statistic decays by a per-game factor chosen so a game
``half_life_games`` ago carries half the weight of the newest one.
Recent behavior dominates, older behavior fades smoothly — but is never
discarded, so a long-standing read survives a noisy patch.

Profiles are built from public information only — actions taken plus
the end-of-game reveal (cards left, who won) that real table play also
exposes.  ``features()`` returns a fixed vector per opponent that the
neural agents consume as state input:

    [confidence, pass_rate, avg_rank, multi_frac, twos_per_game,
     win_rate, avg_cards_left, fives_per_game]

``confidence`` is 1 - decay^n: 0 for a stranger, 0.5 once half a
half-life's worth of evidence has accumulated, asymptoting to 1.
"""

from __future__ import annotations

from typing import Dict, Hashable, List

import numpy as np

from big2.cards import TWO_RANK, rank
from big2.game import Big2Game

PROFILE_DIM = 8


class _Ema:
    __slots__ = ("n", "weight", "actions", "passes", "rank_sum",
                 "cards_played", "multi", "twos", "fives", "wins",
                 "cards_left")

    def __init__(self):
        self.n = 0  # raw games observed (never decays)
        self.weight = 0.0  # decayed game count
        self.actions = 0.0
        self.passes = 0.0
        self.rank_sum = 0.0
        self.cards_played = 0.0
        self.multi = 0.0
        self.twos = 0.0
        self.fives = 0.0
        self.wins = 0.0
        self.cards_left = 0.0

    def decay(self, d: float) -> None:
        self.weight *= d
        self.actions *= d
        self.passes *= d
        self.rank_sum *= d
        self.cards_played *= d
        self.multi *= d
        self.twos *= d
        self.fives *= d
        self.wins *= d
        self.cards_left *= d


class OpponentProfileBook:
    def __init__(self, half_life_games: int = 500):
        self.half_life = half_life_games
        self._d = 0.5 ** (1.0 / half_life_games)
        self._w: Dict[Hashable, _Ema] = {}

    def observe_game(self, game: Big2Game,
                     seat_keys: Dict[int, Hashable]) -> None:
        """Fold one finished game into the profiles (EMA update).

        ``seat_keys`` maps seats to stable opponent identities (e.g.
        "linear", "target", "human"); seats not listed are ignored.
        """
        if not game.game_over:
            return
        per_seat: Dict[int, List] = {
            s: [0, 0, 0.0, 0, 0, 0, 0] for s in seat_keys
        }  # actions, passes, rank_sum, cards, multi, twos, fives
        for rec in game.history:
            s = rec.player
            if s not in per_seat:
                continue
            row = per_seat[s]
            row[0] += 1
            if rec.combo is None:
                row[1] += 1
                continue
            cards = rec.combo.cards
            row[2] += sum(rank(c) for c in cards)
            row[3] += len(cards)
            if len(cards) > 1:
                row[4] += 1
            if len(cards) == 5:
                row[6] += 1
            row[5] += sum(1 for c in cards if rank(c) == TWO_RANK)

        for s, key in seat_keys.items():
            w = self._w.setdefault(key, _Ema())
            w.decay(self._d)  # newest game gets full weight, past fades
            row = per_seat[s]
            w.n += 1
            w.weight += 1.0
            w.actions += row[0]
            w.passes += row[1]
            w.rank_sum += row[2]
            w.cards_played += row[3]
            w.multi += row[4]
            w.twos += row[5]
            w.fives += row[6]
            w.wins += 1.0 if game.winner == s else 0.0
            w.cards_left += len(game.hands[s])

    def features(self, key: Hashable) -> np.ndarray:
        w = self._w.get(key)
        f = np.zeros(PROFILE_DIM, dtype=np.float32)
        if w is None or w.weight <= 0.0:
            return f
        plays = max(1e-9, w.actions - w.passes)
        f[0] = 1.0 - self._d ** w.n  # confidence: 0.5 at one half-life
        f[1] = w.passes / max(1e-9, w.actions)
        f[2] = w.rank_sum / max(1e-9, w.cards_played) / 12.0
        f[3] = w.multi / plays
        f[4] = min(1.0, w.twos / w.weight / 2.0)
        f[5] = w.wins / w.weight
        f[6] = w.cards_left / w.weight / 13.0
        f[7] = min(1.0, w.fives / w.weight / 4.0)
        return f

    def games_seen(self, key: Hashable) -> int:
        w = self._w.get(key)
        return w.n if w else 0
