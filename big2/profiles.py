"""Cross-game opponent profiles: guesses about how each opponent plays,
updated for as long as the same opponent keeps sitting at the table.

The within-match OpponentModel (big2/opponents.py) starts from zero
every deal.  This book persists *across* games: after each game it
folds every opponent's public actions and end-state into a windowed
profile keyed by opponent identity, and the window hard-refreshes after
``refresh_games`` observations of that opponent (default ~500) so stale
reads on a drifting opponent age out.

Profiles are built from public information only — actions taken plus
the end-of-game reveal (cards left, who won) that real table play also
exposes.  ``features()`` returns a fixed vector per opponent that the
neural agents consume as state input:

    [confidence, pass_rate, avg_rank, multi_frac, twos_per_game,
     win_rate, avg_cards_left, fives_per_game]
"""

from __future__ import annotations

from typing import Dict, Hashable, List

import numpy as np

from big2.cards import TWO_RANK, rank
from big2.game import Big2Game

PROFILE_DIM = 8


class _Window:
    __slots__ = ("games", "actions", "passes", "rank_sum", "cards_played",
                 "multi", "twos", "fives", "wins", "cards_left")

    def __init__(self):
        self.games = 0
        self.actions = 0
        self.passes = 0
        self.rank_sum = 0.0
        self.cards_played = 0
        self.multi = 0
        self.twos = 0
        self.fives = 0
        self.wins = 0
        self.cards_left = 0


class OpponentProfileBook:
    def __init__(self, refresh_games: int = 500):
        self.refresh_games = refresh_games
        self._w: Dict[Hashable, _Window] = {}

    def observe_game(self, game: Big2Game,
                     seat_keys: Dict[int, Hashable]) -> None:
        """Fold one finished game into the profiles.

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
            w = self._w.get(key)
            if w is None or w.games >= self.refresh_games:
                w = self._w[key] = _Window()  # scheduled refresh
            row = per_seat[s]
            w.games += 1
            w.actions += row[0]
            w.passes += row[1]
            w.rank_sum += row[2]
            w.cards_played += row[3]
            w.multi += row[4]
            w.twos += row[5]
            w.fives += row[6]
            w.wins += 1 if game.winner == s else 0
            w.cards_left += len(game.hands[s])

    def features(self, key: Hashable) -> np.ndarray:
        w = self._w.get(key)
        f = np.zeros(PROFILE_DIM, dtype=np.float32)
        if w is None or w.games == 0:
            return f
        plays = max(1, w.actions - w.passes)
        f[0] = min(1.0, w.games / self.refresh_games)  # confidence
        f[1] = w.passes / max(1, w.actions)
        f[2] = w.rank_sum / max(1, w.cards_played) / 12.0
        f[3] = w.multi / plays
        f[4] = min(1.0, w.twos / w.games / 2.0)
        f[5] = w.wins / w.games
        f[6] = w.cards_left / w.games / 13.0
        f[7] = min(1.0, w.fives / w.games / 4.0)
        return f

    def games_seen(self, key: Hashable) -> int:
        w = self._w.get(key)
        return w.games if w else 0
