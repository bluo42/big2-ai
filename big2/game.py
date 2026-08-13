"""Big 2 game engine: 4 players, tricks, passing, and scoring.

Seats are numbered 0-3 and play proceeds 0 -> 1 -> 2 -> 3 -> 0, which
represents counter-clockwise order around a physical table.

Trick flow:
- The holder of the 3 of diamonds leads the first trick and the first
  play must include the 3 of diamonds.
- Following players must either beat the combo on the table (same size
  class) or pass.  Passing is always allowed when not leading, even if
  the player could beat the table ("strategic pass").
- A player who passes is locked out for the remainder of the trick.
- When everyone else has passed, the last player to play wins the trick
  and leads the next one (any class).  The leader may not pass.

The game ends the moment one player sheds their last card.  Every other
player pays the winner based on cards remaining, with configurable
modifiers (see ScoringConfig).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from big2.cards import THREE_OF_DIAMONDS, TWO_RANK, Card, full_deck, rank
from big2.combos import Combo, generate_moves

NUM_PLAYERS = 4
CARDS_PER_PLAYER = 13


@dataclass(frozen=True)
class ScoringConfig:
    """Payment rules applied to each loser at game end.

    Base payment is the number of cards left in the loser's hand.
    Each active modifier adds the base again (i.e. acts as +1x):

    - ``two_modifier``:      loser still holds at least one 2.
    - ``per_two``:           count each 2 held as its own +1x (only
                             meaningful when ``two_modifier`` is True).
    - ``big_hand_modifier``: loser holds >= ``big_hand_threshold`` cards.

    With both modifiers active, a loser holding one 2 and 11 cards pays
    11 * (1 + 1 + 1) = 33.
    """

    two_modifier: bool = True
    per_two: bool = False
    big_hand_modifier: bool = True
    big_hand_threshold: int = 10

    def payment(self, cards_left: List[Card]) -> int:
        base = len(cards_left)
        if base == 0:
            return 0
        multiplier = 1
        if self.two_modifier:
            twos = sum(1 for c in cards_left if rank(c) == TWO_RANK)
            if twos:
                multiplier += twos if self.per_two else 1
        if self.big_hand_modifier and base >= self.big_hand_threshold:
            multiplier += 1
        return base * multiplier

    def label(self) -> str:
        parts = []
        if self.two_modifier:
            parts.append("per_two" if self.per_two else "two")
        if self.big_hand_modifier:
            parts.append(f">={self.big_hand_threshold}cards")
        return "+".join(parts) if parts else "plain"


@dataclass
class PlayRecord:
    player: int
    combo: Optional[Combo]  # None = pass


class Big2Game:
    """Mutable game state.  Drive it with legal_moves() / step()."""

    def __init__(
        self,
        scoring: Optional[ScoringConfig] = None,
        rng: Optional[random.Random] = None,
    ):
        self.scoring = scoring or ScoringConfig()
        self.rng = rng or random.Random()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        deck = full_deck()
        self.rng.shuffle(deck)
        self.hands: List[List[Card]] = [
            sorted(deck[i * CARDS_PER_PLAYER : (i + 1) * CARDS_PER_PLAYER])
            for i in range(NUM_PLAYERS)
        ]
        self.turn: int = next(
            p for p in range(NUM_PLAYERS) if THREE_OF_DIAMONDS in self.hands[p]
        )
        self.first_play = True
        self.table_combo: Optional[Combo] = None  # combo to beat, None when leading
        self.table_player: Optional[int] = None  # who played table_combo
        self.passed: List[bool] = [False] * NUM_PLAYERS  # locked out this trick
        self.history: List[PlayRecord] = []
        self.played_cards: List[Card] = []
        self.winner: Optional[int] = None
        self.scores: Optional[Dict[int, int]] = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def game_over(self) -> bool:
        return self.winner is not None

    @property
    def leading(self) -> bool:
        """True if the current player is leading a fresh trick."""
        return self.table_combo is None

    def can_pass(self) -> bool:
        return not self.leading

    def legal_moves(self, player: Optional[int] = None) -> List[Combo]:
        """Legal combos for the player to move (pass excluded; see can_pass).

        When leading, the list is never empty; when following it may be
        empty, in which case the player must pass.
        """
        if self.game_over:
            return []
        p = self.turn if player is None else player
        if p != self.turn:
            return []
        moves = generate_moves(self.hands[p], self.table_combo)
        if self.first_play:
            moves = [m for m in moves if THREE_OF_DIAMONDS in m.cards]
        return moves

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def step(self, combo: Optional[Combo]) -> None:
        """Apply a move for the current player.  ``None`` means pass."""
        if self.game_over:
            raise RuntimeError("game is over")
        player = self.turn

        if combo is None:
            if not self.can_pass():
                raise ValueError("cannot pass when leading a trick")
            self.passed[player] = True
            self.history.append(PlayRecord(player, None))
        else:
            legal = self.legal_moves(player)
            if combo not in legal:
                raise ValueError(f"illegal move {combo} for player {player}")
            hand = self.hands[player]
            for c in combo.cards:
                hand.remove(c)
            self.played_cards.extend(combo.cards)
            self.table_combo = combo
            self.table_player = player
            self.first_play = False
            self.history.append(PlayRecord(player, combo))
            if not hand:
                self._finish(player)
                return

        self._advance_turn()

    def _advance_turn(self) -> None:
        nxt = (self.turn + 1) % NUM_PLAYERS
        while self.passed[nxt]:
            nxt = (nxt + 1) % NUM_PLAYERS
        if nxt == self.table_player:
            # Everyone else passed: trick won, winner leads fresh.
            self.table_combo = None
            self.table_player = None
            self.passed = [False] * NUM_PLAYERS
        self.turn = nxt

    def _finish(self, winner: int) -> None:
        self.winner = winner
        payments = {
            p: self.scoring.payment(self.hands[p])
            for p in range(NUM_PLAYERS)
            if p != winner
        }
        self.scores = {p: -pay for p, pay in payments.items()}
        self.scores[winner] = sum(payments.values())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def play_out(self, policies, max_steps: int = 10_000) -> Dict[int, int]:
        """Run the game to completion with one policy per seat."""
        steps = 0
        while not self.game_over:
            policy = policies[self.turn]
            move = policy.select(self, self.turn)
            self.step(move)
            steps += 1
            if steps > max_steps:
                raise RuntimeError("game did not terminate")
        assert self.scores is not None
        return self.scores
