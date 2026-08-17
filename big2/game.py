"""Big 2 game engine: 2-4 players, tricks, passing, and scoring.

Seats are numbered 0..n-1 and play proceeds in increasing seat order,
which represents counter-clockwise order around a physical table.

Dealing: every player receives 13 cards.  With fewer than 4 players the
remaining cards stay face down and out of play (pagat.com convention).

Trick flow:
- The holder of the lowest card in play (the 3 of diamonds in a 4-player
  game) leads the first trick and the first play must include that card.
- Following players must either beat the combo on the table (same size
  class) or pass.  Passing is always allowed when not leading, even if
  the player could beat the table ("strategic pass").
- With ``RuleConfig(pass_locks=False)`` (house rule, "soft pass") a pass
  only skips the turn — you may play again if the trick comes back
  around — and the trick ends when every other player passes
  consecutively since the last play.  With ``pass_locks=True`` a player
  who passes is locked out for the remainder of the trick.
- When the turn returns to the last player who played, they win the
  trick and lead the next one (any class).  The leader may not pass.

The game ends the moment one player sheds their last card.  Every other
player pays the winner based on cards remaining (see ScoringConfig).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from big2.cards import TWO_RANK, Card, full_deck, rank
from big2.combos import Combo, generate_moves
from big2.rules import DEFAULT_RULES, RuleConfig

NUM_PLAYERS = 4  # default table size
CARDS_PER_PLAYER = 13


@dataclass(frozen=True)
class ScoringConfig:
    """Payment rules applied to each loser at game end.

    Base payment is the number of cards left in the loser's hand,
    scaled by tiered card-count multipliers (the house rule):

    - ``big_hand_double``:  10-12 cards remaining pays double.
    - ``full_hand_triple``: all 13 cards remaining pays triple.

    Legacy modifier kept for experiments (off by default):

    - ``two_modifier``: holding a 2 at game end adds the base again
      (+1x); with ``per_two`` each 2 held adds +1x.
    """

    big_hand_double: bool = True
    full_hand_triple: bool = True
    big_hand_threshold: int = 10
    two_modifier: bool = False
    per_two: bool = False

    def payment(self, cards_left: List[Card]) -> int:
        base = len(cards_left)
        if base == 0:
            return 0
        if self.full_hand_triple and base == CARDS_PER_PLAYER:
            multiplier = 3
        elif self.big_hand_double and base >= self.big_hand_threshold:
            multiplier = 2
        else:
            multiplier = 1
        if self.two_modifier:
            twos = sum(1 for c in cards_left if rank(c) == TWO_RANK)
            if twos:
                multiplier += twos if self.per_two else 1
        return base * multiplier

    def label(self) -> str:
        parts = []
        if self.big_hand_double:
            parts.append(f">={self.big_hand_threshold}x2")
        if self.full_hand_triple:
            parts.append("13x3")
        if self.two_modifier:
            parts.append("per_two" if self.per_two else "two")
        return "+".join(parts) if parts else "plain"


@dataclass
class PlayRecord:
    player: int
    combo: Optional[Combo]  # None = pass
    trick_end: bool = False  # this action closed the trick


class Big2Game:
    """Mutable game state.  Drive it with legal_moves() / step()."""

    def __init__(
        self,
        scoring: Optional[ScoringConfig] = None,
        rules: Optional[RuleConfig] = None,
        num_players: int = NUM_PLAYERS,
        rng: Optional[random.Random] = None,
    ):
        if not 2 <= num_players <= 4:
            raise ValueError("num_players must be 2, 3, or 4")
        self.scoring = scoring or ScoringConfig()
        self.rules = rules or DEFAULT_RULES
        self.num_players = num_players
        self.rng = rng or random.Random()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        deck = full_deck()
        self.rng.shuffle(deck)
        self.hands: List[List[Card]] = [
            sorted(deck[i * CARDS_PER_PLAYER : (i + 1) * CARDS_PER_PLAYER])
            for i in range(self.num_players)
        ]
        self.start_card: Card = min(min(hand) for hand in self.hands)
        self.turn: int = next(
            p for p in range(self.num_players) if self.start_card in self.hands[p]
        )
        self.first_play = True
        self.table_combo: Optional[Combo] = None  # combo to beat, None when leading
        self.table_player: Optional[int] = None  # who played table_combo
        self.passed: List[bool] = [False] * self.num_players
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
        moves = generate_moves(self.hands[p], self.table_combo, self.rules)
        if self.first_play:
            moves = [m for m in moves if self.start_card in m.cards]
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
            if not self.rules.pass_locks:
                # Soft pass: a play reopens the trick for everyone.
                self.passed = [False] * self.num_players
            self.history.append(PlayRecord(player, combo))
            if not hand:
                self._finish(player)
                return

        self._advance_turn()

    def _advance_turn(self) -> None:
        nxt = (self.turn + 1) % self.num_players
        if self.rules.pass_locks:
            while self.passed[nxt]:
                nxt = (nxt + 1) % self.num_players
        # Soft pass needs no skipping: flags reset on every play, so the
        # turn simply walks the table and the trick ends when it reaches
        # the table owner again (a full round of consecutive passes).
        if nxt == self.table_player:
            # Trick won: winner leads fresh.
            self.table_combo = None
            self.table_player = None
            self.passed = [False] * self.num_players
            if self.history:
                self.history[-1].trick_end = True
        self.turn = nxt

    def _finish(self, winner: int) -> None:
        self.winner = winner
        if self.history:
            self.history[-1].trick_end = True
        payments = {
            p: self.scoring.payment(self.hands[p])
            for p in range(self.num_players)
            if p != winner
        }
        self.scores = {p: -pay for p, pay in payments.items()}
        self.scores[winner] = sum(payments.values())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def clone(self, rng: Optional[random.Random] = None) -> "Big2Game":
        """Copy for search/rollouts.  Combos are immutable and shared; the
        last history record is copied because step() may mutate its
        trick_end flag."""
        g = object.__new__(Big2Game)
        g.scoring = self.scoring
        g.rules = self.rules
        g.num_players = self.num_players
        g.rng = rng or random.Random()
        g.hands = [list(h) for h in self.hands]
        g.start_card = self.start_card
        g.turn = self.turn
        g.first_play = self.first_play
        g.table_combo = self.table_combo
        g.table_player = self.table_player
        g.passed = list(self.passed)
        g.history = list(self.history)
        if g.history:
            last = g.history[-1]
            g.history[-1] = PlayRecord(last.player, last.combo, last.trick_end)
        g.played_cards = list(self.played_cards)
        g.winner = self.winner
        g.scores = dict(self.scores) if self.scores else None
        return g

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
