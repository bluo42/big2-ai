"""Big 2 (Deuces) card game engine, Gymnasium environment, and strategies.

Rules implemented (see README.md for details):
- 4 players, 13 cards each, counter-clockwise play (seat order 0 -> 1 -> 2 -> 3).
- Rank order 3 < 4 < ... < K < A < 2; suit order diamonds < clubs < hearts < spades.
- Holder of the 3 of diamonds leads the first trick and must include it.
- Soft pass by default: passing only skips your turn (RuleConfig can
  enable per-trick lock-out).
- Legal classes: singles, pairs, and 5-card poker hands (no lone triples).
- First player to shed all cards wins; losers pay cards-remaining with
  configurable modifiers (holding a 2, holding >= 10 cards).
"""

from big2.cards import Card, card_name, hand_to_str
from big2.combos import Combo, ComboType, classify, beats, generate_moves
from big2.game import Big2Game, ScoringConfig
from big2.rules import RuleConfig

__all__ = [
    "Card",
    "card_name",
    "hand_to_str",
    "Combo",
    "ComboType",
    "classify",
    "beats",
    "generate_moves",
    "Big2Game",
    "ScoringConfig",
    "RuleConfig",
]
