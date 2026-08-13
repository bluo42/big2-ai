"""Card representation for Big 2.

A card is an int in [0, 51]:  card = rank * 4 + suit

Ranks (Big-2 order, low to high):
    0:3  1:4  2:5  3:6  4:7  5:8  6:9  7:10  8:J  9:Q  10:K  11:A  12:2

Suits (low to high):
    0: diamonds  1: clubs  2: hearts  3: spades

So card 0 is the 3 of diamonds (lowest card in the game) and card 51 is
the 2 of spades (highest card in the game).  Comparing two singles is
just integer comparison.
"""

from __future__ import annotations

from typing import Iterable, List

Card = int

NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52

RANK_NAMES = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
SUIT_NAMES = ["d", "c", "h", "s"]  # diamonds < clubs < hearts < spades
SUIT_SYMBOLS = ["♦", "♣", "♥", "♠"]

THREE_OF_DIAMONDS: Card = 0
TWO_RANK = 12  # rank index of the 2s
ACE_RANK = 11  # rank index of the aces (highest rank usable in a straight)


def rank(card: Card) -> int:
    return card // 4


def suit(card: Card) -> int:
    return card % 4


def make_card(rank_idx: int, suit_idx: int) -> Card:
    return rank_idx * 4 + suit_idx


def card_name(card: Card, symbols: bool = False) -> str:
    names = SUIT_SYMBOLS if symbols else SUIT_NAMES
    return f"{RANK_NAMES[rank(card)]}{names[suit(card)]}"


def parse_card(text: str) -> Card:
    """Parse strings like '3d', '10s', 'Ah', '2c' into a card int."""
    text = text.strip()
    suit_ch = text[-1].lower()
    rank_ch = text[:-1].upper()
    if rank_ch not in RANK_NAMES:
        raise ValueError(f"unknown rank {rank_ch!r}")
    if suit_ch not in SUIT_NAMES:
        raise ValueError(f"unknown suit {suit_ch!r}")
    return make_card(RANK_NAMES.index(rank_ch), SUIT_NAMES.index(suit_ch))


def hand_to_str(cards: Iterable[Card], symbols: bool = False) -> str:
    return " ".join(card_name(c, symbols) for c in sorted(cards))


def full_deck() -> List[Card]:
    return list(range(NUM_CARDS))
