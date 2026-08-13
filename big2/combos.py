"""Combo (play) classification, comparison, and legal-move generation.

Legal classes: singles, pairs, and 5-card poker hands.  Triples cannot be
played alone under the house rules; pass ``RuleConfig(allow_triples=True)``
to enable them as their own size class (compared by rank).

5-card hierarchy (low to high):
    straight < flush < full house < four of a kind (+ kicker) < straight flush

Tiebreakers within a type:
- single:      rank, then suit.
- pair:        rank, then highest suit present in the pair.
- straight:    rank of the highest card, then suit of that card.
- flush:       suit first, then highest card, next card, ... (a spade flush
               beats any heart flush regardless of ranks).  With
               ``RuleConfig(flush_rank_first=True)`` ranks compare before
               the suit (poker style).
- full house:  rank of the triple.
- four kind:   rank of the four (kicker never matters).
- str. flush:  rank of the highest card, then suit (== the flush suit).

Straights use the Big-2 rank ordering restricted to 3..A: five consecutive
ranks among 3,4,...,K,A.  2s cannot appear in a straight and wrap-around
(e.g. K-A-3-4-5) is not allowed.  The highest straight is 10-J-Q-K-A.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from big2.cards import ACE_RANK, Card, hand_to_str, rank, suit
from big2.rules import DEFAULT_RULES, RuleConfig


class ComboType(IntEnum):
    SINGLE = 0
    PAIR = 1
    STRAIGHT = 2
    FLUSH = 3
    FULL_HOUSE = 4
    FOUR_OF_A_KIND = 5
    STRAIGHT_FLUSH = 6
    TRIPLE = 7  # appended so 5-card hierarchy values above stay stable


FIVE_CARD_TYPES = (
    ComboType.STRAIGHT,
    ComboType.FLUSH,
    ComboType.FULL_HOUSE,
    ComboType.FOUR_OF_A_KIND,
    ComboType.STRAIGHT_FLUSH,
)


@dataclass(frozen=True, order=False)
class Combo:
    """An immutable, classified play.

    ``sort_key`` totally orders combos of the same size class:
    for 5-card combos it is (type, type-specific tiebreak key) so a
    flush always outranks a straight, etc.
    """

    type: ComboType
    cards: Tuple[Card, ...]
    sort_key: Tuple = field(compare=False)

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return f"{self.type.name}[{hand_to_str(self.cards)}]"

    __repr__ = __str__


def _rank_counts(cards: Sequence[Card]) -> Dict[int, List[Card]]:
    groups: Dict[int, List[Card]] = {}
    for c in cards:
        groups.setdefault(rank(c), []).append(c)
    return groups


def _straight_high(cards: Sequence[Card]) -> Optional[Card]:
    """Return the highest card if ``cards`` form a straight, else None."""
    ranks = sorted(rank(c) for c in cards)
    if len(set(ranks)) != 5 or ranks[-1] > ACE_RANK:
        return None  # duplicates, or contains a 2
    if ranks[-1] - ranks[0] != 4:
        return None
    return max(cards)  # card int order == (rank, suit) order


def classify(
    cards: Iterable[Card], rules: RuleConfig = DEFAULT_RULES
) -> Optional[Combo]:
    """Classify a set of cards as a legal combo, or None if unplayable."""
    cs = tuple(sorted(cards))
    n = len(cs)
    if n == 1:
        return Combo(ComboType.SINGLE, cs, (cs[0],))
    if n == 2:
        if rank(cs[0]) == rank(cs[1]):
            return Combo(ComboType.PAIR, cs, (rank(cs[1]), suit(cs[1])))
        return None
    if n == 3:
        if rules.allow_triples and len({rank(c) for c in cs}) == 1:
            return Combo(ComboType.TRIPLE, cs, (rank(cs[2]), suit(cs[2])))
        return None
    if n != 5:
        return None

    is_flush = len({suit(c) for c in cs}) == 1
    high = _straight_high(cs)
    if high is not None and is_flush:
        key = (ComboType.STRAIGHT_FLUSH, rank(high), suit(high))
        return Combo(ComboType.STRAIGHT_FLUSH, cs, key)

    groups = _rank_counts(cs)
    counts = sorted(len(g) for g in groups.values())
    if counts == [1, 4]:
        quad_rank = next(r for r, g in groups.items() if len(g) == 4)
        return Combo(ComboType.FOUR_OF_A_KIND, cs, (ComboType.FOUR_OF_A_KIND, quad_rank))
    if counts == [2, 3]:
        triple_rank = next(r for r, g in groups.items() if len(g) == 3)
        return Combo(ComboType.FULL_HOUSE, cs, (ComboType.FULL_HOUSE, triple_rank))
    if is_flush:
        ranks_desc = tuple(sorted((rank(c) for c in cs), reverse=True))
        if rules.flush_rank_first:
            key = (ComboType.FLUSH, ranks_desc, suit(cs[0]))
        else:
            key = (ComboType.FLUSH, suit(cs[0]), ranks_desc)
        return Combo(ComboType.FLUSH, cs, key)
    if high is not None:
        return Combo(ComboType.STRAIGHT, cs, (ComboType.STRAIGHT, rank(high), suit(high)))
    return None


def beats(challenger: Combo, incumbent: Combo) -> bool:
    """True if ``challenger`` may be played on top of ``incumbent``.

    Combos must be the same size class (single vs single, pair vs pair,
    triple vs triple when enabled, 5-card vs 5-card).  Within the 5-card
    class the type hierarchy applies; there are no bombs across size
    classes.
    """
    if len(challenger) != len(incumbent):
        return False
    if len(challenger) < 5 and challenger.type != incumbent.type:
        return False
    return challenger.sort_key > incumbent.sort_key


def _five_card_candidates(hand: Sequence[Card]) -> List[Tuple[Card, ...]]:
    """Enumerate all card sets that could form a 5-card combo."""
    candidates = set()
    groups = _rank_counts(hand)

    # Straights (includes straight flushes): one card per rank over runs of 5.
    present = sorted(r for r in groups if r <= ACE_RANK)
    present_set = set(present)
    for start in present:
        run = [start + i for i in range(5)]
        if all(r in present_set for r in run):
            for choice in itertools.product(*(groups[r] for r in run)):
                candidates.add(tuple(sorted(choice)))

    # Flushes: any 5 cards of one suit.
    by_suit: Dict[int, List[Card]] = {}
    for c in hand:
        by_suit.setdefault(suit(c), []).append(c)
    for cards_in_suit in by_suit.values():
        if len(cards_in_suit) >= 5:
            for combo in itertools.combinations(cards_in_suit, 5):
                candidates.add(tuple(sorted(combo)))

    # Full houses: triple of one rank + pair of another.
    triple_ranks = [r for r, g in groups.items() if len(g) >= 3]
    pair_ranks = [r for r, g in groups.items() if len(g) >= 2]
    for tr in triple_ranks:
        for triple in itertools.combinations(groups[tr], 3):
            for pr in pair_ranks:
                if pr == tr:
                    continue
                for pair in itertools.combinations(groups[pr], 2):
                    candidates.add(tuple(sorted(triple + pair)))

    # Four of a kind + kicker.
    quad_ranks = [r for r, g in groups.items() if len(g) == 4]
    for qr in quad_ranks:
        quad = tuple(groups[qr])
        for kicker in hand:
            if rank(kicker) != qr:
                candidates.add(tuple(sorted(quad + (kicker,))))

    return list(candidates)


def generate_moves(
    hand: Sequence[Card],
    to_beat: Optional[Combo] = None,
    rules: RuleConfig = DEFAULT_RULES,
) -> List[Combo]:
    """All legal combos from ``hand``, sorted weakest-first.

    ``to_beat`` is the combo currently on the table (None when leading).
    Passing is not included here; the game layer decides when passing is
    allowed.
    """
    moves: List[Combo] = []
    need = len(to_beat) if to_beat is not None else None

    if need in (None, 1):
        moves.extend(classify((c,)) for c in hand)

    if need in (None, 2):
        for group in _rank_counts(hand).values():
            for pair in itertools.combinations(group, 2):
                moves.append(classify(pair))

    if rules.allow_triples and need in (None, 3):
        for group in _rank_counts(hand).values():
            for triple in itertools.combinations(group, 3):
                moves.append(classify(triple, rules))

    if need in (None, 5):
        for cand in _five_card_candidates(hand):
            combo = classify(cand, rules)
            if combo is not None:
                moves.append(combo)

    if to_beat is not None:
        moves = [m for m in moves if beats(m, to_beat)]

    moves.sort(key=lambda m: (len(m), m.sort_key))
    return moves
