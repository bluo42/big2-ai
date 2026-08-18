"""What an opponent's hand can *do*, not just which cards it holds.

A 52-card posterior says where the cards are.  It does not say whether
anyone can answer the pair of jacks you were about to lead — that is a
question about *combinations*, and combinations are what Big 2 is played
in.  Two hands with identical card-level probabilities can differ
completely in what they threaten: thirteen scattered singles beat almost
nothing, while the same count arranged as two pairs and a flush beats
most of what you can lead.

So each opponent also gets a **shape profile**: the probability that
their hand contains each kind of playable unit.

    pair of rank r            13   can they answer a pair?
    triple of rank r          13   ...a triple, and the core of a house
    four of a kind of rank r  13   the bombs
    flush in suit s            4   five of a suit
    four-flush in suit s       4   one card short — where a flush is
                                    likely to appear next trick
    straight topped at rank r 13
    straight flush in suit s   4
    full house (any)           1
    full house by triple rank 13
    aggregates                 4   any pair / triple / 5-card / bomb
    holds a card above rank r 13   the curve that decides every single

That last block is the one that matters most in the endgame: "can they
beat this card" is exactly ``holds_above[rank]``, so the profile turns
a card posterior into a direct answer about *your* candidate move.

Truth is computed with the engine's own move generator, so what counts
as a straight or a flush is whatever the rules say — no second
implementation to drift out of step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from big2.cards import Card, rank, suit
from big2.combos import ComboType, generate_moves
from big2.rules import DEFAULT_RULES, RuleConfig

NUM_RANKS = 13
NUM_SUITS = 4

# Layout of the profile vector.
PAIR = slice(0, 13)
TRIPLE = slice(13, 26)
QUAD = slice(26, 39)
FLUSH = slice(39, 43)
FOUR_FLUSH = slice(43, 47)
STRAIGHT = slice(47, 60)
STRAIGHT_FLUSH = slice(60, 64)
FULL_HOUSE_ANY = 64
FULL_HOUSE_RANK = slice(65, 78)
ANY_PAIR, ANY_TRIPLE, ANY_FIVE, ANY_BOMB = 78, 79, 80, 81
HOLDS_ABOVE = slice(82, 95)
SHAPE_DIM = 95


def shape_profile(
    hand: Sequence[Card], rules: RuleConfig = DEFAULT_RULES
) -> np.ndarray:
    """Exact indicator profile of one known hand."""
    f = np.zeros(SHAPE_DIM, dtype=np.float32)
    if not hand:
        return f
    hand = list(hand)
    by_rank: Dict[int, List[Card]] = {}
    by_suit: Dict[int, List[Card]] = {}
    for c in hand:
        by_rank.setdefault(rank(c), []).append(c)
        by_suit.setdefault(suit(c), []).append(c)

    for r, cs in by_rank.items():
        if len(cs) >= 2:
            f[PAIR.start + r] = 1.0
        if len(cs) >= 3:
            f[TRIPLE.start + r] = 1.0
        if len(cs) >= 4:
            f[QUAD.start + r] = 1.0
    for s, cs in by_suit.items():
        if len(cs) >= 5:
            f[FLUSH.start + s] = 1.0
        if len(cs) == 4:
            f[FOUR_FLUSH.start + s] = 1.0

    # Five-card shapes come from the engine, so the rules stay single-sourced.
    for m in generate_moves(hand, None, rules):
        if len(m) != 5:
            continue
        top = rank(max(m.cards))
        if m.type == ComboType.STRAIGHT:
            f[STRAIGHT.start + top] = 1.0
        elif m.type == ComboType.STRAIGHT_FLUSH:
            f[STRAIGHT.start + top] = 1.0
            f[STRAIGHT_FLUSH.start + suit(m.cards[0])] = 1.0
        elif m.type == ComboType.FULL_HOUSE:
            f[FULL_HOUSE_ANY] = 1.0
            counts: Dict[int, int] = {}
            for c in m.cards:
                counts[rank(c)] = counts.get(rank(c), 0) + 1
            for r, n in counts.items():
                if n == 3:
                    f[FULL_HOUSE_RANK.start + r] = 1.0
        elif m.type == ComboType.FOUR_OF_A_KIND:
            f[ANY_BOMB] = 1.0

    f[ANY_PAIR] = 1.0 if f[PAIR].any() else 0.0
    f[ANY_TRIPLE] = 1.0 if f[TRIPLE].any() else 0.0
    f[ANY_FIVE] = 1.0 if (
        f[FLUSH].any() or f[STRAIGHT].any() or f[FULL_HOUSE_ANY]
        or f[ANY_BOMB]
    ) else 0.0
    if f[QUAD].any():
        f[ANY_BOMB] = 1.0

    top_rank = max(rank(c) for c in hand)
    for r in range(NUM_RANKS):
        f[HOLDS_ABOVE.start + r] = 1.0 if top_rank > r else 0.0
    return f


def profile_from_worlds(
    worlds: Sequence, opponents: Sequence[int],
    rules: RuleConfig = DEFAULT_RULES,
) -> np.ndarray:
    """(3, SHAPE_DIM) analytic profile: the weighted share of sampled
    worlds in which each opponent's hand has each shape."""
    out = np.zeros((3, SHAPE_DIM), dtype=np.float32)
    total = 0.0
    for world, w in worlds:
        if w <= 0.0:
            continue
        total += w
        for j, p in enumerate(opponents[:3]):
            hand = world.get(p)
            if hand:
                out[j] += w * shape_profile(hand, rules)
    return out / total if total else out


def beat_probability(
    profile: np.ndarray, move_type: ComboType, top_rank: int
) -> float:
    """Read a shape profile as "can this hand answer that move?".

    Approximate by construction — it asks whether the hand holds a unit
    of the right kind at a higher rank, which is the question that
    decides the vast majority of tricks.
    """
    if move_type == ComboType.SINGLE:
        return float(profile[HOLDS_ABOVE.start + top_rank])
    if move_type == ComboType.PAIR:
        higher = profile[PAIR][top_rank + 1:]
        return float(1.0 - np.prod(1.0 - np.clip(higher, 0.0, 1.0)))
    if move_type == ComboType.TRIPLE:
        higher = profile[TRIPLE][top_rank + 1:]
        return float(1.0 - np.prod(1.0 - np.clip(higher, 0.0, 1.0)))
    # any five-card answer, plus bombs
    five = float(profile[ANY_FIVE])
    return float(np.clip(max(five, profile[ANY_BOMB]), 0.0, 1.0))
