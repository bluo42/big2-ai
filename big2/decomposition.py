"""Hand decomposition: minimum number of plays to shed a hand.

Big 2 hands decompose into playable units (singles, pairs, 5-card hands,
optionally triples).  The minimum number of plays to empty the hand is a
small set-partition optimization, solved exactly here with memoized
branch and bound.  The key pruning trick: every partition contains
exactly one unit that covers the lowest remaining card, so branching
only over units containing that card enumerates each partition once.

This is research-ladder step 2 ("decomposition-optimal" scripted
baseline).  ``min_plays`` doubles as a strong input feature for learned
agents, and as an endgame oracle once hands get short.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

from big2.cards import Card
from big2.combos import Combo, generate_moves
from big2.game import Big2Game
from big2.rules import DEFAULT_RULES, RuleConfig
from big2.strategies import Strategy


class _Budget:
    def __init__(self, n: int):
        self.left = n

    def spend(self) -> bool:
        self.left -= 1
        return self.left >= 0


def min_plays(
    hand: List[Card],
    rules: RuleConfig = DEFAULT_RULES,
    budget: int = 50_000,
    _memo: Optional[Dict[FrozenSet[Card], Tuple[int, Tuple[Combo, ...]]]] = None,
) -> Tuple[int, List[Combo]]:
    """Exact minimum plays to shed ``hand`` and one optimal partition.

    ``budget`` caps search-node expansions; if exhausted the result may
    be an upper bound rather than the true optimum (still a valid
    partition).  In practice random 13-card hands solve well within the
    default budget.
    """
    memo: Dict = {} if _memo is None else _memo
    b = _Budget(budget)

    def solve(remaining: FrozenSet[Card]) -> Tuple[int, Tuple[Combo, ...]]:
        if not remaining:
            return 0, ()
        cached = memo.get(remaining)
        if cached is not None:
            return cached
        low = min(remaining)
        if not b.spend():
            # Budget exhausted: peel the lowest card as a single.
            single = generate_moves([low], rules=rules)[0]
            k, part = solve(remaining - {low})
            result = (k + 1, (single,) + part)
            memo[remaining] = result
            return result
        best_k, best_part = len(remaining) + 1, ()
        moves = [
            m
            for m in generate_moves(sorted(remaining), rules=rules)
            if low in m.cards
        ]
        # Bigger units first: reaches good bounds sooner.
        moves.sort(key=lambda m: -len(m))
        for m in moves:
            k, part = solve(remaining - set(m.cards))
            if k + 1 < best_k:
                best_k, best_part = k + 1, (m,) + part
                if best_k == (len(remaining) + 4) // 5:  # perfect: all 5-card
                    break
        memo[remaining] = (best_k, best_part)
        return best_k, best_part

    k, partition = solve(frozenset(hand))
    return k, list(partition)


class DecompositionStrategy(Strategy):
    """Plays to keep its minimum-plays count falling.

    - Leading: plays the weakest unit of an optimal partition (by the
      unit's strongest card, so control cards stay back).
    - Following: picks the weakest legal move that yields the smallest
      ``min_plays`` for the remainder; passes when every legal move
      would inflate the decomposition (unless the endgame is close).
    """

    name = "decomp"

    def __init__(self, endgame_size: int = 6, eval_cap: int = 14,
                 budget: int = 8_000):
        self.endgame_size = endgame_size
        self.eval_cap = eval_cap  # max candidate moves scored per decision
        self.budget = budget

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        moves = game.legal_moves(player)
        if not moves:
            return None
        hand = game.hands[player]
        rules = game.rules
        # One memo across the whole decision: candidate remainders share
        # most of their sub-hands.
        memo: Dict = {}
        current_k, partition = min_plays(hand, rules, self.budget, _memo=memo)

        if game.leading:
            unit_sets = {frozenset(u.cards) for u in partition}
            unit_moves = [m for m in moves if frozenset(m.cards) in unit_sets]
            pool = unit_moves or moves
            return min(pool, key=lambda m: (max(m.cards), len(m)))

        candidates = moves
        if len(candidates) > self.eval_cap:
            # Weakest moves are the likely picks; keep a couple of strong
            # ones so denial plays stay available.
            candidates = moves[: self.eval_cap - 2] + moves[-2:]
        scored = []
        for m in candidates:
            rest = list(hand)
            for c in m.cards:
                rest.remove(c)
            k, _ = min_plays(rest, rules, self.budget, _memo=memo)
            scored.append((k, m.sort_key, m))
        scored.sort(key=lambda t: (t[0], t[1]))
        best_k, _, best_move = scored[0]
        # Playing costs 1 + best_k total plays vs current_k by holding.
        # best_k == current_k (one extra play) is the normal price of
        # staying in the trick; only genuine structure damage passes.
        if best_k > current_k and len(hand) > self.endgame_size:
            return None
        return best_move
