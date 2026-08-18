"""Search-augmented play: a learned prior, decided by the endgame tree.

Big 2 changes character as it empties.  Early, the hidden state is huge
and a policy network's pattern matching is the only affordable answer.
Late, the state collapses — a handful of cards, most of them accounted
for — and *exact* reasoning becomes both possible and much better than a
guess.  So the two are blended on a ramp:

    lambda = 0                       while many cards remain (pure net)
    lambda: 0 -> 1  as cards vanish  (search takes over)

The blend is log-linear (a standard product-of-experts pooling):

    score(m) = (1 - lambda) * log p_net(m) + lambda * log p_search(m)

with ``p_search`` a softmax over the PIMC expected values, so a move
whose exact EV is several points better dominates, while near-ties fall
back to the network's taste.  Two guards matter:

* PIMC assumes hidden cards are revealed after the deal (**strategy
  fusion**), which overvalues plans that depend on opponents blundering.
  Restricting it to small states, where most of the deal is known
  anyway, is what keeps that bias small.
* A **provable win** short-circuits everything: if a move empties the
  hand and cannot be answered, no search or network opinion is needed.

The same blended distribution doubles as an expert-iteration target:
``search_distribution`` gives the policy something better than its own
output to imitate at exactly the states where it is weakest.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.combos import Combo
from big2.endgame import (
    MoveKey,
    move_key,
    pimc_move_values,
    remaining_cards,
)
from big2.game import Big2Game
from big2.inference import InferenceState
from big2.strategies import Strategy

# Cards left on the table (all hands) at which search starts / takes over.
# Chosen from measured solver cost: a 12-card state solves in milliseconds,
# a 16-card one in ~0.15s at the default budget, and beyond ~20 the tree
# stops fitting in any affordable budget (the static fallback would be
# doing the talking, so the network is the better answer).
SEARCH_START = 16
SEARCH_FULL = 10
SEARCH_BUDGET = 8_000
# Above this many cards the tree cannot finish inside any affordable
# budget, so the attempt would be discarded anyway: skip it and save the
# work rather than paying for a result we will not trust.
SEARCH_HARD_CAP = 24


def world_count(game: Big2Game, cap: int = 8) -> int:
    """How many determinizations we can afford here: exact solving gets
    rapidly dearer as cards remain, so the sample shrinks to match."""
    n = remaining_cards(game)
    if n <= 11:
        return cap
    if n <= 14:
        return max(2, cap // 2)
    return max(2, cap // 4)


def search_weight(game: Big2Game, start: int = SEARCH_START,
                  full: int = SEARCH_FULL) -> float:
    """How much to trust the tree.

    Two things make search worthwhile, and either is enough.  **Size**:
    few cards left, so the tree is small and nearly complete
    information.  **Urgency**: somebody is one or two cards from going
    out, so the hand is about to be decided even if plenty of cards are
    still out — the case a pure card-count gate misses entirely, since a
    Big 2 hand ends with ~14 cards still held on average.

    Over-triggering is safe: a search that cannot finish inside its node
    budget is discarded rather than trusted (see endgame.pimc_move_values),
    so the network simply keeps the decision.
    """
    n = remaining_cards(game)
    if n > SEARCH_HARD_CAP:
        return 0.0
    shortest = min(len(h) for h in game.hands)
    if n <= full or shortest <= 2:
        return 1.0
    if n <= start:
        return (start - n) / float(start - full)
    return 0.5 if shortest <= 4 else 0.0


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / max(temp, 1e-6)
    e = np.exp(z)
    return e / e.sum()


def search_distribution(
    game: Big2Game,
    player: int,
    options: Sequence[Optional[Combo]],
    prior: np.ndarray,
    worlds: int = 24,
    top_worlds: int = 8,
    budget: int = SEARCH_BUDGET,
    temp: float = 1.5,
    rng: Optional[random.Random] = None,
    lam: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[MoveKey, float]]:
    """Blend a policy prior with PIMC values; returns (probs, EVs)."""
    p_net = _softmax(np.asarray(prior, dtype=np.float64))
    lam = search_weight(game) if lam is None else lam
    if lam <= 0.0:
        return p_net, {}
    inf = InferenceState(game, player, rng=rng or random.Random(0))
    sampled = inf.worlds_for_search(k=worlds, top=top_worlds)
    if not sampled:
        return p_net, {}
    evs, agreement = pimc_move_values(
        game, player, sampled, budget=budget, with_agreement=True
    )
    if not evs:
        return p_net, {}
    # How much the tree gets to say is how settled the position is, not
    # just how few cards are left: a hand where every plausible deal
    # points at the same move is decided, and one where the deals
    # disagree is still the network's call.
    lam = lam * agreement
    vals = np.array(
        [evs.get(move_key(m), float("-inf")) for m in options], dtype=np.float64
    )
    if not np.isfinite(vals).any():
        return p_net, {}
    vals[~np.isfinite(vals)] = vals[np.isfinite(vals)].min() - 5.0
    p_search = _softmax(vals, temp=temp)
    log_mix = (1.0 - lam) * np.log(p_net + 1e-12) + lam * np.log(
        p_search + 1e-12
    )
    mix = _softmax(log_mix)
    return mix, evs


class SearchAugmentedPolicy(Strategy):
    """Wraps any policy exposing ``option_scores`` with endgame search."""

    name = "search"

    def __init__(
        self,
        base: Strategy,
        worlds: int = 24,
        top_worlds: int = 8,
        budget: int = SEARCH_BUDGET,
        start: int = SEARCH_START,
        full: int = SEARCH_FULL,
        seed: int = 0,
    ):
        self.base = base
        self.worlds = worlds
        self.top_worlds = top_worlds
        self.budget = budget
        self.start = start
        self.full = full
        self.rng = random.Random(seed)
        self.name = f"search({getattr(base, 'name', 'policy')})"

    def _prior(self, game: Big2Game, player: int):
        if hasattr(self.base, "option_scores"):
            return self.base.option_scores(game, player)
        options = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        pick = self.base.select(game, player)
        key = move_key(pick)
        # A policy that only names its move gets a *preference*, not a
        # veto: too sharp a spike here (softmax ~98%) would override the
        # search even where the tree has an exact answer.
        prior = np.array(
            [1.2 if move_key(m) == key else 0.0 for m in options],
            dtype=np.float64,
        )
        return options, prior

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        options: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        if len(options) == 1:
            return options[0]

        # A move that empties the hand and cannot be answered wins on the
        # spot — no search, no network.
        from big2.planning import PlanContext

        ctx = PlanContext(game, player)
        for m in options:
            if m is not None and len(m) == len(game.hands[player]):
                if ctx.is_boss(m):
                    return m

        lam = search_weight(game, self.start, self.full)
        options, prior = self._prior(game, player)
        if lam <= 0.0:
            return options[int(np.argmax(prior))]
        probs, _ = search_distribution(
            game, player, options, prior, worlds=self.worlds,
            top_worlds=world_count(game, self.top_worlds),
            budget=self.budget, rng=self.rng, lam=lam,
        )
        return options[int(np.argmax(probs))]
