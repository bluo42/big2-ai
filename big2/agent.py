"""v2.1: one agent, searching everywhere, deciding at the margin.

An earlier draft of this switched agents by stage — solver late, tree in
the middle, network early.  That was wrong.  The stages of a Big 2 hand
are not separated by a line you can draw at sixteen cards, and swapping
the decision-maker at an arbitrary threshold makes the agent's play
discontinuous for no reason the game supports.

What is genuinely different about the endgame is that it becomes
*deterministic*: few enough cards remain that a line can be verified
rather than estimated.  That earns an exact override.  Everything before
it — early and middle alike — is the same kind of problem, and gets the
same treatment: information-set MCTS with the policy as its prior.

That structure needs no stage logic, because PUCT already contains the
handover.  With the search unable to resolve anything (early, where the
determinizations disagree about everything), the prior term dominates
and the search returns the network's own move.  As the hand narrows,
measured value accumulates and can outvote the prior.  The transition is
continuous and driven by evidence rather than by a constant.

Where the two do disagree, the decision is made **at the margin**: the
search's pick is taken only when its measured advantage over the
policy's pick clears a threshold, so a coin-flip difference defers to
the network and a real one does not.

Every decision is capped at ``time_budget`` seconds of wall clock (one
by default) and stops as soon as the leader is out of reach, so the
common case costs milliseconds.  The cap matters in both directions: it
keeps a human game responsive, and it keeps self-play from becoming
arithmetic no training run can afford.

``explain`` returns that reasoning alongside the move — priors, visit
counts, values, how many worlds were sampled, how long it took, which
side won, and by how much — which is what the in-game view renders.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from big2.combos import Combo
from big2.endgame import MoveKey, move_key, remaining_cards
from big2.game import Big2Game
from big2.ismcts_pv import TIME_BUDGET, PolicyValueISMCTS, SearchResult
from big2.strategies import Strategy

# Cards left at which lines can be verified instead of estimated.
SOLVE_CARDS = 14
# Points per game the search must beat the policy by to override it.
OVERRIDE_MARGIN = 0.35


@dataclass
class Decision:
    """A move, and the reasoning that produced it."""

    move: MoveKey
    source: str                       # policy | search | solver | forced
    policy_move: MoveKey = None
    prior: Dict[MoveKey, float] = field(default_factory=dict)
    visits: Dict[MoveKey, int] = field(default_factory=dict)
    values: Dict[MoveKey, float] = field(default_factory=dict)
    margin: float = 0.0
    cards_left: int = 0
    simulations: int = 0
    worlds: int = 0
    elapsed: float = 0.0
    exact: bool = False

    @property
    def agreed_with_policy(self) -> bool:
        """Did the move end up being the network's own pick?

        Meaningful when the search ran (``source`` of policy or search);
        a solved or forced decision never records a policy move to
        compare against.  ``None`` is a legitimate move key -- it is the
        pass -- so this is plain equality, not a null check.
        """
        return self.move == self.policy_move

    def as_dict(self) -> Dict:
        fmt = lambda k: "pass" if k is None else list(k)
        keys = list(self.prior) or list(self.values)
        return {
            "move": fmt(self.move),
            "policy_move": fmt(self.policy_move),
            "source": self.source,
            "margin": round(self.margin, 3),
            "cards_left": self.cards_left,
            "simulations": self.simulations,
            "worlds": self.worlds,
            "elapsed_ms": round(1000.0 * self.elapsed, 1),
            "exact": self.exact,
            "candidates": [
                {
                    "move": fmt(k),
                    "prior": round(self.prior.get(k, 0.0), 4),
                    "visits": self.visits.get(k, 0),
                    "value": (None if k not in self.values
                              else round(self.values[k], 4)),
                }
                for k in keys
            ],
        }


class IntegratedAgent(Strategy):
    """Policy-guided IS-MCTS everywhere, exact solving once it is possible."""

    def __init__(
        self,
        policy: Strategy,
        simulations: int = 64,
        c_puct: float = 1.5,
        depth: int = 8,
        solve_cards: int = SOLVE_CARDS,
        override_margin: float = OVERRIDE_MARGIN,
        use_solver: bool = True,
        use_search: bool = True,
        seed: int = 0,
        time_budget: float = TIME_BUDGET,
        breadth: Optional[int] = None,
        top_p: Optional[float] = None,
        name: Optional[str] = None,
    ):
        self.policy = policy
        self.time_budget = float(time_budget)
        search_kw = {}
        if breadth is not None:
            search_kw["breadth"] = breadth
        if top_p is not None:
            search_kw["top_p"] = top_p
        self.searcher = PolicyValueISMCTS(
            policy, simulations=simulations, c_puct=c_puct, depth=depth,
            seed=seed, time_budget=time_budget, **search_kw,
        )
        self.solve_cards = solve_cards
        self.override_margin = override_margin
        self.use_solver = use_solver
        self.use_search = use_search
        self.rng = random.Random(seed)
        self.name = name or f"v2.1({getattr(policy, 'name', 'policy')})"

    # ------------------------------------------------------------------

    def _solver_decision(
        self, game: Big2Game, player: int, options: Sequence[Optional[Combo]]
    ) -> Optional[Decision]:
        """Exact answer when the position is small enough to verify."""
        from big2.endgame import pimc_move_values
        from big2.inference import MirrorState
        from big2.planning import PlanContext

        started = time.monotonic()
        left = remaining_cards(game)
        ctx = PlanContext(game, player)
        for m in options:
            if m is not None and len(m) == len(game.hands[player]) \
                    and ctx.is_boss(m) is True:
                return Decision(move_key(m), "solver", cards_left=left,
                                exact=True,
                                elapsed=time.monotonic() - started)
        if left > self.solve_cards:
            return None
        inf = MirrorState(game, player, rng=self.rng)
        worlds = inf.worlds_for_search(k=24, top=6)
        if not worlds:
            return None
        # Full node budget at every time budget: the measured strength
        # of this agent lives in the solver (138 solver decisions vs 46
        # search overrides drove +0.78/game vs the field), and scaling
        # nodes down with the clock was what erased the gain at cheap
        # budgets.  The wall clock is enforced by the *deadline* instead:
        # the sweep stops mid-worlds when time is up, keeping whatever
        # it has solved.  The solver gets 65% of the move budget; the
        # search inherits the remainder.
        values, agreement = pimc_move_values(
            game, player, worlds, budget=8000, with_agreement=True,
            deadline=started + 0.65 * self.time_budget,
        )
        if not values or agreement < 0.6:
            return None      # the deals disagree: not actually determined
        best = max(values, key=values.get)
        return Decision(best, "solver", values=values, cards_left=left,
                        worlds=len(worlds), exact=True,
                        elapsed=time.monotonic() - started)

    def explain(self, game: Big2Game, player: int) -> Decision:
        t0 = time.monotonic()
        options: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        left = remaining_cards(game)
        if len(options) == 1:
            return Decision(move_key(options[0]), "forced", cards_left=left)

        if self.use_solver:
            solved = self._solver_decision(game, player, options)
            if solved is not None:
                return solved

        if not self.use_search:
            return Decision(move_key(self.policy.select(game, player)),
                            "policy", cards_left=left)

        # Whatever the solver spent comes out of the search's clock, so
        # the decision as a whole honors one budget, not two.
        self.searcher.time_budget = max(
            0.05, self.time_budget - (time.monotonic() - t0)
        )
        res: SearchResult = self.searcher.search(game, player)
        dec = Decision(
            move=res.policy_move,
            source="policy",
            policy_move=res.policy_move,
            prior=res.prior,
            visits=res.visits,
            values=res.values,
            margin=res.margin,
            cards_left=left,
            simulations=res.simulations,
            worlds=res.worlds,
            elapsed=res.elapsed,
            exact=res.exact,
        )
        # Decide at the edge between the two candidates: the search wins
        # the move only when it can show the difference is real.
        if not res.agreed and res.margin >= self.override_margin:
            dec.move, dec.source = res.move, "search"
        return dec

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        dec = self.explain(game, player)
        options: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        for m in options:
            if move_key(m) == dec.move:
                return m
        return options[0] if options else None
