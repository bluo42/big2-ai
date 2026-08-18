"""Information-set MCTS guided by the policy, evaluated by the value.

The plain ISMCTS in this project is a flat bandit with random rollouts:
no idea which moves are worth trying, and a noisy verdict on the ones it
does try.  That is fine as a reference opponent and useless as a partner
to a trained network.

This is the version that composes with one.  Selection is PUCT,

    Q(a) + c * P(a) * sqrt(N) / (1 + N(a))

with ``P`` the policy's own distribution over the legal moves.  The
consequence is the property that makes this the right structure for a
card game: **with few simulations the prior term dominates and the
search returns what the policy already believed**, while every extra
simulation lets measured value pull it away.  There is no stage switch
and no threshold deciding "now use search" — early in the hand, where
determinizations disagree about everything, the search simply cannot
outvote the prior, and late in the hand it easily can.  The handover is
continuous and comes from the evidence.

What the policy contributes, concretely:

* **Breadth** — the prior ranks the legal set and the search keeps only
  the moves covering ``top_p`` of its probability mass (never more than
  ``breadth``).  A 30-move position is searched as the two or three
  moves actually in contention; moves the net calls obviously bad never
  cost a simulation.
* **Rollouts** — the plies after the candidate move are played by the
  policy rather than at random, so the measured value is "what happens
  if I play this and everyone answers competently", not "…and everyone
  answers like a coin".

What the belief model contributes:

* **Weighted determinizations** — each simulation draws one consistent
  deal of the unseen cards *in proportion to its posterior weight*, so
  common worlds are searched often and implausible ones rarely.  The
  statistics are then an expectation over what the opponents plausibly
  hold rather than over an arbitrary enumeration.
* **Sampled replies** — inside a world, the opponents' answers are
  *sampled* from the policy's distribution over the hand they were
  dealt, not taken greedily.  A determinization already commits to one
  guess about the cards; playing it out deterministically on top of that
  would compound the guess into a single fictitious line.

Two more pieces:

* **Solved leaves** — once a line collapses into solver range the leaf
  is evaluated exactly instead of estimated, so deep lines terminate in
  real numbers.
* **A solved root** — when the whole position is inside solver range
  there is nothing to sample about: every candidate is evaluated against
  every world exactly (weighted PIMC), and the bandit is skipped.  This
  is the one place the endgame genuinely differs from the rest of the
  hand, and it earns exact treatment rather than a threshold.

Every search is bounded by wall clock (``time_budget``, one second by
default) as well as by simulation count, and stops early once the leader
cannot be caught — so ordinary moves cost milliseconds and only genuinely
contested ones spend the budget.

``search`` returns the statistics as well as the move, so a caller can
see where the search and the policy disagreed and by how much.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.combos import Combo
from big2.endgame import MoveKey, move_key, remaining_cards
from big2.game import Big2Game
from big2.strategies import Strategy

SCORE_SCALE = 39.0
SOLVE_BELOW = 12          # leaves this small are solved, not estimated
MAX_ROLLOUT_STEPS = 400

# Wall-clock ceiling for one decision.  Deliberately modest: the search
# runs inside self-play as well as inside the web app, and a budget that
# makes a single game pleasant makes a training run impossible.
TIME_BUDGET = 1.0
# Never stop before this many simulations, budget or no budget: a search
# that ran twice is worse than no search at all.
MIN_SIMULATIONS = 8
# The prior's cut is by probability mass, not by count: candidates are
# kept in descending prior until TOP_P of the mass is covered, floored at
# 2 and hard-capped at BREADTH.  A confident net searches its top two
# moves; a torn one gets its whole dilemma searched.  Moves the policy
# calls obviously bad never cost a simulation — at ~20ms each, spending
# any of them refuting garbage starves the real comparison.  The safety
# valve for a policy blind spot is the solved root: inside solver range
# every legal move is evaluated exactly, with no pruning at all.
BREADTH = 4
TOP_P = 0.90
# Determinizations drawn per decision.  The loop resamples from these with
# replacement, so more of them buys resolution in the posterior, not more
# simulations — and drawing them is not free.
MAX_WORLDS = 48


@dataclass
class SearchResult:
    """What the search found, and what the policy thought."""

    move: MoveKey
    policy_move: MoveKey
    visits: Dict[MoveKey, int] = field(default_factory=dict)
    values: Dict[MoveKey, float] = field(default_factory=dict)
    prior: Dict[MoveKey, float] = field(default_factory=dict)
    simulations: int = 0
    cards_left: int = 0
    exact: bool = False
    elapsed: float = 0.0
    worlds: int = 0

    @property
    def agreed(self) -> bool:
        return self.move == self.policy_move

    @property
    def margin(self) -> float:
        """How much better the search rates its pick than the policy's.

        Expressed in points per game.  Near zero means the two agree in
        substance even if they name different cards; large means the search
        found something the policy did not.
        """
        if self.move not in self.values or self.policy_move not in self.values:
            return 0.0
        return (self.values[self.move]
                - self.values[self.policy_move]) * SCORE_SCALE


class PolicyValueISMCTS:
    """PUCT over determinized worlds, with a learned prior and value."""

    def __init__(
        self,
        policy: Strategy,
        simulations: int = 64,
        c_puct: float = 1.5,
        depth: int = 8,
        seed: int = 0,
        solve_below: int = SOLVE_BELOW,
        time_budget: float = TIME_BUDGET,
        breadth: int = BREADTH,
        top_p: float = TOP_P,
        rollout_temp: float = 0.8,
    ):
        self.policy = policy
        self.simulations = simulations
        self.c_puct = c_puct
        self.depth = depth
        self.rng = random.Random(seed)
        self.solve_below = solve_below
        self.time_budget = float(time_budget)
        self.breadth = int(breadth)
        self.top_p = float(top_p)
        self.rollout_temp = float(rollout_temp)

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def _prior(self, game: Big2Game, player: int):
        """(options, probabilities) from the policy over the legal set."""
        if hasattr(self.policy, "option_scores"):
            options, scores = self.policy.option_scores(game, player)
            z = np.asarray(scores, dtype=np.float64)
            z = z - z.max()
            p = np.exp(z)
            return options, p / p.sum()
        options: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        pick = move_key(self.policy.select(game, player))
        p = np.array([3.0 if move_key(m) == pick else 1.0 for m in options],
                     dtype=np.float64)
        return options, p / p.sum()

    def _sampled_move(self, game: Big2Game, seat: int) -> Optional[Combo]:
        """One draw from the policy's distribution for ``seat``.

        Used for the opponents inside a rollout.  The determinization has
        already guessed *which* cards they hold; answering that guess
        greedily would turn one guess into one fictitious line, and the
        average over worlds would inherit the pretence that opponents are
        predictable.  Sampling keeps the rollout an average over how the
        hand could be played, not just over how it could be dealt.
        """
        if not hasattr(self.policy, "option_scores"):
            return self.policy.select(game, seat)
        options, scores = self.policy.option_scores(game, seat)
        if not options:
            return None
        if len(options) == 1:
            return options[0]
        z = np.asarray(scores, dtype=np.float64) / max(1e-3, self.rollout_temp)
        z = np.exp(z - z.max())
        p = z / z.sum()
        return options[self.rng.choices(range(len(options)), weights=p)[0]]

    # ------------------------------------------------------------------
    # Leaf evaluation
    # ------------------------------------------------------------------

    def _value(self, game: Big2Game, player: int) -> Tuple[float, bool]:
        """(leaf value in [-1, 1], was_it_exact)."""
        if game.game_over:
            return float(game.scores[player]) / SCORE_SCALE, True
        if remaining_cards(game) <= self.solve_below:
            from big2.endgame import solve

            try:
                vec = solve(game)
                if vec:
                    return float(vec[player]) / SCORE_SCALE, True
            except Exception:
                pass
        net = getattr(self.policy, "net", None)
        if net is not None:
            try:
                import torch

                from big2.neural import encode_decision

                options, state, acts = encode_decision(
                    game, player,
                    include_profiles=getattr(self.policy, "uses_profiles", True),
                    include_danger=getattr(self.policy, "uses_danger", True),
                    include_plan=getattr(self.policy, "uses_plan", True),
                )
                if options:
                    with torch.no_grad():
                        _l, v, _b = net(
                            torch.from_numpy(state).unsqueeze(0),
                            torch.from_numpy(acts).unsqueeze(0),
                            torch.ones(1, len(options), dtype=torch.bool),
                        )
                    return float(v[0]), False
            except Exception:
                pass
        # Last resort: a card-count read of the race.
        mine = len(game.hands[player])
        others = [len(game.hands[p]) for p in range(game.num_players)
                  if p != player]
        return float(np.clip(
            ((sum(others) / max(1, len(others))) - mine) / 13.0, -1.0, 1.0
        )), False

    def _play_forward(self, world: Big2Game, player: int, plies: int) -> None:
        """Play on: we answer greedily, the opponents are sampled.

        The rollout does not stop at a raw ply count: it stops at the
        *player's own next decision* once the depth is spent.  The value
        head was trained exclusively at states where the encoded player
        is the one to move — pausing mid-trick would hand it a state it
        has never seen (an off-turn player has no legal moves, so the
        action set collapses to a bare pass).  Stopping on our turn keeps
        every leaf on-distribution.
        """
        steps = 0
        while not world.game_over and steps < MAX_ROLLOUT_STEPS:
            if remaining_cards(world) <= self.solve_below:
                return
            if steps >= plies and world.turn == player:
                return
            seat = world.turn
            move = (self.policy.select(world, seat) if seat == player
                    else self._sampled_move(world, seat))
            world.step(move)
            steps += 1

    # ------------------------------------------------------------------
    # Determinizations
    # ------------------------------------------------------------------

    def _worlds(self, game: Big2Game, player: int, k: int):
        """Belief-weighted deals of the unseen cards, with a sampler."""
        from big2.inference import InferenceState

        inf = InferenceState(game, player, rng=self.rng)
        worlds = inf.worlds_for_search(k=k)
        if not worlds:
            return [({}, 1.0)], [1.0]
        weights = [max(0.0, float(w)) for _, w in worlds]
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0] * len(worlds)
            total = float(len(worlds))
        return worlds, [w / total for w in weights]

    @staticmethod
    def _deal(game: Big2Game, player: int, world: Dict[int, List[int]]):
        from big2.endgame import search_clone

        sim = search_clone(game)
        for p, hand in world.items():
            if p != player:
                sim.hands[p] = sorted(hand)
        return sim

    # ------------------------------------------------------------------

    def _solved_root(
        self, game: Big2Game, player: int, options, prior, keys, policy_move,
        left: int, started: float,
    ) -> SearchResult:
        """The position is small enough to settle, so settle it.

        No bandit, no sampling of which move to try: every candidate is
        solved exactly against every plausible deal and averaged by the
        posterior.  This is the only part of the hand where the tree is
        genuinely deterministic, and it is worth spending the whole
        budget being right rather than being fast.
        """
        from big2.endgame import pimc_move_values

        worlds, _p = self._worlds(game, player, k=24)
        values, agree = pimc_move_values(
            game, player, worlds, budget=8000, with_agreement=True
        )
        if not values:
            return SearchResult(policy_move, policy_move, {}, {},
                                dict(zip(keys, prior)), 0, left,
                                elapsed=time.monotonic() - started,
                                worlds=len(worlds))
        n_worlds = sum(1 for _, w in worlds if w > 0.0)
        best = max(values, key=values.get)
        return SearchResult(
            move=best,
            policy_move=policy_move,
            visits={k: (n_worlds if k in values else 0) for k in keys},
            values=values,
            prior=dict(zip(keys, prior)),
            simulations=n_worlds * len(values),
            cards_left=left,
            exact=True,
            elapsed=time.monotonic() - started,
            worlds=n_worlds,
        )

    # ------------------------------------------------------------------

    def search(self, game: Big2Game, player: int) -> SearchResult:
        started = time.monotonic()
        deadline = started + self.time_budget

        options, prior = self._prior(game, player)
        keys = [move_key(m) for m in options]
        policy_move = keys[int(np.argmax(prior))]
        left = remaining_cards(game)
        if len(options) == 1:
            return SearchResult(keys[0], keys[0], {keys[0]: 0},
                                {}, dict(zip(keys, prior)), 0, left,
                                exact=True, elapsed=time.monotonic() - started)

        if left <= self.solve_below:
            return self._solved_root(game, player, options, prior, keys,
                                     policy_move, left, started)

        # The prior's job is to say which moves are not worth a
        # simulation.  Keep candidates in descending prior until top_p of
        # the mass is covered (>=2, <=breadth); everything outside the
        # cut keeps its entry in the report with zero visits, so the
        # caller still sees it was considered and dismissed.
        order = sorted(range(len(options)), key=lambda i: -prior[i])
        keep, mass = [], 0.0
        for i in order:
            if len(keep) >= 2 and (mass >= self.top_p
                                   or len(keep) >= max(2, self.breadth)):
                break
            keep.append(i)
            mass += prior[i]
        cand = sorted(keep)

        # Enough worlds that the posterior is represented, few enough that
        # sampling them is not itself the cost of the search.
        worlds, wp = self._worlds(
            game, player, k=min(MAX_WORLDS, max(8, self.simulations // 4))
        )

        n = [0] * len(options)
        w = [0.0] * len(options)
        exact_hits = [0] * len(options)
        done = 0
        for t in range(self.simulations):
            if done >= MIN_SIMULATIONS and time.monotonic() >= deadline:
                break
            # Sweep first: every candidate is measured once before PUCT
            # is allowed to concentrate.  Without it a confident prior can
            # spend the whole budget on its own opinion and never learn
            # that the alternative wins.
            if t < len(cand):
                idx = cand[t]
            else:
                total = sum(n)
                idx = max(
                    cand,
                    key=lambda i: (
                        (w[i] / n[i] if n[i] else 0.0)
                        + self.c_puct * prior[i] * math.sqrt(total + 1)
                        / (1 + n[i])
                    ),
                )
            world_hands = worlds[
                self.rng.choices(range(len(worlds)), weights=wp)[0]
            ][0]
            sim = self._deal(game, player, world_hands)
            try:
                sim.step(options[idx])
            except (ValueError, RuntimeError):
                n[idx] += 1
                w[idx] += -1.0
                done += 1
                continue
            if not sim.game_over:
                self._play_forward(sim, player, self.depth)
            value, was_exact = self._value(sim, player)
            n[idx] += 1
            w[idx] += value
            exact_hits[idx] += int(was_exact)
            done += 1
            # Settled: nothing left in the budget can overtake the leader.
            if done >= MIN_SIMULATIONS and len(cand) > 1:
                top = sorted((n[i] for i in cand), reverse=True)
                if top[0] - top[1] > self.simulations - done:
                    break

        values = {keys[i]: (w[i] / n[i] if n[i] else 0.0)
                  for i in range(len(options))}
        visits = {keys[i]: n[i] for i in range(len(options))}
        # Robust choice: most-visited, ties broken by value.
        best = max(range(len(options)),
                   key=lambda i: (n[i], w[i] / n[i] if n[i] else -9.9))
        return SearchResult(
            move=keys[best],
            policy_move=policy_move,
            visits=visits,
            values=values,
            prior=dict(zip(keys, prior)),
            simulations=done,
            cards_left=left,
            exact=bool(n[best] and exact_hits[best] == n[best]),
            elapsed=time.monotonic() - started,
            worlds=len(worlds),
        )
