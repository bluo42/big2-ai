"""Play the same position with the cards face up, and see what changes.

Every belief feature added so far has been justified by intuition.  This
measures the thing they are all competing for: **how much is the hidden
information actually worth?**

The method is to take a real position and answer it three ways —

* **blind**: what the model plays now, knowing only the public history;
* **partial**: the best move when a random fraction of the opponents'
  cards is revealed;
* **face up**: the best move when the whole deal is known, which the
  endgame solver can answer exactly.

Two numbers come out of that, and they mean different things:

* **divergence** — how often more information changes the move.  A
  position where the face-up answer is the move the model already plays
  is one where beliefs cannot help, no matter how good they get.
* **value of information** — the EV gap between the face-up move and the
  model's move.  Summed over a game, this is the ceiling on what *any*
  improvement to the belief model could ever be worth, which is the
  number that says whether more work on inference is worth doing at all.

The partial level is the interesting one for training: it is the regime
the model actually lives in, and the states where a *little* extra
knowledge flips the decision are exactly the states where sharpening the
posterior pays.  ``rank_information_states`` returns those, so belief
training can be pointed at them instead of spread uniformly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from big2.combos import Combo
from big2.endgame import MoveKey, move_key
from big2.game import Big2Game
from big2.strategies import Strategy


@dataclass
class InfoPoint:
    """One position, answered at several levels of knowledge."""

    ply: int
    player: int
    cards_left: Tuple[int, ...]
    model_move: MoveKey
    faceup_move: MoveKey
    model_ev: float
    faceup_ev: float
    partial_move: Optional[MoveKey] = None
    partial_ev: Optional[float] = None

    @property
    def changed(self) -> bool:
        return self.model_move != self.faceup_move

    @property
    def value_of_information(self) -> float:
        return max(0.0, self.faceup_ev - self.model_ev)


def reveal_worlds(
    game: Big2Game,
    player: int,
    frac: float,
    k: int = 16,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Dict[int, List[int]], float]]:
    """Sampled deals consistent with revealing ``frac`` of the real cards.

    ``frac=1`` collapses to the true deal (one world, face up); ``frac=0``
    is the ordinary posterior.  In between, the revealed cards are fixed
    and the rest are re-dealt, which is what "the model knows a little
    more" actually means.
    """
    from big2.inference import InferenceState

    rng = rng or random.Random(0)
    others = [p for p in range(game.num_players) if p != player]
    if frac >= 1.0:
        return [({p: list(game.hands[p]) for p in others}, 1.0)]

    revealed: Dict[int, List[int]] = {}
    for p in others:
        hand = list(game.hands[p])
        n = int(round(frac * len(hand)))
        revealed[p] = rng.sample(hand, n) if n else []
    fixed = {c for cards in revealed.values() for c in cards}

    inf = InferenceState(game, player, rng=rng)
    out = []
    for world, w in inf.sample_worlds(k * 4):
        if w <= 0.0:
            continue
        # keep only deals that agree with what was revealed
        pool = [c for cards in world.values() for c in cards
                if c not in fixed]
        rng.shuffle(pool)
        rebuilt, i, ok = {}, 0, True
        for p in others:
            need = len(game.hands[p]) - len(revealed[p])
            if need < 0 or i + need > len(pool):
                ok = False
                break
            rebuilt[p] = sorted(revealed[p] + pool[i:i + need])
            i += need
        if ok:
            out.append((rebuilt, w))
        if len(out) >= k:
            break
    return out


def best_under_information(
    game: Big2Game,
    player: int,
    worlds: Sequence[Tuple[Dict[int, List[int]], float]],
    budget: int = 8000,
) -> Tuple[Optional[MoveKey], Dict[MoveKey, float]]:
    """The move that wins on average across these deals, and the values."""
    from big2.endgame import pimc_move_values

    evs = pimc_move_values(game, player, worlds, budget=budget)
    if not evs:
        return None, {}
    return max(evs, key=evs.get), evs


def information_point(
    game: Big2Game,
    player: int,
    model: Strategy,
    opponents: Sequence[Strategy],
    ply: int = 0,
    partial: float = 0.5,
    rollouts: int = 12,
    k_worlds: int = 12,
    rng: Optional[random.Random] = None,
) -> Optional[InfoPoint]:
    """Answer one position blind, partially informed, and face up."""
    from big2.critique import move_ev

    rng = rng or random.Random(0)
    options: List[Optional[Combo]] = list(game.legal_moves(player))
    if game.can_pass():
        options.append(None)
    if len(options) < 2:
        return None

    model_key = move_key(model.select(game, player))
    face = reveal_worlds(game, player, 1.0, rng=rng)
    face_key, face_evs = best_under_information(game, player, face)
    if face_key is None:
        return None

    def find(key):
        return next((m for m in options if move_key(m) == key), None)

    model_ev, _ = move_ev(game, player, find(model_key), opponents, rollouts)
    face_ev, _ = move_ev(game, player, find(face_key), opponents, rollouts)

    part = reveal_worlds(game, player, partial, k=k_worlds, rng=rng)
    part_key, _ = best_under_information(game, player, part) if part else (None, {})

    return InfoPoint(
        ply=ply,
        player=player,
        cards_left=tuple(len(h) for h in game.hands),
        model_move=model_key,
        faceup_move=face_key,
        model_ev=model_ev,
        faceup_ev=face_ev,
        partial_move=part_key,
        partial_ev=None,
    )


def rank_information_states(
    rows: Sequence[Dict],
    model: Strategy,
    opponents: Optional[Sequence[Strategy]] = None,
    seat: Optional[int] = None,
    stride: int = 4,
    max_games: int = 20,
    partial: float = 0.5,
    rollouts: int = 12,
    seed: int = 0,
) -> List[InfoPoint]:
    """Walk recorded games and score every Nth decision for information."""
    from big2.offline import _replay_body, iter_decisions
    from big2.strategies import SmartHeuristic

    opponents = list(opponents or [model, SmartHeuristic()])
    rng = random.Random(seed)
    points: List[InfoPoint] = []
    for gi, row in enumerate(rows[:max_games]):
        body = _replay_body(row)
        if body is None:
            continue
        for ply, (game, p, _cards) in enumerate(iter_decisions(body)):
            if ply % stride or (seat is not None and p != seat):
                continue
            pt = information_point(
                game, p, model, opponents, ply=ply, partial=partial,
                rollouts=rollouts, rng=rng,
            )
            if pt is not None:
                points.append(pt)
    points.sort(key=lambda q: -q.value_of_information)
    return points


def summarize(points: Sequence[InfoPoint]) -> Dict[str, float]:
    if not points:
        return {"points": 0}
    changed = [p for p in points if p.changed]
    partial_changed = [
        p for p in points
        if p.partial_move is not None and p.partial_move != p.model_move
    ]
    return {
        "points": len(points),
        "faceup_changes_move": len(changed) / len(points),
        "partial_changes_move": len(partial_changed) / len(points),
        "mean_value_of_information": sum(
            p.value_of_information for p in points
        ) / len(points),
        "max_value_of_information": max(
            p.value_of_information for p in points
        ),
    }
