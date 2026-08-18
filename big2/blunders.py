"""Objective blunders: mistakes that need no opinion to call.

Most Big 2 decisions are judgement calls, and arguing about them needs
a model.  A few are not.  If a move empties your hand and nothing left
in the deck can answer it, playing anything else is simply wrong.  If
the player on your left has one card and you hand them a cheap single
they can beat while you were holding an unanswerable one, that is wrong
too — not "suboptimal by 0.3 points", wrong.

Those are the errors worth hunting, because they can be *counted*
rather than debated.  Each detector here is exact: it uses the same
boss-move test as the endgame solver (can any arrangement of the unseen
cards beat this?) and reports a rate per hundred decisions, which makes
models directly comparable and makes regressions obvious.

    MISSED_WIN      held a move that empties the hand and cannot be
                    answered — a guaranteed win — and played otherwise
    GIFT            fed a beatable move to a player one card from out
                    while holding an unanswerable alternative
    WASTED_BOSS     spent a beatable move while holding a boss unit and
                    somebody was two cards from going out

A model that scores zero on all three is not necessarily strong.  A
model that scores badly on them is definitely leaking games.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from big2.cards import hand_to_str
from big2.combos import Combo
from big2.game import Big2Game, ScoringConfig
from big2.planning import PlanContext
from big2.rules import DEFAULT_RULES
from big2.strategies import Strategy
from big2.combos import Combo  # noqa: F401  (re-exported for typing)

MISSED_WIN = "missed_guaranteed_win"
GIFT = "gift_to_near_winner"
WASTED_BOSS = "wasted_boss"
KINDS = (MISSED_WIN, GIFT, WASTED_BOSS)


@dataclass
class Blunder:
    kind: str
    player: int
    ply: int
    hand: str
    played: str
    better: str
    note: str

    def describe(self) -> str:
        return (f"{self.kind}: seat {self.player} ply {self.ply} "
                f"[{self.hand}] played {self.played}, had {self.better} "
                f"({self.note})")


def _fmt(move: Optional[Combo]) -> str:
    return "pass" if move is None else hand_to_str(list(move.cards))


def check_decision(
    game: Big2Game, player: int, played: Optional[Combo], ply: int = 0
) -> List[Blunder]:
    """Every objective error visible in this one decision."""
    out: List[Blunder] = []
    options: List[Optional[Combo]] = list(game.legal_moves(player))
    if not options:
        return out
    ctx = PlanContext(game, player)
    hand_size = len(game.hands[player])
    played_key = None if played is None else tuple(played.cards)

    # 1. A move that empties the hand and cannot be answered wins outright.
    winners = [
        m for m in options
        if len(m) == hand_size and ctx.is_boss(m) is True
    ]
    if winners and played_key != tuple(winners[0].cards):
        out.append(Blunder(
            MISSED_WIN, player, ply, hand_to_str(game.hands[player]),
            _fmt(played), _fmt(winners[0]),
            "unanswerable and empties the hand",
        ))
        return out  # nothing else about this decision matters

    if played is None:
        return out
    played_boss = ctx.is_boss(played)
    others = [p for p in range(game.num_players) if p != player]
    nearest = min(len(game.hands[p]) for p in others) if others else 13
    next_out = len(game.hands[(player + 1) % game.num_players])
    boss_alts = [
        m for m in options
        if ctx.is_boss(m) is True and tuple(m.cards) != played_key
    ]

    # 2. Handing a beatable move to somebody one card from going out,
    #    while holding one they could not have answered.
    if played_boss is False and next_out <= 1 and boss_alts:
        out.append(Blunder(
            GIFT, player, ply, hand_to_str(game.hands[player]),
            _fmt(played), _fmt(boss_alts[0]),
            f"next player holds {next_out}",
        ))
    # 3. Spending a beatable move with control in hand and danger on the table.
    elif played_boss is False and nearest <= 2 and boss_alts:
        out.append(Blunder(
            WASTED_BOSS, player, ply, hand_to_str(game.hands[player]),
            _fmt(played), _fmt(boss_alts[0]),
            f"shortest opponent holds {nearest}",
        ))
    return out


def scan_selfplay(
    policy: Strategy,
    opponents: Sequence[Strategy],
    n_games: int = 40,
    seed: int = 0,
    seat: int = 0,
) -> Dict[str, float]:
    """Blunder rates per 100 decisions for ``policy`` in fresh games."""
    rng = random.Random(seed)
    counts = {k: 0 for k in KINDS}
    decisions = 0
    found: List[Blunder] = []
    for g in range(n_games):
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4, rng=random.Random(rng.randrange(2**31)))
        seats = {p: (policy if p == seat else opponents[p % len(opponents)])
                 for p in range(4)}
        ply = 0
        while not game.game_over:
            p = game.turn
            move = seats[p].select(game, p)
            if p == seat:
                decisions += 1
                for b in check_decision(game, p, move, ply):
                    counts[b.kind] += 1
                    found.append(b)
            game.step(move)
            ply += 1
    rates = {k: (100.0 * v / decisions if decisions else 0.0)
             for k, v in counts.items()}
    rates["decisions"] = decisions
    rates["total_per_100"] = sum(
        100.0 * v / decisions if decisions else 0.0 for v in counts.values()
    )
    return rates, found


def scan_replays(
    rows: Sequence[Dict], seat: Optional[int] = None
) -> Dict[str, float]:
    """The same rates measured on recorded games (humans or bots)."""
    from big2.offline import _replay_body, iter_decisions
    from big2.combos import classify

    counts = {k: 0 for k in KINDS}
    decisions = 0
    found: List[Blunder] = []
    for row in rows:
        body = _replay_body(row)
        if body is None:
            continue
        for ply, (game, p, cards) in enumerate(iter_decisions(body)):
            if seat is not None and p != seat:
                continue
            move = (None if not cards
                    else classify([int(c) for c in cards], game.rules))
            decisions += 1
            for b in check_decision(game, p, move, ply):
                counts[b.kind] += 1
                found.append(b)
    rates = {k: (100.0 * v / decisions if decisions else 0.0)
             for k, v in counts.items()}
    rates["decisions"] = decisions
    rates["total_per_100"] = sum(
        100.0 * v / decisions if decisions else 0.0 for v in counts.values()
    )
    return rates, found


class GuardedPolicy(Strategy):
    """Wrap a policy and refuse the three objective blunders.

    Search would catch these too, but only where the tree is affordable
    — and the worst of them happen while somebody is one card from out
    with plenty of cards still on the table, which is exactly where the
    solver is too expensive to run.  These checks are arithmetic on sets
    and cost nothing, so they hold everywhere.

    The guard only ever *removes* objectively losing choices: when it
    fires it substitutes a move the underlying policy already had
    available, and when nothing is flagged the policy's own decision
    stands untouched.
    """

    def __init__(self, base: Strategy, name: Optional[str] = None):
        self.base = base
        self.name = name or f"guarded({getattr(base, 'name', 'policy')})"

    def option_scores(self, game: Big2Game, player: int):
        return self.base.option_scores(game, player)

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        move = self.base.select(game, player)
        options = list(game.legal_moves(player))
        if not options:
            return move
        ctx = PlanContext(game, player)
        hand_size = len(game.hands[player])

        # A move that empties the hand and cannot be answered ends it.
        for m in options:
            if len(m) == hand_size and ctx.is_boss(m) is True:
                return m

        if move is None:
            return move
        if ctx.is_boss(move) is not False:
            return move          # unanswerable or unknown: leave it alone

        others = [p for p in range(game.num_players) if p != player]
        nearest = min(len(game.hands[p]) for p in others) if others else 13
        if nearest > 2:
            return move          # nobody is about to go out

        # Prefer an unanswerable move of the same size class, so the
        # substitution keeps the policy's shape and only fixes the leak.
        boss_alts = [m for m in options if ctx.is_boss(m) is True]
        if not boss_alts:
            return move
        same_size = [m for m in boss_alts if len(m) == len(move)]
        pool = same_size or boss_alts
        return min(pool, key=lambda m: max(m.cards))
