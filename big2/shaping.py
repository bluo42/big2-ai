"""Trick-level potential: what a position is worth between tricks.

The terminal payout reaches an opening decision through forty cards of
other people's randomness — measured on khabib_v1, the value head's
correlation with the final score is 0.41 early against 0.84 late, so
early moves train on gradient that is mostly noise.  This module gives
the learner (and the search) a dense, cheap statement of position value
whose components are exactly the levers a strong human plays:

* **control** — leading a fresh trick is the high-value state; when
  someone else leads, it matters *who*: the player immediately before
  you is least bad (you respond first, cheapest), the player
  immediately after you is worst (you respond last).
* **liability** — cards remaining, priced by the actual scoring ladder
  (10+ cards pay double, 13 triple), not linearly.
* **tempo** — the exact minimum number of plays to shed the hand:
  a hand that exits in 4 plays is closer to winning than one that
  needs 8, whatever the card count says.
* **boss cards** — singles no unseen card beats, pairs no unseen pair
  beats: each is a trick you cannot be denied.
* **rank pressure** — mean rank of the hand against the mean rank of
  the unseen pool.

Used as a *potential*: the shaped per-step reward is
gamma*phi(s') - phi(s) with gamma = 1 and phi(terminal) = 0, which
telescopes to a per-episode constant — the optimal policy is unchanged
(Ng, Harada, Russell 1999), only the credit arrives within a trick of
the decision instead of at the shuffle.  The same phi is added at the
search's trick-boundary leaf so the tree optimizes what the policy
trains on.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from big2.cards import NUM_CARDS, NUM_RANKS, Card, rank
from big2.decomposition import min_plays
from big2.game import Big2Game

# Component weights, in points of game score.  The potential must stay
# small next to real payout swings (a loss costs 1-39 points): it is a
# hint about *when* credit lands, not a second objective.
W_CONTROL = 1.2
W_LIABILITY = 0.25
W_TEMPO = 0.35
W_BOSS_SINGLE = 0.5
W_BOSS_PAIR = 0.75
W_RANK = 1.0

# Winner-relative control weights, indexed by (leader - seat) mod n for
# n=4: leading yourself is the prize; the seat right before you leading
# is least bad (you respond first); the seat right after you is worst
# (you respond last).  3-player games reuse the 4-player shape minus
# the middle seat.
_CTRL_4 = {0: 1.0, 3: -0.15, 2: -0.35, 1: -0.5}
_CTRL_3 = {0: 1.0, 2: -0.2, 1: -0.5}

_MIN_PLAYS_MEMO: Dict[frozenset, int] = {}
_MEMO_CAP = 200_000


def _tempo(hand: List[Card], rules) -> int:
    key = frozenset(hand)
    hit = _MIN_PLAYS_MEMO.get(key)
    if hit is None:
        if len(_MIN_PLAYS_MEMO) >= _MEMO_CAP:
            _MIN_PLAYS_MEMO.clear()
        hit = min_plays(list(hand), rules, budget=4000)[0]
        _MIN_PLAYS_MEMO[key] = hit
    return hit


def potential(game: Big2Game, seat: int) -> float:
    """phi(state, seat) in points; 0 at terminal states."""
    hand = game.hands[seat]
    if game.game_over or not hand:
        return 0.0

    seen = set(game.played_cards)
    for c in hand:
        seen.add(c)
    unseen = [c for c in range(NUM_CARDS) if c not in seen]

    # --- control ------------------------------------------------------
    ctrl = 0.0
    if game.table_combo is None:            # fresh trick: someone leads
        n = game.num_players
        d = (game.turn - seat) % n
        table = _CTRL_4 if n == 4 else _CTRL_3 if n == 3 else {0: 1.0, 1: -0.5}
        ctrl = table.get(d, -0.35)

    # --- liability: the ladder you would actually pay ------------------
    liability = float(game.scoring.payment(hand))

    # --- tempo: exact plays-to-shed ------------------------------------
    tempo = float(_tempo(hand, game.rules))

    # --- boss singles and pairs vs the unseen ---------------------------
    top_unseen = max(unseen) if unseen else -1
    boss_singles = sum(1 for c in hand if c > top_unseen) if unseen else len(hand)
    unseen_rank_counts = [0] * NUM_RANKS
    for c in unseen:
        unseen_rank_counts[rank(c)] += 1
    best_unseen_pair_rank = max(
        (r for r in range(NUM_RANKS) if unseen_rank_counts[r] >= 2),
        default=-1,
    )
    hand_rank_counts = [0] * NUM_RANKS
    for c in hand:
        hand_rank_counts[rank(c)] += 1
    boss_pairs = sum(
        1 for r in range(NUM_RANKS)
        if hand_rank_counts[r] >= 2 and r > best_unseen_pair_rank
    )

    # --- rank pressure ---------------------------------------------------
    if unseen:
        rel = (sum(rank(c) for c in hand) / len(hand)
               - sum(rank(c) for c in unseen) / len(unseen)) / (NUM_RANKS - 1)
    else:
        rel = 1.0

    return (W_CONTROL * ctrl
            - W_LIABILITY * liability
            - W_TEMPO * tempo
            + W_BOSS_SINGLE * boss_singles
            + W_BOSS_PAIR * boss_pairs
            + W_RANK * rel)


def shaped_deltas(potentials: List[float]) -> List[float]:
    """Per-step shaping rewards for one trajectory: r_t = phi_{t+1} -
    phi_t, with phi after the last decision taken as 0 (terminal).  Sum
    telescopes to -phi_0, so episode return is unchanged up to a
    constant that depends only on the deal."""
    out = []
    for t in range(len(potentials)):
        nxt = potentials[t + 1] if t + 1 < len(potentials) else 0.0
        out.append(nxt - potentials[t])
    return out
