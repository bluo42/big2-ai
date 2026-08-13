# Big 2 — Playable UI, Gym Environment, and an RL Research Ladder

A [Gymnasium](https://gymnasium.farama.org/) take on the card game **Big 2**
(Deuces / 鋤大D): a pure-Python engine with configurable house rules, a sleek
web UI for playing against the AIs, scripted and learned strategies organized
as a research ladder (decomposition search → ISMCTS → DMC → deep RL), and a
documented experiment methodology (`docs/EXPERIMENTS.md`).

```
big2/
  cards.py          card encoding (3♦ = 0 ... 2♠ = 51)
  combos.py         hand classification, comparison, legal-move generation
  rules.py          house-rule variants (small, deliberate variant space)
  game.py           2-4 player engine: tricks, passing, scoring, clone()
  env.py            Gymnasium env (action-masked Discrete space)
  strategies.py     scripted baselines
  decomposition.py  exact minimum-plays hand decomposition + strategy
  ismcts.py         determinized ISMCTS reference opponent (no learning)
  rl.py             linear move-scorer trained with cross-entropy method
  dmc.py            DouZero-style DMC: Q(s, a) over encoded actions
  experiments.py    seat-rotated tournaments across scoring variants
  server.py         Flask backend for the web UI
  static/           the UI (self-contained HTML/JS/CSS)
  policies/         trained weights (committed)
docs/EXPERIMENTS.md experiment methodology & research roadmap
tests/              46 unit tests
```

## Play it

```bash
pip install -r requirements.txt
python -m big2.server          # then open http://127.0.0.1:8080
```

- Choose **1, 2, or 3 AI opponents** (2-4 players total; with fewer than 4,
  the undealt cards stay hidden, and the lowest card actually in play opens).
- Pick each AI: `Smart` (heuristic), `DMC` / `Linear` (trained), `ISMCTS`
  (search), `Decomposition`, and simpler baselines.
- Adjust **house rules** before dealing: lone triples, pass lock-out vs.
  soft pass, flush comparison, card-count multipliers, 2-holder penalty.
- In game: click cards to select (the Play button labels the combo it
  forms), **sort by rank or suit**, follow the **action history** panel,
  and ask for a **hint** from the trained policy.

## Rules implemented

- 4 players (2-3 supported), 13 cards each, seats play counter-clockwise.
- **Rank order** `3 < 4 < ... < K < A < 2`;
  **suit order** `diamonds < clubs < hearts < spades`.
- Holder of the lowest dealt card (3♦ in a 4-player game) leads the first
  trick and must include it.
- Classes: **singles, pairs, and 5-card poker hands**. Following means
  beating within the same size class; passing is always allowed when not
  leading. The trick winner leads any class.
- **5-card hierarchy**: straight < flush < full house < four-of-a-kind
  (+ kicker) < straight flush. Straights: top card's rank then suit (no 2s,
  no wrap-around). Flushes: suit first, then ranks. Full house: the triple.
- **Payment**: losers pay the winner cards-remaining, with tiered
  multipliers — **10-12 cards pay double, all 13 pay triple**.

### House-rule variants (`big2/rules.py`, all in the UI)

| flag | default | variant |
|------|---------|---------|
| `allow_triples` | off | lone triples playable as their own class, compared by rank |
| `pass_locks` | on | off = a pass only skips your turn (trick ends on a full round of consecutive passes) |
| `flush_rank_first` | off | on = flushes compare top ranks before suit (poker style) |
| `ScoringConfig.big_hand_double/full_hand_triple` | on | tiered card-count multipliers |
| `ScoringConfig.two_modifier` | off | legacy: holding a 2 at game end adds the base payment again |

## The agent ladder

Following the Dou Dizhu lineage (DouZero → PerfectDou) rather than poker —
see `docs/EXPERIMENTS.md` for the full methodology and roadmap.

1. ➖ *Fast vectorized env* — pure-Python engine profiled at ~65 µs/step,
   sufficient through step 4; bitboards deferred.
2. ✅ *Scripted baselines* — `lowest`, `highest`, `dumper`, `smart`, and
   `decomp`: exact **minimum-plays hand decomposition** (memoized
   branch-and-bound, <1 ms typical) used as a baseline, a feature, and an
   endgame oracle.
3. ✅ *ISMCTS* — root-determinized search over sampled opponent hands with
   UCB rollout allocation. Strongest no-training opponent.
4. ✅ *DMC with action encoding* — DouZero's trick: encode each candidate
   move as input (52-bit mask + metadata) and regress Q(s, a) on final
   Monte-Carlo returns from shared-weights self-play. Linear/numpy here;
   torch MLP is the designated upgrade.
5. ⬜ PPO + set-attention head + perfect-info critic + belief auxiliary.
6. ⬜ League/PSRO population (proxy exploitability via best-response).
7. ⬜ Search at inference (policy prior + value + belief particles).
8. ⬜ Runtime opponent adaptation with bounded deviation.

There is also `linear` — a CEM-trained move scorer over hand-crafted
features (`big2/rl.py`) that predates the ladder and remains a strong
reference.

## Results (seat-rotated, seed 1; cells: avg score/game, win rate)

Baselines, 1,000 games/variant:

```
   variant |             lowest |            highest |             dumper |              smart
----------------------------------------------------------------------------------------------
     plain |       -1.35 ( 20%) |       -3.89 (  4%) |       +3.06 ( 37%) |       +2.18 ( 39%)
    tiered |       -1.55 ( 20%) |       -4.38 (  4%) |       +3.65 ( 37%) |       +2.28 ( 39%)
       two |       -1.56 ( 20%) |       -3.85 (  4%) |       +3.20 ( 37%) |       +2.21 ( 39%)
tiered+two |       -1.76 ( 20%) |       -4.34 (  4%) |       +3.79 ( 37%) |       +2.31 ( 39%)
```

Stronger lineup, 1,000 games/variant — the trained agents on top:

```
   variant |              smart |             decomp |             linear |                dmc
----------------------------------------------------------------------------------------------
    tiered |       -1.55 ( 21%) |       -1.44 ( 26%) |       +2.22 ( 30%) |       +0.77 ( 24%)
     plain |       -1.22 ( 21%) |       -0.72 ( 26%) |       +1.63 ( 30%) |       +0.31 ( 24%)
```

ISMCTS (100 sims), 200 games, tiered — the strongest no-training agent:

```
   variant |             ismcts |              smart |             decomp |             dumper
----------------------------------------------------------------------------------------------
    tiered |       +1.60 ( 32%) |       +0.33 ( 27%) |       -1.65 ( 22%) |       -0.28 ( 20%)
```

Early findings worth keeping: always playing your strongest cards
(`highest`) is the reliably worst strategy; `decomp` wins often (26% in a
strong lineup) but its structure-preserving passes make its *losses* heavy
under tiered multipliers — win rate and average score genuinely diverge,
which is exactly why we report both.

## Training & experiments

```bash
python -m unittest discover -s tests            # test suite
python -m big2.experiments --games 1000         # tournament across variants
python -m big2.experiments --games 500 --strategies smart decomp linear dmc
python -m big2.rl  --iters 25 --pop 32          # retrain CEM linear policy
python -m big2.dmc --episodes 60000             # retrain DMC policy
```

### Gym environment

```python
import numpy as np
from big2.env import Big2Env
from big2.strategies import SmartHeuristic, FiveCardDumper, PlayLowest

env = Big2Env(opponents=[SmartHeuristic(), FiveCardDumper(), PlayLowest()])
obs, info = env.reset(seed=0)
done = False
while not done:
    action = int(np.random.choice(np.flatnonzero(obs["action_mask"])))
    obs, reward, done, truncated, info = env.step(action)
```

Action space: `Discrete(2048)` slots — 0 is PASS, slots 1..N the legal
combos sorted weakest-first; masks via `obs["action_mask"]` /
`env.action_masks()` (sb3-contrib MaskablePPO-compatible). Terminal reward
is the seat's score under the configured `ScoringConfig`.

### Engine

```python
from big2 import Big2Game, RuleConfig, ScoringConfig
from big2.strategies import SmartHeuristic
from big2.ismcts import ISMCTSStrategy

game = Big2Game(
    rules=RuleConfig(allow_triples=True, pass_locks=False),
    scoring=ScoringConfig(),          # tiered: 10-12 x2, 13 x3
    num_players=3,
)
print(game.play_out([ISMCTSStrategy(n_sims=100), SmartHeuristic(), SmartHeuristic()]))
```
