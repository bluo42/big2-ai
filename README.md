# Big 2 — Gym Environment, Strategies, and RL Testbed

A [Gymnasium](https://gymnasium.farama.org/) take on the card game **Big 2**
(Deuces / 鋤大D), plus a pure-Python game engine, baseline strategies, a
scoring-variant experiment runner, and a first learned policy — the
foundation for exploring optimal play with reinforcement learning, ML, and
game-theory methods.

```
big2/
  cards.py        card encoding (3♦ = 0 ... 2♠ = 51)
  combos.py       hand classification, comparison, legal-move generation
  game.py         4-player engine: tricks, passing, scoring modifiers
  env.py          Gymnasium env (action-masked Discrete space)
  strategies.py   baseline strategies
  rl.py           linear move-scorer trained with cross-entropy method
  experiments.py  strategy tournaments across scoring variants
tests/            unit tests for rules, scoring, game flow, and the env
```

## Rules implemented

- 4 players, 13 cards each. Seats play in order 0 → 1 → 2 → 3
  (counter-clockwise around a table).
- **Rank order** `3 < 4 < ... < K < A < 2`;
  **suit order** `diamonds < clubs < hearts < spades`.
- The holder of the **3♦ leads the first trick** and the first play must
  include it.
- Classes: **singles, pairs, and 5-card poker hands** — no lone triples.
- Following a play means beating it within the same size class; **passing is
  always allowed when not leading**, but a pass locks you out for the rest of
  that trick. The trick winner leads the next trick with any class.
- **5-card hierarchy**: straight < flush < full house < four-of-a-kind
  (+ kicker) < straight flush.
  - *Straight*: highest card's rank, then that card's suit. 2s can't appear
    in straights and there's no wrap-around; `10-J-Q-K-A` is the top straight.
  - *Flush*: suit first, then highest card (a ♠ flush beats any ♥ flush).
  - *Full house*: rank of the triple. *Quads*: rank of the four.
- **Winning & payment**: first player to shed all cards wins; every other
  player pays the winner their cards-remaining. Modifiers (each toggleable,
  each adds the base payment again):
  - holding a 2 at game end (`two_modifier`, optional `per_two` stacking),
  - holding ≥ 10 cards at game end (`big_hand_modifier`).

  A loser with 11 cards including a 2 pays `11 × 3 = 33` with both modifiers.

Assumptions where the house rules were silent: four-of-a-kind + kicker and
straight flushes are included as 5-card hands (standard Big 2); there are no
cross-class bombs; strategic passing is allowed. All of these live in
`combos.py` / `game.py` and are easy to change.

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests          # run the test suite
python -m big2.experiments --games 1000       # baseline tournament
python -m big2.rl --iters 25 --pop 32         # train the linear policy
python -m big2.experiments --games 2000 --policy-file big2/policies/linear_cem.npz
```

### Using the engine directly

```python
from big2 import Big2Game, ScoringConfig
from big2.strategies import PlayLowest, SmartHeuristic

game = Big2Game(scoring=ScoringConfig(two_modifier=True, big_hand_modifier=True))
scores = game.play_out([PlayLowest(), PlayLowest(), SmartHeuristic(), SmartHeuristic()])
print(scores)  # zero-sum, e.g. {0: -5, 1: -14, 2: 27, 3: -8}
```

### Using the Gym environment

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
print(reward, info["scores"])
```

- **Action space**: `Discrete(2048)` slots — slot 0 is PASS, slots 1..N are
  the current legal combos **sorted weakest-first** (so the lowest legal slot
  is always the weakest play). The mask is in `obs["action_mask"]` and via
  `env.action_masks()` (sb3-contrib `MaskablePPO`-compatible).
- **Observation**: 175-dim vector — own hand (52), table combo (52) + type
  one-hot, all played cards (52), opponent card counts, passed flags, table
  owner, leading flag.
- **Reward**: 0 until the game ends; terminal reward is the agent's score
  under the configured `ScoringConfig`.

## Baseline strategies

| name      | idea |
|-----------|------|
| `random`  | uniform over legal moves (+ pass) |
| `lowest`  | always the weakest feasible play; leads its lowest single |
| `highest` | always the strongest feasible play; leads its biggest class |
| `dumper`  | leads 5-card hands (lowest first), then pairs, then singles; follows with the weakest play |
| `smart`   | partitions its hand into units (5-card hands / pairs / singles), refuses to break units while the hand is big, passes to protect strength, plays high to deny nearly-finished opponents |

### Tournament results (1000 games/variant, seats rotated, seed 1)

```
   variant |             lowest |            highest |             dumper |              smart
----------------------------------------------------------------------------------------------
     plain |       -1.35 ( 20%) |       -3.89 (  4%) |       +3.06 ( 37%) |       +2.18 ( 39%)
       two |       -1.56 ( 20%) |       -3.85 (  4%) |       +3.20 ( 37%) |       +2.21 ( 39%)
       big |       -1.54 ( 20%) |       -4.41 (  4%) |       +3.66 ( 37%) |       +2.28 ( 39%)
   two+big |       -1.75 ( 20%) |       -4.37 (  4%) |       +3.81 ( 37%) |       +2.31 ( 39%)

cells: average score per game (win rate)
```

Early findings: greedily playing your strongest cards (`highest`) is the
worst thing you can do; shedding 5-card hands early (`dumper`) and
structure-preserving play (`smart`) dominate. The scoring modifiers mostly
amplify the spread rather than reorder these fixed strategies — adapting *to*
the modifiers (e.g. dumping 2s early under the `two` rule, racing under the
`big` rule) is exactly what a learned policy should discover.

## First learned policy (`big2/rl.py`)

A linear scorer over ~20 state/move features (move strength, cards left,
whether the move breaks a hand unit, whether it spends a 2, danger/endgame
flags, ...) picks the argmax-scoring legal option each turn. Weights are
trained with the **cross-entropy method** against fixed opponents
(`lowest` / `dumper` / `smart`), with common random numbers across each
population. Trained weights live in `big2/policies/linear_cem.npz`.

## Roadmap toward optimal play

1. **Better features / self-play CEM** — train against copies of itself
   instead of fixed opponents to avoid overfitting to exploitable baselines.
2. **Deep RL** — the env is MaskablePPO-ready (sb3-contrib) out of the box;
   DQN over (state, move-features) pairs is another natural fit for the
   variable action set.
3. **Counterfactual regret / game theory** — Big 2 is zero-sum but 4-player
   imperfect-information; neural-CFR or best-response ladders
   (policy iteration through exploiters) are the principled route.
4. **Per-variant specialists** — train one policy per scoring variant
   (`plain`, `two`, `big`, `two+big`) and compare how optimal play shifts:
   how early do you shed 2s when holding them is expensive? How much do you
   race when the ≥10-card penalty looms?
5. **Search hybrids** — determinized MCTS over the engine (deal opponents'
   unseen cards, roll out with the learned policy) for a strong reference
   player.
