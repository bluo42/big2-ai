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
  beliefs.py        opponent-hand probabilities: analytic + Monte-Carlo layers
  env.py            Gymnasium env (action-masked, belief-augmented obs)
  strategies.py     scripted baselines
  decomposition.py  exact minimum-plays hand decomposition + strategy
  ismcts.py         determinized ISMCTS (optionally belief-weighted worlds)
  rl.py             linear move-scorer trained with cross-entropy method
  dmc.py            DouZero-style DMC: Q(s, a) over encoded actions + beliefs
  nn.py             numpy MLP Q-network (1-3 hidden layers, Adam)
  features.py       encoding v3: action encoding + beliefs + hand structure
  league.py         population training: trainables vs the field, not benchmarks
  evolve.py         PBT islands: ~10^6 agents-only games, 1v1 curriculum
  experiments.py    seat-rotated tournaments across scoring variants
  server.py         Flask backend for the web UI
  static/           the UI (self-contained HTML/JS/CSS)
  policies/         trained weights (committed)
docs/EXPERIMENTS.md experiment methodology & research roadmap
tests/              57 unit tests
```

## Play it

```bash
pip install -r requirements.txt
python -m big2.server          # then open http://127.0.0.1:8080
```

### Deploy to Vercel

The app is stateless (the full game state travels with each request), so
it runs as-is on Vercel's Python runtime — `vercel.json` routes every
path to `api/index.py`, which serves the same Flask app you run locally.
Two ways to deploy:

1. **Import the repo** at [vercel.com/new](https://vercel.com/new),
   pick this repository/branch, and deploy — no build settings needed.
2. **CLI**: `npm i -g vercel && vercel` from the repo root.

Note: because the client holds the full state (required for serverless),
a determined player can inspect opponents' hands in dev tools — fine for
a demo and exactly what the admin explorer wants; not for money games.

### Game explorer (`/admin`)

Watch how the strategies actually play: pick 2-4 agents, simulate games,
and step through replays with **every hand exposed** — action-by-action
cursor, trick-separated log, autoplay, keyboard navigation. The same
page charts **training progress** while `big2.evolve` runs: probe scores
vs fixed baselines (solid) and vs the anchor champions (dashed) per
island, so you can see improvement and plateaus as games accumulate.

- Choose **1, 2, or 3 AI opponents** (2-4 players total; with fewer than 4,
  the undealt cards stay hidden, and the lowest card actually in play opens).
- Pick each AI: `Smart` (heuristic), `DMC` / `Linear` (trained), `ISMCTS`
  (search), `Decomposition`, and simpler baselines.
- Adjust **house rules** before dealing: lone triples, pass lock-out vs.
  soft pass, flush comparison, card-count multipliers, 2-holder penalty.
- In game: click cards to select (the Play button labels the combo it
  forms), **sort by rank or suit**, follow the **action history** panel,
  and ask for a **hint** from the trained policy.
- Toggle **🧠 Beliefs** to see, per opponent, live probabilities of
  holding a 2 / an ace / a pair / a triple, the chance they can beat
  what's on the table, and a 13-rank heat strip — sharpening to
  certainty ("hand known") as the endgame eliminates possibilities.

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
| `pass_locks` | **off** (soft pass: you may pass and still play later in the same trick; the trick ends on a full round of consecutive passes) | on = passing locks you out for the trick |
| `flush_rank_first` | off | on = flushes compare top ranks before suit (poker style) |
| `ScoringConfig.big_hand_double/full_hand_triple` | on | tiered card-count multipliers |
| `ScoringConfig.two_modifier` | off | legacy: holding a 2 at game end adds the base payment again |

## Beliefs: probability maps of hidden hands

Big 2 deals the whole deck, so from any seat the hidden state is just an
assignment of the unseen cards to known opponent hand counts.
`big2/beliefs.py` exploits that with two layers:

- **Analytic (exact)** — hypergeometric marginals for the per-card and
  per-rank probability maps, "holds a 2 / an ace", and "holds a single
  beating this card"; these sharpen automatically as cards get played,
  and collapse to certainty in the endgame (`known_hand`).
- **Monte-Carlo with pass evidence** — sampled worlds, down-weighted
  when an observed pass looks dishonest in that world (`pass_honesty`),
  give whole-hand events: has a pair / triple / bomb, can beat this
  combo.

Beliefs feed the DMC action encoding (P(opponent beats this move),
P(holds a 2)), the Gym observation (per-rank belief maps), optional
belief-weighted determinization in ISMCTS, and the UI belief panel.

**Opponent modeling** (`big2/opponents.py`): each agent also watches how
its opponents play — pass rate, rank aggression, multi-card tendency,
2s spent — and uses it two ways: the pass-evidence weight adapts *per
opponent* (a habitual passer's passes carry little information; a rare
passer's pass is strong evidence), and the style vector plus
feature-driven holding guesses (P(pair), P(triple) per opponent) are
inputs to the v4 neural encoding, so learned agents condition on who
they're playing against.

## League training: candidates vs the field

`python -m big2.league` replaces benchmark training: a population of
scripted anchors, trainable agents (CEM + DMC), and frozen checkpoints
of past generations plays rated matches; each generation the DMC trains
with opponents *sampled from the field per episode*, CEM candidates are
scored against sampled lineups, and frozen copies of both join the
population. League champions are saved over the standard policy files.
See `docs/EXPERIMENTS.md` for what separates this from full PSRO
(exploiters + meta-solver) and the plan to get there.

## Evolutionary training at scale (`python -m big2.evolve`)

The ~1,000,000-game trainer. MLP Q-agents (encoding v3, 1-3 hidden
layers up to 256 wide) evolve on **islands** (one process per core):
seats in every game are drawn at random from the trainable population,
frozen round-checkpoints, champions migrated from other islands, and a
thin dumper/decomp/lowest scripted floor — agents only ever play other
agents. Each round the worst-rated trainee adopts the best's network
and mutates its learning rate and exploration (PBT exploit + explore).
The first ~60% of games are played **1v1** for cheap, dense learning
signal before graduating to 4-player. Island champions meet in a final
seat-rotated playoff and the winner ships as `big2/policies/evo_mlp.npz`
(the `evo` agent in tournaments and the UI).

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
   UCB rollout allocation; optional belief-posterior determinization.
   Strongest no-training opponent.
4. ✅ *DMC with action encoding* — DouZero's trick: encode each candidate
   move as input (52-bit mask + metadata + belief features) and regress
   Q(s, a) on final Monte-Carlo returns. Linear/numpy here; torch MLP is
   the designated upgrade.
5. ➖ PPO + set-attention head + perfect-info critic + belief auxiliary —
   the exact belief module is the groundwork; the learned head comes next.
6. ➖ League/PSRO population — `league.py` is the first rung (population
   sampling + frozen checkpoints); exploiters and a meta-solver remain.
7. ⬜ Search at inference (policy prior + value + belief particles —
   `BeliefState.sample_worlds` already provides the particles).
8. ➖ Runtime opponent adaptation — pass-honesty weighting is a fixed
   primitive opponent model; learned/adaptive models with bounded
   deviation remain.

There is also `linear` — a CEM-trained move scorer over hand-crafted
features (`big2/rl.py`) that predates the ladder and remains a strong
reference.

## Results (cells: avg score/game, win rate)

Current-rule results are in "[Current results under the house
rules](#current-results-under-the-house-rules-soft-pass-tiered-seed-17)"
below; the tables immediately following are **historical** — measured
under the original pass-lock-out rule (seat-rotated, seed 1) and kept
for the record of how each ladder step changed the standings.

Baselines, 1,000 games/variant:

```
   variant |             lowest |            highest |             dumper |              smart
----------------------------------------------------------------------------------------------
     plain |       -1.35 ( 20%) |       -3.89 (  4%) |       +3.06 ( 37%) |       +2.18 ( 39%)
    tiered |       -1.55 ( 20%) |       -4.38 (  4%) |       +3.65 ( 37%) |       +2.28 ( 39%)
       two |       -1.56 ( 20%) |       -3.85 (  4%) |       +3.20 ( 37%) |       +2.21 ( 39%)
tiered+two |       -1.76 ( 20%) |       -4.34 (  4%) |       +3.79 ( 37%) |       +2.31 ( 39%)
```

Stronger lineup, 1,000 games/variant, seed 3 — `linear` and `dmc` are the
**league-trained champions** (see below):

```
   variant |              smart |             decomp |             linear |                dmc
----------------------------------------------------------------------------------------------
    tiered |       -1.26 ( 24%) |       -1.23 ( 25%) |       +2.42 ( 31%) |       +0.07 ( 20%)
     plain |       -0.86 ( 24%) |       -0.67 ( 25%) |       +1.70 ( 31%) |       -0.17 ( 20%)
```

League final standings (4 generations, 1,600 final rated games, seed 11) —
the CEM lineage tops the field and every trainable outrates the anchors:

```
          member       kind  games     avg
             cem        cem    623   +2.58
          cem-g4 checkpoint    651   +2.45
          cem-g3 checkpoint    629   +1.24
             dmc        dmc    625   +0.31
          dmc-g4 checkpoint    673   +0.31
          dumper   scripted    640   -0.40
           smart   scripted    652   -0.42
          decomp   scripted    654   -0.97
          dmc-g3 checkpoint    634   -1.55
          lowest   scripted    619   -3.62
```

ISMCTS (100 sims), 200 games, tiered — the strongest no-training agent:

```
   variant |             ismcts |              smart |             decomp |             dumper
----------------------------------------------------------------------------------------------
    tiered |       +1.60 ( 32%) |       +0.33 ( 27%) |       -1.65 ( 22%) |       -0.28 ( 20%)
```

### Current results under the house rules (soft pass, tiered, seed 17)

All numbers below use the current defaults. Baselines, 1,000 games:

```
   variant |             lowest |            highest |             dumper |              smart
----------------------------------------------------------------------------------------------
    tiered |       -0.82 ( 23%) |       -4.43 (  4%) |       +3.12 ( 37%) |       +2.13 ( 37%)
```

Strong lineup, 1,000 games — `linear`/`evo` are the soft-pass league CEM
champion and the anchored-evolution champion:

```
   variant |              smart |             decomp |             linear |                evo
----------------------------------------------------------------------------------------------
    tiered |       -1.43 ( 21%) |       -1.30 ( 26%) |       +2.32 ( 30%) |       +0.41 ( 24%)
```

### The anchored 1,000,000-game evolution run (soft pass, v4 features)

Second million-game run, this time with the previous champions riding as
**anchor opponents** inside every island's matchmaking, encoding v4
(opponent modeling + holding guesses + decomposition deltas +
payment-tier risk), a 4-layer architecture in the gene pool, and
**plateau probes every 25k games** (chart them at `/admin`). The probe
story: islands reach scripted-baseline parity ~100k games in, close a
3.5-point anchor gap to parity by ~150k, break above the anchors
(+1.3 to +1.7 at peaks) around 200k, then **oscillate around anchor
parity** — a visible plateau at this model capacity. Island 2's
[128, 64] champion won the playoff (+0.67/game) and ships as `evo`: it
beats every agent in the repo except the CEM linear move-scorer, which
keeps the overall crown (+2.32 vs +0.41 in the shared table above).
The plateau at linear-parity across two million-game runs is the
clearest signal yet that the next step is a capacity/algorithm jump
(torch MLP with more data per parameter, PPO with a set head, learned
belief head), not more games at this scale.

### The first 1,000,000-game evolution run

4 islands × 250k games (60% of them 1v1), populations of MLP Q-agents
with evolved hyperparameters, ~75 minutes on 4 cores. Island 2's
champion — a **2-layer [128, 64]** net, lr ≈ 0.0017 — won the
cross-island playoff (+0.52/game over 1,000 games), beating all three
islands that converged on deeper 3-layer nets. Against the established
agents (1,000 games/variant, seed 9):

```
   variant |              smart |             decomp |             linear |                evo
----------------------------------------------------------------------------------------------
    tiered |       -1.00 ( 24%) |       -0.88 ( 27%) |       +1.62 ( 27%) |       +0.26 ( 22%)
     plain |       -0.77 ( 24%) |       -0.27 ( 27%) |       +1.02 ( 27%) |       +0.02 ( 22%)
```

and in a trained-agents-vs-search table (200 games, tiered): `evo`
+0.44 vs ISMCTS −0.94 — the first learned agent here to outrate
determinized search in a shared lineup. The honest headline: after a
million games, the evolved MLP beats search, DMC, and every scripted
baseline, but the CEM-trained linear move-scorer over hand-crafted
features is *still* the single strongest agent. Feature engineering
plus direct payoff optimization remains hard to beat at this scale —
closing that gap (deeper training, PPO, belief heads) is the live
research question the ladder's next steps answer.

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
python -m big2.league --generations 4           # league-train CEM + DMC vs the field
python -m big2.rl  --iters 25 --pop 32          # benchmark-train CEM (legacy)
python -m big2.dmc --episodes 60000             # self-play-train DMC (legacy)
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
