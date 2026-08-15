# Toward Optimal Big 2: Experiment Methodology & Research Plan

*A short working paper for this repository. It defines how we run and
report experiments, the agent ladder we are climbing, and why the design
choices follow the imperfect-information games literature.*

---

## 1. Problem framing

Big 2 is a 4-player, zero-sum, imperfect-information shedding game. The
closest solved analog is **not poker — it is Dou Dizhu**: combinatorial
action space (all singles/pairs/5-card poker hands from a 13-card hand),
shedding dynamics, and the full deck being dealt. The DouZero →
PerfectDou lineage is therefore our primary recipe, with techniques
borrowed from Suphx (Mahjong) and Pluribus (multiplayer poker) where they
transfer.

Two structural facts drive the design:

1. **The action space is the binding constraint.** Legal move sets
   routinely run into the hundreds (a 13-card single-suit hand admits
   1,300 combos). A fixed softmax head over enumerated actions is the
   wrong shape; we score *encoded actions* instead (§4, step 4).
2. **All 52 cards are known.** Unlike poker there is no hidden chance
   beyond the deal, so belief over opponent hands is a constrained
   combinatorial object (known counts, known played cards). This makes
   determinized search unusually effective and enables
   perfect-information critics during training.

A caution on "game theory optimal": with four players, Nash equilibria
are PPAD-hard to compute, do not compose across opponents, and carry no
value guarantee. Pluribus succeeded in 6-max poker anyway, but its
authors were explicit that the justification was empirical. Our "GTO
phase" is therefore **self-play against a diverse population** (step 6),
not regret minimization, and we measure *proxy exploitability* — train a
fresh best-response against a frozen policy and report how much it gains.

## 2. Testbed

- **Engine** (`big2/game.py`, `big2/combos.py`): exact house rules —
  suit order ♦<♣<♥<♠, rank order 3..A,2, lowest-dealt-card opens,
  pass locks by default, singles/pairs/5-card classes.
- **Rule variants** (`big2/rules.py`), kept deliberately small: lone
  triples on/off, pass lock-out vs. soft pass, flush compared
  suit-first vs. rank-first.
- **Scoring variants** (`big2/game.py:ScoringConfig`): payment =
  cards-remaining with tiered multipliers (10–12 cards ×2, 13 ×3), and
  a legacy 2-holder modifier for experiments. Each variant is run as a
  separate experiment; agents are expected to *adapt* to the variant
  (e.g. race harder when the ×2 tier looms).
- **Gym environment** (`big2/env.py`): action-masked Gymnasium env,
  MaskablePPO-compatible.

## 3. Evaluation protocol

Every result we report follows these rules:

1. **Optimize and report the actual scoring rule, not win rate.** The
   card-count multipliers make average point differential (ADP) and win
   probability meaningfully different objectives — the Dou Dizhu
   literature separates ADP from WP for exactly this reason. We report
   both: `avg_score (win_rate)`.
2. **Seat rotation / duplicate dealing.** Variance in Big 2 is brutal.
   Tournaments rotate strategies through all seat permutations
   (`big2/experiments.py`), and matched-deal evaluation reuses the same
   shuffles across candidates (common random numbers in
   `big2/rl.py:evaluate`). Bridge-style duplicate scoring — identical
   deals replayed with agents in every seat — is the standard for any
   result we claim is significant.
3. **Sample sizes.** Scripted-vs-scripted comparisons use ≥1,000 games
   per variant; anything within ±0.5 avg score needs ≥5,000 games or a
   paired (duplicate-deal) design before we call a winner.
4. **Zero-sum invariant.** Scores must sum to zero every game (asserted
   in tests); any violation is an engine bug, not a result.
5. **Fixed seeds, reported configs.** Every table in the README/docs
   states games-per-variant, seed policy, and the exact strategy
   arguments.

## 4. The agent ladder

The sequencing below is the project roadmap. Steps marked ✅ are
implemented in this repo; the rest are specified so the next
contributor can pick them up directly.

| # | Step | Status |
|---|------|--------|
| 1 | Fast vectorized env (bitboards) | ➖ partial: profiled pure-Python engine (~65 µs/step, ~10⁴ games/min) is enough through step 4; bitboard rewrite deferred until deep self-play needs 10⁶+ games |
| 2 | Scripted baselines: greedy-lowest → decomposition-optimal | ✅ `strategies.py`, `decomposition.py` |
| 3 | ISMCTS, no learning | ✅ `ismcts.py` (root-determinized; optional belief-posterior determinization) |
| 4 | DMC with action encoding | ✅ `dmc.py` (linear + belief features) and `nn.py`/`evolve.py` (MLP estimators, 1-3 hidden layers, trained at ~10⁶-game scale) |
| 5 | PPO + set-attention head + perfect-info critic + belief auxiliary | ➖ groundwork: exact belief module (`beliefs.py`) feeds features; MLP value nets exist; the learned belief head + policy-gradient head remain |
| 6 | League / PSRO population | ➖ two rungs: `league.py` (population sampling + frozen checkpoints) and `evolve.py` (PBT islands: hyperparameter evolution, migration, agents-only matchups); exploiters + meta-solver missing |
| 7 | Search at inference (policy prior + value + belief particles) | ⬜ planned (belief particles exist — `BeliefState.sample_worlds`) |
| 8 | Runtime opponent adaptation, bounded deviation | ➖ primitive: pass-honesty likelihood weighting is a fixed opponent model; learned/adaptive models planned |

**Step 2 — decomposition-optimal.** Minimum-plays hand decomposition is
a solvable subproblem: memoized branch-and-bound over partitions,
branching only on units containing the lowest remaining card (each
partition enumerated once). Exact on random 13-card hands in <1 ms,
0.2 s worst case (13-card flush). It serves three roles at once: a
strong scripted baseline (`decomp`), a high-signal feature for learned
agents, and an endgame oracle. Don't make a network rediscover it.

**Step 3 — ISMCTS.** Root-level determinized search: sample opponent
hands consistent with observed counts and played cards, allocate
rollouts across root moves by UCB1, play the best mean. PIMC has known
pathologies (strategy fusion, non-locality — Frank & Basin), but Big 2
sits in the favorable regime identified by Long et al. (AAAI 2010):
high leaf correlation, low disambiguation. At 100–250 determinizations
it beats all scripted baselines with zero training and is our standing
sanity-check opponent. Full tree-based ISMCTS (per-information-set
statistics below the root) is the upgrade if we need it stronger.

**Step 4 — DMC with action encoding (first real agent).** DouZero's
core trick: encode each candidate action as *input* — 52-dim card mask
plus move metadata concatenated with the state — and learn Q(s, a) by
regressing on final Monte-Carlo returns from self-play, all seats
sharing weights, epsilon-greedy behavior. Episodes are short (~13
tricks) and returns low-variance, so plain Monte-Carlo targets dodge
bootstrapping pathologies entirely. Rewards are raw game scores
(ADP-style), so the agent feels the scoring variant it trains under.
Our implementation is linear (numpy, minutes on CPU); swapping the
estimator for a small MLP in torch — keeping the encoding — is the
next increment, followed by parallel actors (DouZero used 45 actors on
a single machine).

### Beliefs: exploiting the known deck (`big2/beliefs.py`)

Because the whole deck is visible-or-held, the hidden state from any
viewpoint is an assignment of the *unseen* cards (deck − my hand −
played) to known opponent hand counts. Two layers exploit this:

- **Analytic (exact, O(1) per query).** Marginals are hypergeometric:
  an opponent with n of the U unseen cards holds a specific card with
  probability n/U, and at least one of H target cards with probability
  1 − C(U−H, n)/C(U, n). This gives the per-card probability map,
  P(holds a 2 / an ace / any card beating this single), and per-rank
  maps — and it sharpens automatically as U shrinks. Endgame
  elimination falls out for free: when U equals a player's count their
  hand is *known* (`known_hand`).
- **Monte-Carlo with pass evidence.** Sample worlds consistent with the
  counts; weight each world by pass plausibility — a world where a
  passer could have beaten the table is down-weighted by
  ``pass_honesty`` (strategic passing is legal, so passes are soft
  evidence, not voids). Whole-hand events (has a pair/triple/bomb,
  can beat this 5-card combo) come from these weighted samples.

Where beliefs plug in today: DMC's action encoding carries
P(opponent beats this move's top card) and P(holds a 2) per opponent;
the Gym observation carries the per-rank belief map; ISMCTS can draw
determinizations from the pass-weighted posterior instead of uniform;
and the UI has a belief panel showing all of it live. The step-5
*neural* belief head learns what this module cannot: correlations
induced by opponents' *policies* (what they chose to play, not just
what they legally could hold).

### League training (`big2/league.py`)

Trainables stop playing fixed benchmarks and instead train against a
**population**: scripted anchors (smart, dumper, lowest, decomp), the
other trainables (CEM linear, DMC), and frozen checkpoints of past
generations. Each generation: rated matches with random 4-member
lineups → DMC trains with opponents sampled per episode from the field
→ CEM candidates are evaluated against sampled lineups (common random
numbers per iteration) → frozen copies join the population (bounded
history, anchors never pruned). Ratings are mean score/game vs the
field. This is PSRO-lite: still missing are dedicated exploiter agents
(train a best response to each frozen champion, add it to the pool —
also our exploitability proxy) and a meta-solver over the empirical
payoff matrix instead of uniform opponent sampling.

### Evolutionary population training at scale (`big2/evolve.py`)

The 10⁶-game trainer. Agents **only play other agents** — no benchmark
evaluation inside the loop; ratings come from the training games
themselves. Design, following population-based training (PBT):

- **Islands, one per core.** Each island evolves an independent
  population of MLP Q-agents (encoding v3: DouZero action encoding +
  belief features + hand-structure/decomposition features) whose
  *hyperparameters are sampled*: 1-3 hidden layers up to 256 wide,
  log-uniform learning rates, exploration epsilons.
- **Random matchups.** Seats are drawn per game: ~60% trainables (an
  agent can draw several seats — true self-play), ~20% frozen opponents
  (round checkpoints of this island's past best + champions migrated
  from other islands), ~20% scripted floor (dumper, decomp, lowest) so
  the population can't drift into conventions that lose to simple play.
  Several trainables learn from the same game, each from its own seat.
- **Exploit + explore.** Each round the worst-rated trainee adopts the
  best's network and architecture, then mutates learning rate and
  epsilon. The round's best is frozen into the opponent pool.
- **1v1 curriculum.** The first ~60% of games are 2-player (~3x faster,
  denser reward signal), then the population graduates to 4-player.
- **Champion playoff.** Island finalists meet in a seat-rotated
  4-player tournament; the winner ships as `evo_mlp.npz`.

Known limitations, stated up front: Monte-Carlo returns on a shared
scale (÷39) mean 2p and 4p rewards are comparable but not identical
distributions across the phase switch; ratings are vs a shifting field
(non-stationary), which is why champion selection uses the final round
and a playoff rather than lifetime averages; and mutation never *grows*
architectures mid-lineage (deeper nets enter via fresh sampling and
exploit-adoption).

**Step 5 — PPO with a set head.** Charlesworth (2018) already showed
PPO self-play reaches strong human-competitive play in 4-player Big 2;
his state/action encoding is worth mirroring. Modern upgrades, per
PerfectDou and Suphx:
- *set-attention / pointer head*: encode the legal action set, attend
  over it, softmax across the set — a proper distribution over
  variable action sets, unlocking policy gradients;
- *perfect-information critic*: the critic sees all four hands during
  training while the actor sees only legal observations ("perfect
  training, imperfect execution");
- *belief auxiliary head*: predict each opponent's hand as a masked
  52-way distribution; the auxiliary loss improves the trunk even if
  the output is never used at inference (Suphx, Hanabi literature).

**Step 6 — league / PSRO.** A single self-play run in a 4-player game
converges to self-consistent conventions that collapse against
unfamiliar opponents. The league keeps main agents, frozen historical
checkpoints, and dedicated exploiters; report proxy exploitability
(best-response gain) per generation.

**Step 7 — search at inference.** Determinized MCTS with the learned
policy as prior, learned value at leaves, and belief particles filtered
by observed play (AlphaZero-ish; Player of Games is the principled
version). The engine's `clone()` + determinization in `ismcts.py`
already provide the scaffolding.

**Step 8 — opponent adaptation.** Within-match: a recurrent/transformer
encoder over move history yields an opponent embedding (nearly free
once history encoding exists). Across-match: Suphx-style parametric
Monte-Carlo policy adaptation. Guardrail: exploitation raises our own
exploitability — deviations must be bounded (restricted Nash response;
Ganzfried & Sandholm's safe exploitation).

## 5. Current results (tiered scoring, 1,000+ games, rotated seats)

See README for the live table. Summary as of this revision: the
CEM-trained linear move-scorer and DMC beat all scripted baselines;
ISMCTS is the strongest no-training opponent; `highest` (always play
your strongest) is reliably the worst strategy in every variant —
burning control cards early is the cardinal sin of Big 2.

## 6. Variant experiments we care about

- **2-holder rule on/off**: does optimal play shed 2s early (dump risk)
  or hold them longer (control)? Compare per-variant specialists.
- **Tiered multipliers**: how much does the ×2/×3 cliff change racing
  behavior when an opponent is close to out?
- **Lone triples**: triples strengthen decompositions (full houses
  compete with them for cards) — measure decomposition depth and win
  rates with/without.
- **Soft pass vs. lock-out**: soft pass increases information leakage
  (passes are cheaper), which should shift the value of strategic
  passing.

## 7. References

- Zha et al., *DouZero: Mastering DouDizhu with Self-Play Deep RL*, ICML 2021.
- Yang et al., *PerfectDou: Dominating DouDizhu with Perfect Information Distillation*, NeurIPS 2022.
- Li et al., *Suphx: Mastering Mahjong with Deep RL*, 2020.
- Brown & Sandholm, *Superhuman AI for multiplayer poker* (Pluribus), Science 2019.
- Schmid et al., *Player of Games*, 2021.
- Long et al., *Understanding the success of perfect information Monte Carlo sampling*, AAAI 2010.
- Frank & Basin, *Search in games with incomplete information*, AIJ 1998.
- Charlesworth, *Application of self-play RL to a four-player game of
  imperfect information* (Big 2 PPO), 2018 —
  https://github.com/henrycharlesworth/big2_PPOalgorithm
- Big 2 opponent/movement prediction studies:
  https://www.mdpi.com/2076-3417/11/9/4206 and
  https://www.researchgate.net/publication/359927024
- Rules reference: https://www.pagat.com/climbing/bigtwo.html
