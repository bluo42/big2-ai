# The Depth Program

*2026-08-19.  The generation after v2/Sicario/Leonidas/Khabib: train
fewer games, think harder per move, and put the learning signal where
the games are actually won.*

## Why: the credit-assignment measurement

The four current models are beaten by strong humans in a specific,
measured way (deviation study on 570 recorded games: 378 disagreements,
163 provably better for the human, concentrated in early-hand patience).
The cause is not features and not the deal — it is where the learning
signal reaches.  The value head's correlation with the final payout, by
stage of the hand (khabib_v1, 150 games):

| cards left | corr | pred sd vs actual sd |
|-----------:|-----:|---------------------:|
| 52–43      | 0.41 | 4.5 / 10.4           |
| 42–35      | 0.53 | 4.7 / 9.6            |
| 34–27      | 0.67 | 5.6 / 9.7            |
| 26–19      | 0.77 | 7.0 / 8.9            |
| 18–11      | 0.84 | 5.2 / 6.6            |
| 10–1       | 0.82 | 3.8 / 3.8            |

Early decisions train on a gradient that is ~83% noise; late decisions
on one that is ~70% signal.  The bots are therefore superhuman late
(solver + sharp values) and loose early — the exact mirror of where
humans beat them.  Every workstream below attacks that asymmetry:
**depth of signal per game, not number of games.**

A second observation from self-play (Brandon): *who wins the trick
matters beyond who leads next* — the player immediately before you
winning it is less bad than the player immediately after you, because
of where it leaves you in the response order.  That is a trick-level
quantity no terminal reward can teach efficiently.  It becomes an
explicit potential (workstream 3).

## The models and the naming scheme

The training generation is renamed to say what each model *is* rather
than who it was built to kill:

| new canonical name   | lineage (file)            | recipe                              |
|----------------------|---------------------------|--------------------------------------|
| **v2_patient**       | new — copy of wangbot_v2  | AWR on measured-advantage human decisions |
| **v2_adversarial**   | sicario_v1                | trained to exploit WangBot            |
| **v2_self_trained**  | leonidas_v1               | humanlike start, imitation anchor     |
| **v2_human_trained** | khabib_v1                 | fresh recipe vs the full prior field  |

Deployed labels and stamps (`v2@…`, `Sicario@…`, `Khabib@…`) are
unchanged so the leaderboard, saved games, and per-bot records keep
resolving; the new names are registered as kind aliases and are the
canonical names in training code and docs from here on.

## Workstreams

### 1. v2_patient — imitate what provably beat us
Extract every recorded human decision that measurably beat the bot's
choice (deviation.py plays both branches over belief-sampled deals).
Fine-tune a copy of v2 with advantage-weighted regression
(offline.train_awr) plus a KL anchor to v2 so it stays a strong player
that *also* has the humans' early-hand patience.  Credit assignment is
bypassed entirely here: the advantage of each decision is measured by
playout, not bootstrapped.  v2_patient then enters the training diet at
high weight — the bots only learn that impatience is punishable if
someone at the table punishes it.

### 2. Policy-aware card beliefs (Bayesian, damped)
Today's beliefs are deterministic exclusion + hypergeometric — correct
under uniform play, blind to evidence.  But actions leak information:
a player who leads a lone single early is less likely to hold its
pair-mate; the probability the others hold one rises.  From the human
replay corpus plus AI self-play games we fit **odds-ratio factors**
per (action context, holding) event against the combinatorial
baseline, and apply them multiplicatively with damping (exponent
λ≈0.25, factors clamped) — deliberately small adjustments for now:
free gains without betting the posterior on a fitted model.  These
factors reweight both the per-card probability map and the
determinization sampler feeding IS-MCTS, so search worlds become
consistent with the play so far, not merely with the card count.
Later: a library of policy types with probabilistic player assignment,
P(holdings | player type) — the factors table is deliberately keyed so
per-type tables can slot in.

### 3. Trick-level rewards (potential-based, search-visible)
A potential Φ(state, seat) encodes position value between tricks:

* **control** — leading a fresh trick is the high-reward state;
* **winner distance** — when a trick resolves to a lead by seat w, the
  value to you depends on (w − you) mod 4: yourself best, the player
  immediately before you least bad, the player immediately after you
  worst (you respond last to their lead);
* **cards remaining**, on the scoring ladder actually used at payout;
* **minimum plays to shed** the hand (partition size — fewer tricks
  needed is closer to out);
* **boss singles / boss pairs** — how many of your singles/pairs are
  the strongest among cards not yet seen;
* **relative rank** — your mean rank vs the mean rank of the unseen.

Shaping reward at each decision is γΦ(s′) − Φ(s) with γ=1 and
Φ(terminal)=0: telescopes to a per-episode constant, so the optimal
policy is invariant (Ng et al. 1999) while the *advantage* signal
arrives within a trick or two of the decision instead of forty cards
later.  The same Φ is added at the search's trick-boundary playout
cutoff (leaf bonus Φ(leaf) − Φ(root)), so the tree optimizes the same
shaped objective the policy trains on — the search "sees" trick value
directly.

### 4. Search-in-training, fixed
The earlier negative result logged the search's executed move with the
*policy's* log-probability — a mismatched importance ratio, not a fact
about search.  Corrected pattern (AlphaZero): decisions where the
search overrode contribute a **cross-entropy loss toward the search's
visit distribution** and are excluded from the PPO ratio term (value
and belief targets keep flowing).  The policy is pulled toward what
deeper thinking chose, with no off-policy ratio to lie about it.

### 5. Deeper search, visit-distribution targets
Training search depth rises from 4 to 12 moves (~3 rotations,
trick-boundary cutoff still in force), candidate shortlist per the
production rules (top-2 >0%, top-3 if all >5%, pass always measured).
The visit distribution over the shortlist is the distillation target.

### 6. Joint training of all four
All four models sit at one table and improve together — no frozen
champion, no leapfrog chains:

* every seat search-assisted at **1024 simulations / 5 s per move**;
* **batches of 5 000 games**; after each batch every net takes its own
  PPO + distillation update from its own seats' experience;
* **mixed strategies in training**: early-hand moves are *sampled*
  from the visit distribution (temperature 1 above ~26 cards left,
  annealing to argmax below ~20) so each model trains against — and
  as — a mixed strategy.  The ideal deployed model plays mixed where
  Q-values tie or nearly tie, exactly so a strong pure-strategy human
  cannot straightforwardly exploit a deterministic line, while the
  policy-prior tie-break keeps the EV cost against weak opponents
  negligible;
* **diet maintenance**: every batch, each model probes against the
  overall baseline diet (WangBot_v1, PPO v1, humanlike, heuristics) —
  progress against each other must not cost the field; a model that
  regresses ≥1.0 pt/game vs the diet rolls back to its last passing
  snapshot;
* seat rotation and shuffled seating every game, so no model
  specializes to a chair.

### Compute reality
1024 sims / 5 s is ~50× the per-move budget of the chain era.  The
candidate-shortlist rules mean confident positions skip the search, so
the *average* move is far cheaper than the cap; a pilot run calibrates
true throughput before the first 5k batch is scheduled, and batch size
is honest about wall-clock rather than aspirational.

## Addendum: what the Suphx oracle does and does not buy us

Suphx (arXiv:2003.13590) trains an *oracle agent* on perfect
information — including the other players' private tiles — then anneals
it into a normal agent by dropping the perfect features out:

> L(θ) = E[ π_θ(a | x_n(s), δ_t x_o(s)) / π_θ′(...) · A(...) ]

where δ_t is a Bernoulli dropout matrix with P(δ_t = 1) = γ_t, and γ_t
decays from 1 to 0.  The paper is explicit that the obvious shortcut
fails first: *"simple knowledge distillation does not work well"* —
the oracle is too strong for a limited-information student to mimic.

**Tested here for distribution modelling, and it does not pay.**  Our
belief head is already oracle-supervised (``belief_target`` is the true
opponent hands), so the tempting move is to let it drive the search's
determinizations.  Measured on real positions from the recorded human
games (≥16 cards left, 106k card-player pairs), per-card log-loss:

| model                     | log-loss | vs analytic |
|---------------------------|---------:|------------:|
| analytic (hypergeometric + fitted play factors) | 0.6149 | — |
| belief head               | 0.6465 | **−5.1%** |
| geometric blend           | 0.6233 | −1.4% |

It is not a calibration artifact: ranking power is worse too (AUC
0.571 against the analytic map's 0.614), and Platt scaling — which
shrinks its logits by 0.42×, confirming gross overconfidence — still
lands at 0.628.  The head is a useful auxiliary task for representation
learning; it is *not* a density model, and feeding it to the
determinizer would make search worlds worse.  Hand-fitted factors
(workstream 2) stay the distribution channel.

**Where the oracle does pay: the critic.**  The measured bottleneck is
not the policy's information, it is the *baseline's* — the value head
cannot tell early positions apart (corr 0.41) largely because it cannot
see the hands.  A baseline that does not depend on the action leaves
the policy gradient unbiased, so a privileged critic V(s, true hands)
is free variance reduction (CTDE; Suphx applies the same privilege to
the policy instead).  ``build_net(oracle_dim=...)`` adds exactly that:
a second value head that also reads the 156-dim true-hands vector,
used only to compute advantages during training.  The ordinary value
head — the one inference and the search's leaves use — is unchanged
and bit-identical, and ``oracle_dim=0`` (the default) leaves every
existing checkpoint byte-compatible.  The privileged critic can also
serve as a lower-variance regression target for the ordinary head,
which is oracle distillation on a scalar rather than on a policy: the
tractable half of Suphx's idea.

## Order of execution
1. this document; 2. v2_patient (AWR); 3. belief factors;
4. shaping Φ (+ search leaf bonus); 5. distillation fix + depth;
6. joint trainer pilot → calibrate → first 5k batch.
Each lands with tests and its own gate before the next begins.
