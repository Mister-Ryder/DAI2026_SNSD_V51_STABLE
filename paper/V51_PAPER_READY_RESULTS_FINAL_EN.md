# v5.1 Final Paper-ready Results (C3)

## Evidence hierarchy and protocol

The primary evidence is the **learned-policy evaluation** of v5.1.  The model
checkpoint was selected on validation data and frozen before formal testing.
We first evaluated the registered seed set
`{11,22,33,44,55,66,77,88}` and then ran a fresh confirmatory set
`{101,202,303,404,505,606,707,808}` that was specified before its execution.
The fresh set is the
confirmatory decision source; the pooled 16-seed analysis is descriptive.

Every matrix contains eight configurations and 64 complete C3 non-smoke runs.
Every run uses six partition-level agents, 12 negotiation steps, four macro
rounds, and exactly 288 paid worker calls.  Factorial cells use same-state
paired candidate evaluations.  The learned original and confirmatory matrices
loaded the same 456,075-parameter checkpoint (SHA-256
`a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59`) in
64/64 runs, with `oracle_mode=false` and strict loading.

The final audit found **zero violations** in every matrix: no missing,
unexpected, or duplicate cells; no budget or additional-worker-call
violations; no semantic-schedule violations; and no pollution from unaccepted
semantic candidates.  All same-state pairing checks passed.  The separate
analytic Oracle matrix also completed 64/64 audited runs under the same query
budget, but it is used only as a mechanism diagnostic.

## Learned-policy performance

The proposed **Phased/Alternating** configuration is
`AlternatingConflictBenefit`, which allocates 12 decisions to conflict-only
messages and 12 to benefit-only messages over the four macro rounds.  It has
the highest mean gain in the original, fresh, and pooled evaluations.

| Configuration | Original 8 seeds: gain ± SD | Fresh 8 seeds: gain ± SD | Pooled 16 seeds: gain ± SD |
|---|---:|---:|---:|
| **Phased/Alternating** | **38,805.5 ± 5,752.8** | **32,963.8 ± 5,022.8** | **35,884.6 ± 6,026.4** |
| Full simultaneous messaging | 36,630.4 ± 7,956.1 | 31,149.0 ± 5,115.7 | 33,889.7 ± 7,054.4 |
| Shuffled semantics | 35,894.3 ± 8,015.7 | 32,725.3 ± 6,467.9 | 34,309.8 ± 7,223.9 |
| One-shot communication | 34,777.9 ± 7,994.1 | 27,370.0 ± 5,702.9 | 31,073.9 ± 7,722.3 |
| SelectionOnly (A0B1) | 34,133.1 ± 5,945.2 | 27,569.0 ± 7,530.3 | 30,851.1 ± 7,378.8 |
| FrozenAnchorComm (A1B0) | 15,106.4 ± 6,545.1 | 12,244.9 ± 4,849.4 | 13,675.6 ± 5,757.6 |
| FrozenAnchorSelectionOnly (A0B0) | 16,416.1 ± 4,987.8 | 12,437.6 ± 5,277.3 | 14,426.9 ± 5,369.1 |
| NoPeerFinalShield | 16,423.0 ± 4,983.1 | 12,437.6 ± 5,277.3 | 14,430.3 ± 5,368.4 |

For Phased/Alternating, the mean final objective is **6,121,118.6** on the
original seeds and **6,116,224.0** on the fresh seeds.  Its status as the
best-performing configuration refers to mean performance: it is not a claim
of seed-wise dominance.

## Paired factorial and protocol evidence

The strongest result is factor B, the structured coalitional
delete--replace--subgraph-refill operator.  Factor A denotes learned semantic
conflict--benefit messaging.  The table reports paired mean differences,
sample SD, and positive/negative seed counts.  `p_sign` is the exact two-sided
sign-test value and `p_flip` is the exhaustive two-sided sign-flip test of the
paired mean.

| Learned contrast | Fresh confirmatory set | Descriptive pooled 16-seed set | Evidence level |
|---|---:|---:|---|
| B given A | +18,904.1 ± 4,094.6 (8/0), `p_sign=p_flip=.0078` | **+20,214.1 ± 5,685.9 (16/0), `p_sign=p_flip=3.05e-5`** | Strong, seed-consistent main effect |
| B without A | +15,131.4 ± 6,837.8 (8/0), `p_sign=p_flip=.0078` | **+16,424.2 ± 5,095.9 (16/0), `p_sign=p_flip=3.05e-5`** | Strong, seed-consistent main effect |
| A given B | +3,580.0 ± 3,858.2 (7/1), `p_sign=.0703`, `p_flip=.0391` | +3,038.6 ± 5,773.6 (12/4), `p_sign=.0768`, `p_flip=.0531` | Marginal complementary evidence |
| A without B | -192.8 ± 5,395.7 (4/4) | -751.3 ± 5,489.2 (6/10) | No independent A effect |
| Phased − SelectionOnly | +5,394.8 ± 5,278.1 (7/1), `p_sign=.0703`, `p_flip=.0234` | **+5,033.6 ± 5,072.7 (14/2), `p_sign=.0042`, `p_flip=.0017`** | Reproduced advantage; strongest protocol contrast |
| Phased − OneShot | **+5,593.8 ± 3,918.3 (8/0), `p_sign=p_flip=.0078`** | **+4,810.7 ± 4,742.8 (14/2), `p_sign=.0042`, `p_flip=.0019`** | Strong evidence for sustained phased coordination |
| Phased − Full | +1,814.8 ± 4,908.2 (5/3) | +1,994.9 ± 4,759.3 (**10/6**) | Higher mean, not universal superiority |
| Phased − Shuffled | +238.5 ± 6,402.8 (5/3) | +1,574.9 ± 5,799.1 (**12/4**) | Directionally positive, not conclusive |

Thus, the defensible claim is that structured joint action B supplies the
large and stable performance gain.  Learned semantics A add value when coupled
to B, but the evidence is marginal and does not support an independent or
seed-universal semantic effect.  Phased communication is clearly stronger than
SelectionOnly and OneShot, whereas its smaller gaps over Full and Shuffled
must be reported with the mixed paired signs: **10/16** and **12/16**,
respectively.

## Analytic Oracle mechanism diagnostic — reported separately

The analytic Oracle is a hand-specified mechanism diagnostic.  It is **not an
upper bound**, is not used by the learned runs, and must not be interpreted as
a learned-policy ranking.

| Oracle diagnostic quantity, original 8 seeds | Mean ± SD | Positive/negative seeds |
|---|---:|---:|
| Phased/Alternating gain | 34,202.0 ± 5,874.2 | -- |
| Full gain | 35,589.1 ± 4,433.3 | -- |
| OneShot gain | 36,860.0 ± 3,965.6 | -- |
| B given A | +17,428.5 ± 5,358.2 | 8/0 |
| B without A | +17,994.8 ± 4,135.2 | 8/0 |
| A given B | +2,307.0 ± 6,822.9 | 4/4 |

The Oracle diagnostic independently reproduces the seed-consistent B effect,
which supports the proposed mechanism interpretation.  Its configuration
ranking differs from the learned ranking; this is expected because the Oracle
is an analytic decision scaffold rather than a globally optimal solver.

## Paper-ready Results paragraph

> We evaluated the frozen validation-selected policy on C3 using an original
> eight-seed matrix and a fresh eight-seed confirmatory matrix whose seeds
> were specified before execution.  Each configuration received 288 paid neighborhood-
> search calls, and all 128 learned runs passed the formal budget, schedule,
> same-state pairing, checkpoint, and candidate-isolation audits with zero
> violations.  The phased conflict--benefit configuration achieved the
> highest mean gain on both the original (38,805.5) and fresh confirmatory
> seeds (32,963.8), yielding 35,884.6 ± 6,026.4 over the descriptive pooled
> 16-seed set.  The structured coalitional exchange--refill operator produced
> the most stable paired effects: +20,214.1 with semantic messaging and
> +16,424.2 without it, both positive on all 16 seeds.  Phased communication
> exceeded SelectionOnly and OneShot by 5,033.6 and 4,810.7 on average,
> respectively, with 14/16 positive paired differences in both comparisons.
> Its learned semantic increment under the coalition operator was +3,038.6
> on average (12/16), providing marginal rather than universal evidence for
> message semantics.

## Paper-ready Discussion paragraph

> The results locate the main contribution in how communication changes the
> feasible joint-action domain.  Partitioning inevitably removes cross-agent
> information; structured boundary exchange restores part of that information
> by coordinating deletions, replacements, and local refills before exact
> global validation.  The 16/16 signs for both B contrasts show that this
> operator remains effective with and without learned semantics.  Phasing
> conflict and benefit messages is the best-performing configuration by mean
> and is consistently stronger than one-shot communication and selection
> without semantic messages.  However, Phased exceeds simultaneous Full on
> only 10/16 seeds and Shuffled on 12/16 seeds.  We therefore do not claim
> universal protocol superiority; instead, we conclude that structured joint
> action is robust, while the learned ordering and timing of semantic messages
> remain a promising but less certain source of additional gain.

## Claim boundary

### Supported wording

- The learned v5.1 system was evaluated with a frozen validation-selected
  checkpoint on a separately executed fresh seed set specified before launch.
- Phased/Alternating is the best-performing registered configuration by mean
  on the original, fresh, and pooled evaluations.
- Structured coalition factor B has a large paired benefit under both A cells
  and is positive on all 16 seeds.
- Phased communication has strong paired advantages over SelectionOnly and
  OneShot; learned A provides marginal complementary evidence under B.
- All formal matrices completed with equal query budgets and zero registered
  audit violations.

### Unsupported wording

- Do not claim that Phased/Alternating beats Full on every seed: the count is
  10/16.  Do not claim universal superiority to Shuffled: the count is 12/16.
- Do not call A independently effective or statistically decisive; `A|B` is
  marginal and `A|not-B` is negative on average.
- Do not describe the analytic Oracle as an upper bound, learned result, or
  runtime information source.
- Do not claim cross-instance, cross-dataset, or cross-scenario generalization
  from the single C3 instance, and do not claim global optimality.
- Do not treat the pooled set as a replacement for the separately executed
  fresh confirmatory analysis; pooled results are descriptive.
