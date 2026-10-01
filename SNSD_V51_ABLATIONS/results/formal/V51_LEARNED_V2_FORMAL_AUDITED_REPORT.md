# v5.1 Paired Semantic-Counterfactual Safeguard: Formal C3 Oracle Results

The formal matrix contains 64/64 complete non-smoke runs. Every cell uses the same 12-step x 4-macro protocol and 288 paid worker calls.

## Method-level results

| Method | Mean gain +/- SD | Mean final objective | Semantic strict wins | Semantic accepts | Mean CA-C0 |
|---|---:|---:|---:|---:|---:|
| Full | 36,630.4 +/- 7,956.1 | 6,118,943.5 | 4.5 | 4.5 | 102.1 |
| FrozenAnchorComm | 15,106.4 +/- 6,545.1 | 6,097,419.5 | 7.4 | 0.0 | -96.9 |
| SelectionOnly | 34,133.1 +/- 5,945.2 | 6,116,446.2 | 0.0 | 0.0 | 0.0 |
| FrozenAnchorSelectionOnly | 16,416.1 +/- 4,987.8 | 6,098,729.2 | 0.0 | 0.0 | 0.0 |
| AlternatingConflictBenefit | 38,805.5 +/- 5,752.8 | 6,121,118.6 | 3.1 | 3.1 | -63.5 |
| OneShotComm | 34,777.9 +/- 7,994.1 | 6,117,091.0 | 0.4 | 0.4 | -47.9 |
| ShuffledComm | 35,894.2 +/- 8,015.7 | 6,118,207.4 | 0.8 | 0.8 | -3.4 |
| NoPeerFinalShield | 16,423.0 +/- 4,983.1 | 6,098,736.1 | 0.0 | 0.0 | 0.0 |

## Paired factorial effects

| Effect | Mean +/- SD | Positive / Tie / Negative seeds | Min | Max |
|---|---:|---:|---:|---:|
| A \| B | 2,497.2 +/- 7,475.1 | 5/0/3 | -8,938.0 | 14,012.0 |
| A \| not B | -1,309.8 +/- 5,894.1 | 2/0/6 | -12,734.0 | 8,577.0 |
| B \| A | 21,524.0 +/- 6,970.7 | 8/0/0 | 14,924.0 | 32,644.0 |
| B \| not A | 17,717.0 +/- 2,251.6 | 8/0/0 | 14,095.0 | 21,460.0 |
| A x B | 3,807.0 +/- 7,506.6 | 5/0/3 | -6,151.0 | 14,732.0 |
| Full - A0B0 | 20,214.2 +/- 7,957.5 | 8/0/0 | 7,672.0 | 32,602.0 |
| Full - OneShot | 1,852.5 +/- 5,733.5 | 5/0/3 | -8,547.0 | 11,490.0 |
| Full - Shuffled | 736.1 +/- 6,455.2 | 3/0/5 | -4,613.0 | 14,953.0 |
| Full - NoPeer | 20,207.4 +/- 7,961.7 | 8/0/0 | 7,672.0 | 32,602.0 |

## Formal safeguards

- Complete matrix: `True`.
- Worker-budget violations: `0`.
- Unaccepted-CA domain-pollution violations: `0`.
- Additional paid worker-call violations: `0`.
- Semantic-schedule violations from `events.jsonl`: `0`.
