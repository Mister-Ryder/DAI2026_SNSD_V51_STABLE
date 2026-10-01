# v5.1 Paired Semantic-Counterfactual Safeguard: Formal C3 Learned Confirmatory Fresh-Seed Results

The formal matrix contains 64/64 complete non-smoke runs. Every cell uses the same 12-step x 4-macro protocol and 288 paid worker calls.

## Method-level results

| Method | Mean gain +/- SD | Mean final objective | Semantic strict wins | Semantic accepts | Mean CA-C0 |
|---|---:|---:|---:|---:|---:|
| Full | 31,149.0 +/- 5,115.7 | 6,114,409.2 | 2.9 | 2.9 | 12.0 |
| FrozenAnchorComm | 12,244.9 +/- 4,849.4 | 6,095,505.1 | 8.9 | 0.0 | -61.8 |
| SelectionOnly | 27,569.0 +/- 7,530.3 | 6,110,829.2 | 0.0 | 0.0 | 0.0 |
| FrozenAnchorSelectionOnly | 12,437.6 +/- 5,277.3 | 6,095,697.9 | 0.0 | 0.0 | 0.0 |
| AlternatingConflictBenefit | 32,963.8 +/- 5,022.8 | 6,116,224.0 | 3.0 | 3.0 | 24.5 |
| OneShotComm | 27,370.0 +/- 5,702.9 | 6,110,630.2 | 0.9 | 0.9 | 15.2 |
| ShuffledComm | 32,725.2 +/- 6,467.9 | 6,115,985.5 | 1.2 | 1.2 | 45.9 |
| NoPeerFinalShield | 12,437.6 +/- 5,277.3 | 6,095,697.9 | 0.0 | 0.0 | 0.0 |

## Paired factorial effects

| Effect | Mean +/- SD | Positive / Tie / Negative seeds | Min | Max |
|---|---:|---:|---:|---:|
| A \| B | 3,580.0 +/- 3,858.2 | 7/0/1 | -4,175.0 | 9,009.0 |
| A \| not B | -192.8 +/- 5,395.7 | 4/0/4 | -7,979.0 | 9,485.0 |
| B \| A | 18,904.1 +/- 4,094.6 | 8/0/0 | 12,939.0 | 24,707.0 |
| B \| not A | 15,131.4 +/- 6,837.8 | 8/0/0 | 1,897.0 | 26,599.0 |
| A x B | 3,772.8 +/- 8,173.8 | 7/0/1 | -13,660.0 | 12,718.0 |
| Full - A0B0 | 18,711.4 +/- 4,196.3 | 8/0/0 | 10,906.0 | 24,453.0 |
| Full - OneShot | 3,779.0 +/- 6,399.7 | 6/0/2 | -4,739.0 | 13,341.0 |
| Full - Shuffled | -1,576.2 +/- 3,948.5 | 3/0/5 | -8,144.0 | 4,517.0 |
| Full - NoPeer | 18,711.4 +/- 4,196.3 | 8/0/0 | 10,906.0 | 24,453.0 |

## Formal safeguards

- Complete matrix: `True`.
- Worker-budget violations: `0`.
- Unaccepted-CA domain-pollution violations: `0`.
- Additional paid worker-call violations: `0`.
- Semantic-schedule violations from `events.jsonl`: `0`.
