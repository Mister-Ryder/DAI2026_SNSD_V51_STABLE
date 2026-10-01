# v5.1 Paired Semantic-Counterfactual Safeguard: Formal C3 Oracle Results

The formal matrix contains 64/64 complete non-smoke runs. Every cell uses the same 12-step x 4-macro protocol and 288 paid worker calls.

## Method-level results

| Method | Mean gain +/- SD | Mean final objective | Semantic strict wins | Semantic accepts | Mean CA-C0 |
|---|---:|---:|---:|---:|---:|
| Full | 35,589.1 +/- 4,433.3 | 6,117,902.2 | 2.6 | 2.6 | -72.2 |
| FrozenAnchorComm | 18,160.6 +/- 4,853.1 | 6,100,473.8 | 12.0 | 0.0 | 9.1 |
| SelectionOnly | 33,282.1 +/- 6,951.2 | 6,115,595.2 | 0.0 | 0.0 | 0.0 |
| FrozenAnchorSelectionOnly | 15,287.4 +/- 6,039.9 | 6,097,600.5 | 0.0 | 0.0 | 0.0 |
| AlternatingConflictBenefit | 34,202.0 +/- 5,874.2 | 6,116,515.1 | 3.4 | 3.4 | 230.6 |
| OneShotComm | 36,860.0 +/- 3,965.6 | 6,119,173.1 | 1.6 | 1.6 | 54.7 |
| ShuffledComm | 32,971.0 +/- 5,964.4 | 6,115,284.1 | 1.1 | 1.1 | 1.1 |
| NoPeerFinalShield | 14,419.6 +/- 6,674.3 | 6,096,732.8 | 0.0 | 0.0 | 0.0 |

## Paired factorial effects

| Effect | Mean +/- SD | Positive / Tie / Negative seeds | Min | Max |
|---|---:|---:|---:|---:|
| A \| B | 2,307.0 +/- 6,822.9 | 4/0/4 | -5,297.0 | 12,331.0 |
| A \| not B | 2,873.2 +/- 5,072.9 | 7/0/1 | -8,482.0 | 7,456.0 |
| B \| A | 17,428.5 +/- 5,358.2 | 8/0/0 | 10,400.0 | 26,292.0 |
| B \| not A | 17,994.8 +/- 4,135.2 | 8/0/0 | 9,897.0 | 22,016.0 |
| A x B | -566.2 +/- 6,636.5 | 4/0/4 | -9,299.0 | 8,487.0 |
| Full - A0B0 | 20,301.8 +/- 4,594.0 | 8/0/0 | 13,421.0 | 27,382.0 |
| Full - OneShot | -1,270.9 +/- 4,781.5 | 5/0/3 | -9,669.0 | 4,164.0 |
| Full - Shuffled | 2,618.1 +/- 8,266.2 | 6/0/2 | -11,131.0 | 11,675.0 |
| Full - NoPeer | 21,169.5 +/- 4,615.6 | 8/0/0 | 13,421.0 | 27,382.0 |

## Formal safeguards

- Complete matrix: `True`.
- Worker-budget violations: `0`.
- Unaccepted-CA domain-pollution violations: `0`.
- Additional paid worker-call violations: `0`.
- Semantic-schedule violations from `events.jsonl`: `0`.
