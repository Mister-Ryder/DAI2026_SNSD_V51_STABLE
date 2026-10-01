# v5.0--v5.1 formal paired analysis

All deltas are computed seed by seed. `+/-` is sample SD; the triplet is positive/tie/negative seeds.

## Learned: v5.1 minus v5.0

| Method | Paired gain delta | Sign p | Sign-flip mean p |
|---|---:|---:|---:|
| AlternatingConflictBenefit | 3859.9 +/- 6564.2 (5/0/3) | 0.7266 | 0.1484 |
| FrozenAnchorComm | -727.1 +/- 4954.0 (3/0/5) | 0.7266 | 0.6797 |
| FrozenAnchorSelectionOnly | -6.9 +/- 19.4 (0/7/1) | 1.0000 | 1.0000 |
| Full | 1607.4 +/- 6547.4 (5/0/3) | 0.7266 | 0.5156 |
| NoPeerFinalShield | 11023.5 +/- 4991.7 (8/0/0) | 0.0078 | 0.0078 |
| OneShotComm | 1430.9 +/- 5899.0 (5/0/3) | 0.7266 | 0.4844 |
| SelectionOnly | -45.2 +/- 2334.1 (4/3/1) | 0.3750 | 1.0000 |
| ShuffledComm | 2004.9 +/- 4260.2 (5/0/3) | 0.7266 | 0.2344 |

## Learned v5.1: factorial effects

| Effect | Paired gain effect | Sign p | Sign-flip mean p |
|---|---:|---:|---:|
| A\_given\_B | 2497.2 +/- 7475.1 (5/0/3) | 0.7266 | 0.3438 |
| A\_without\_B | -1309.8 +/- 5894.1 (2/0/6) | 0.2891 | 0.5391 |
| B\_given\_A | 21524.0 +/- 6970.7 (8/0/0) | 0.0078 | 0.0078 |
| B\_without\_A | 17717.0 +/- 2251.6 (8/0/0) | 0.0078 | 0.0078 |
| A\_x\_B\_interaction | 3807.0 +/- 7506.6 (5/0/3) | 0.7266 | 0.2266 |

## Change in learned factorial effects: v5.1 minus v5.0

| Effect | Paired effect delta | Sign p | Sign-flip mean p |
|---|---:|---:|---:|
| A\_given\_B | 1652.6 +/- 7603.4 (5/0/3) | 0.7266 | 0.5547 |
| A\_without\_B | -720.2 +/- 4952.8 (3/0/5) | 0.7266 | 0.6875 |
| A\_x\_B\_interaction | 2372.9 +/- 10434.5 (4/0/4) | 1.0000 | 0.5234 |
| B\_given\_A | 2334.5 +/- 9054.4 (5/0/3) | 0.7266 | 0.4922 |
| B\_without\_A | -38.4 +/- 2334.4 (5/2/1) | 0.2188 | 1.0000 |

## Learned v5.1: paired protocol contrasts

| Contrast | Paired gain delta | Sign p | Sign-flip mean p |
|---|---:|---:|---:|
| Alternating\_minus\_Full | 2175.1 +/- 4936.8 (5/0/3) | 0.7266 | 0.2734 |
| Alternating\_minus\_OneShot | 4027.6 +/- 5607.8 (6/0/2) | 0.2891 | 0.0859 |
| Alternating\_minus\_Shuffled | 2911.2 +/- 5194.7 (7/0/1) | 0.0703 | 0.1328 |
| Full\_minus\_OneShot | 1852.5 +/- 5733.5 (5/0/3) | 0.7266 | 0.4062 |
| Full\_minus\_Shuffled | 736.1 +/- 6455.2 (3/0/5) | 0.7266 | 0.8594 |
| Full\_minus\_A0B0 | 20214.2 +/- 7957.5 (8/0/0) | 0.0078 | 0.0078 |
| Full\_minus\_NoPeer | 20207.4 +/- 7961.7 (8/0/0) | 0.0078 | 0.0078 |

## Analytic mechanism diagnostic: v5.1 minus v5.0

| Method | Paired gain delta | Sign p | Sign-flip mean p |
|---|---:|---:|---:|
| AlternatingConflictBenefit | 2300.9 +/- 5392.0 (5/0/3) | 0.7266 | 0.2656 |
| FrozenAnchorComm | 1795.1 +/- 2743.3 (4/2/2) | 0.6875 | 0.1250 |
| FrozenAnchorSelectionOnly | 867.8 +/- 2454.4 (1/7/0) | 1.0000 | 1.0000 |
| Full | 3558.8 +/- 2247.6 (7/0/1) | 0.0703 | 0.0156 |
| NoPeerFinalShield | 11074.1 +/- 5499.2 (8/0/0) | 0.0078 | 0.0078 |
| OneShotComm | 2890.8 +/- 5819.8 (6/0/2) | 0.2891 | 0.2188 |
| SelectionOnly | 1405.8 +/- 4062.5 (5/1/2) | 0.4531 | 0.4219 |
| ShuffledComm | -344.2 +/- 4854.3 (2/1/5) | 0.4531 | 0.8125 |

## Fresh-seed confirmation specified before execution

| Method | Gain mean +/- SD |
|---|---:|
| AlternatingConflictBenefit | 32963.8 +/- 5022.8 |
| FrozenAnchorComm | 12244.9 +/- 4849.4 |
| FrozenAnchorSelectionOnly | 12437.6 +/- 5277.3 |
| Full | 31149.0 +/- 5115.7 |
| NoPeerFinalShield | 12437.6 +/- 5277.3 |
| OneShotComm | 27370.0 +/- 5702.9 |
| SelectionOnly | 27569.0 +/- 7530.3 |
| ShuffledComm | 32725.2 +/- 6467.9 |

| Confirmatory contrast | Paired gain delta | Sign p | Sign-flip mean p |
|---|---:|---:|---:|
| Alternating\_minus\_Full | 1814.8 +/- 4908.2 (5/0/3) | 0.7266 | 0.3672 |
| Alternating\_minus\_SelectionOnly | 5394.8 +/- 5278.1 (7/0/1) | 0.0703 | 0.0234 |
| Alternating\_minus\_OneShot | 5593.8 +/- 3918.3 (8/0/0) | 0.0078 | 0.0078 |
| Alternating\_minus\_Shuffled | 238.5 +/- 6402.8 (5/0/3) | 0.7266 | 0.9219 |
| Full\_minus\_OneShot | 3779.0 +/- 6399.7 (6/0/2) | 0.2891 | 0.1406 |
| Full\_minus\_Shuffled | -1576.2 +/- 3948.5 (3/0/5) | 0.7266 | 0.3125 |

## Descriptive pooled 16-seed view

The pooled view combines the original and the fresh seed set specified before its execution. It is descriptive; the fresh set remains the confirmatory decision source.

| Method | Gain mean +/- SD |
|---|---:|
| AlternatingConflictBenefit | 35884.6 +/- 6026.4 |
| FrozenAnchorComm | 13675.6 +/- 5757.6 |
| FrozenAnchorSelectionOnly | 14426.9 +/- 5369.1 |
| Full | 33889.7 +/- 7054.4 |
| NoPeerFinalShield | 14430.3 +/- 5368.4 |
| OneShotComm | 31073.9 +/- 7722.3 |
| SelectionOnly | 30851.1 +/- 7378.8 |
| ShuffledComm | 34309.8 +/- 7223.9 |

| Pooled contrast | Paired gain delta | Sign p | Sign-flip mean p |
|---|---:|---:|---:|
| Alternating\_minus\_Full | 1994.9 +/- 4759.3 (10/0/6) | 0.4545 | 0.1171 |
| Alternating\_minus\_SelectionOnly | 5033.6 +/- 5072.7 (14/0/2) | 0.0042 | 0.0017 |
| Alternating\_minus\_OneShot | 4810.7 +/- 4742.8 (14/0/2) | 0.0042 | 0.0019 |
| Alternating\_minus\_Shuffled | 1574.9 +/- 5799.1 (12/0/4) | 0.0768 | 0.2962 |
| Full\_minus\_OneShot | 2815.8 +/- 5953.4 (11/0/5) | 0.2101 | 0.0790 |
| Full\_minus\_Shuffled | -420.1 +/- 5305.4 (6/0/10) | 0.4545 | 0.7757 |

## Interpretation guardrails

- The original eight seeds are appropriate for paired ablations but are too small for strong distributional claims.
- Selecting the highest method from the original matrix is exploratory; the separately executed fresh seed set is the confirmatory check.
- The analytic oracle is a mechanism diagnostic scaffold, not an upper bound and not directly comparable as a learned policy ranking.
- Mean advantages with mixed seed signs must be reported together with the sign count and variability.
