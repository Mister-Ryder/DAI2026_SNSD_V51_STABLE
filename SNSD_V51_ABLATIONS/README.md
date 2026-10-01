# SNSD-v5.1 ablations and comparisons

This sibling project contains everything intentionally excluded from the clean
final-method CLI:

- eight formal method cells;
- learned original seeds and fresh confirmatory seeds;
- Oracle mechanism diagnosis;
- strict matrix summarization and paired analysis;
- the 192 archived formal runs and their audit reports.

It imports the implementation from `../SNSD_V51_FINAL`; it does not maintain a
second copy of the algorithm. This guarantees that ablations toggle the same
core used by the final method.

Run one cell:

```bash
python run_ablation.py \
  --data ../SNSD_V51_FINAL/data/C3.csv \
  --checkpoint ../SNSD_V51_FINAL/checkpoints/checkpoint_message_v4_5_strict.pt \
  --output runs/seed11/SelectionOnly \
  --method SelectionOnly \
  --seed 11 \
  --device cuda:0
```

The formal method set is:

```text
Full
FrozenAnchorComm
SelectionOnly
FrozenAnchorSelectionOnly
AlternatingConflictBenefit
OneShotComm
ShuffledComm
NoPeerFinalShield
```

The launchers in `scripts/` reproduce the original 64-cell learned,
fresh-seed, and Oracle matrix shapes. Archived evidence is under
`results/formal/`. Oracle remains a mechanism diagnostic, not an upper bound.
