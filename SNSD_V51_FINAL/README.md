# SNSD-v5.1 final method

This is the clean, self-contained project for the paper's final method:
**Semantic-Safeguarded Tri-Graph Joint Negotiation (SNSD-v5.1)**.

## Public contract

- The only public solver entry is run.py.
- It always runs learned AlternatingConflictBenefit (Phased).
- The protocol is fixed at 6 agents, 4 macro rounds, 12 steps per round, and
  288 paid local-worker calls.
- Oracle and seven comparison methods live only in SNSD_V51_ABLATIONS.
- Historical experiment_v*.py and run_v15_v*.py files are not present.
- Internal protocol layers are the required runtime closure of the single
  final method, not separately selectable historical versions.

## Layout

~~~text
SNSD_V51_FINAL/
├── run.py
├── train_warmstart.py
├── train_message_warmstart.py
├── train_gnn.py
├── src/snsd_core/
├── src/snsd_v51/
│   └── training/
├── scripts/
│   ├── reproduce_selected_checkpoint.sh
│   ├── train_gnn_8gpu.sh
│   ├── select_checkpoint.py
│   └── check_checkpoint.py
├── tests/
├── data/C3.csv
└── checkpoints/
    ├── checkpoint_message_v4_5_strict.pt
    └── provenance/
~~~

## Run the final solver

~~~bash
python run.py --data data/C3.csv --checkpoint checkpoints/checkpoint_message_v4_5_strict.pt --output runs/c3_seed11 --seed 11 --device cuda:0
~~~

Structural preflight does not require the checkpoint:

~~~bash
python run.py --data data/C3.csv --output unused --preflight-only
~~~

## GNN training

All three training stages are included:

~~~bash
python train_warmstart.py --help
python train_message_warmstart.py --help
python train_gnn.py --help
~~~

The exact validation-selected cell recorded by the recovered formal manifest can
be reconstructed with:

~~~bash
bash scripts/reproduce_selected_checkpoint.sh --output runs/reproduced_selected
~~~

scripts/train_gnn_8gpu.sh remains the configurable eight-GPU exploration
launcher. The authoritative historical sweep and selection receipts are under
checkpoints/provenance/formal_training_root/.

## Formal model contract

- selected trial: warmstart_e2e_lr3e4_eps05_d12_e12_fast
- selection: minimum validation loss; test metrics excluded
- best epoch: 3
- best validation loss: 3.3512077443301678
- parameters: 456,075
- strict load: no missing or unexpected keys
- file SHA-256:
  a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59
- state-dict hash:
  e2fbc9aa02b879ac9cd0bbe55f00838b9668d193933427e822bc00ff910505f8

The formal binary and receipts are present. TRAINING_REPRODUCTION.md documents
one remaining historical limitation: the exact pre-launch source byte snapshot
was not preserved because trainer files were updated while the already-running
training process continued.
