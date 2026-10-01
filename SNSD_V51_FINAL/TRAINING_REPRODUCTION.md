# GNN training reproduction and provenance

## Recovered state

The complete model binary and training provenance were recovered from the
reopened server on 2026-08-03. The local project now contains:

1. the validation-selected strict rolling checkpoint;
2. its complete selected-run dataset and training receipts;
3. the complete v3 warm-start trial;
4. all eight strict depth-12 warm-up sources;
5. the eligible-candidate sweep and validation-only selector receipts;
6. the Original Learned, Fresh Learned, and Oracle launch manifests;
7. a snapshot of the server cplexpy37 software and GPU environment.

See checkpoints/provenance/README.md for the exact inventory.

## Selected model contract

The selector evaluated eight eligible main candidates and minimized validation
loss without using test metrics.

- selected trial: warmstart_e2e_lr3e4_eps05_d12_e12_fast
- best epoch: 3
- best validation loss: 3.3512077443301678
- model parameters: 456,075
- trainable parameters: 398,411
- encoder-trainable parameters: 292,676
- training steps: 12
- folds: 2
- epochs: 12
- patience: 4
- learning rate: 3e-4
- weight decay: 1e-5
- gradient clip: 5.0
- exploration epsilon: 0.05
- trainer seed: 20260803
- Real / Zero / Shuffled weights: 1.0 / 0.25 / 0.10
- zero target semantics: intrinsic
- heads only: false
- include shuffled: false
- initial checkpoint: formal_warmstart_message_v3/checkpoint_message_v3.pt

Seed split:

- train: 44, 55, 66, 77, 88
- validation and model selection: 33
- held-out test reporting: 11, 22
- split unit: worker/trajectory seed on one C3 instance
- seed overlap: false

This split tests stochastic robustness on C3; it is not cross-instance
generalization.

## Exact selected-cell launcher

From SNSD_V51_FINAL:

~~~bash
bash scripts/reproduce_selected_checkpoint.sh --output runs/reproduced_selected
~~~

The launcher consumes the recovered eight warm-up manifests and v3 checkpoint
and passes every frozen trainer parameter explicitly. Use --plan-only to print
the command without training.

The configurable train_gnn_8gpu.sh is retained for new exploration. The
historical source of truth for the paper selection is:

- checkpoints/provenance/formal_training_root/fast_sweep_manifest.tsv
- checkpoints/provenance/formal_training_root/selected_fast_main.json
- checkpoints/provenance/formal_selected_run/model_manifest_message_v4_5_strict.json
- checkpoints/provenance/formal_selected_run/training_history_message_v4_5_strict.jsonl

## Binary identities

Formal selected checkpoint:

- file SHA-256:
  a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59
- state-dict hash:
  e2fbc9aa02b879ac9cd0bbe55f00838b9668d193933427e822bc00ff910505f8

v3 warm-start checkpoint:

- file SHA-256:
  6e9c72018d5341e04326076c22a26debc8a070d8e5ef5d29ac136cf17ac58ea8
- state-dict hash recorded by its manifest:
  46dea02bdc1fb656dcde6beaef52174cb49d6952140715f351e51a4bebbd34a6

## Remaining historical limitation

The selected training process started before the server trainer files were
updated for a later, separately stored late-stage experiment. Python had already
loaded the earlier source into the running process, but that exact pre-launch
source byte snapshot was not retained. The current training source is the
compatible later implementation and the formal selected manifest contains no
late-stage option.

Therefore:

- inference reproduction is complete and strictly verified;
- model, data, labels, warm-start, configuration, history, selection, and
  environment provenance are present;
- rerunning the frozen contract is supported;
- bitwise recreation of the original training process must not be claimed.

The checkpoint remains a strict rolling Real/Zero message model. It was not
trained end-to-end on the later SNSD-v5.1 coalition C_A-C_0 objective.
