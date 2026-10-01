#!/usr/bin/env bash
# Launch a configurable strict depth-12 exploration sweep, one trial per GPU.
# Use reproduce_selected_checkpoint.sh for the exact paper-selected cell.
#
# Model selection is validation-only (seed 33).  Test seeds 11 and 22 are
# evaluated by the trainer for the final locked report but never enter the
# sweep ranking implemented by summarize_train_v45_strict.py.
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="$PROJECT_ROOT/data/C3.csv"
WARMUP_ROOT=""
OUTPUT_ROOT=""
WARMSTART_CHECKPOINT="$PROJECT_ROOT/checkpoints/provenance/formal_warmstart_message_v3/checkpoint_message_v3.pt"
EPOCHS=12
PATIENCE=4
TRAINER_SEED=20260803
PLAN_ONLY=0

DEPTH=12
WARMUP_QUARTETS=7
FOLDS=2
TRAIN_SEEDS="44,55,66,77,88"
VALIDATION_SEEDS="33"
TEST_SEEDS="11,22"
RUN_DIR_SEEDS=(11 22 33 44 55 66 77 88)

usage() {
  echo "Usage: $0 --output-root DIR [--warmup-root DIR] [--warmstart-checkpoint FILE] [--data C3.csv] [--epochs N] [--patience N] [--trainer-seed N] [--python FILE] [--project-root DIR] [--plan-only]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --warmup-root) WARMUP_ROOT="$2"; shift 2 ;;
    --warmstart-checkpoint) WARMSTART_CHECKPOINT="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --trainer-seed) TRAINER_SEED="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then usage >&2; exit 2; fi
if [[ -z "$WARMUP_ROOT" ]]; then
  WARMUP_ROOT="$PROJECT_ROOT/checkpoints/provenance/warmup_teacher_strict"
fi
TRAIN_SCRIPT="$PROJECT_ROOT/train_gnn.py"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Strict trainer CLI not found: $TRAIN_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$DATA" ]]; then
  echo "Data file not found: $DATA" >&2
  exit 2
fi
if [[ ! -f "$WARMSTART_CHECKPOINT" ]]; then
  echo "Warm-start checkpoint not found: $WARMSTART_CHECKPOINT" >&2
  exit 2
fi
if [[ "$EPOCHS" -le 0 || "$PATIENCE" -le 0 ]]; then
  echo "--epochs and --patience must be positive" >&2
  exit 2
fi
for warmup_seed in "${RUN_DIR_SEEDS[@]}"; do
  warmup_dir="$WARMUP_ROOT/strict_depth${DEPTH}_q${WARMUP_QUARTETS}_seed${warmup_seed}"
  if [[ ! -f "$warmup_dir/warmup_manifest.json" ]]; then
    echo "Missing strict warm-up manifest: $warmup_dir/warmup_manifest.json" >&2
    exit 2
  fi
done
if [[ $PLAN_ONLY -eq 0 ]]; then
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  if [[ "$GPU_COUNT" -lt 8 ]]; then
    echo "Eight-GPU training sweep requested but only $GPU_COUNT GPUs are visible" >&2
    exit 2
  fi
fi

# Six of eight cells are end-to-end/intrinsic.  The last two are explicitly
# labelled sensitivity/ablation cells rather than candidate main models.
TRIALS=(
  scratch_e2e_lr3e4_eps05
  warmstart_e2e_lr3e4_eps05
  scratch_e2e_lr1e4_eps05
  scratch_e2e_lr5e4_eps05
  scratch_e2e_lr3e4_eps10
  scratch_e2e_lr3e4_eps05_shuffled
  warmstart_heads_lr3e4_eps05
  scratch_e2e_lr3e4_eps05_zero_global
)
INIT_MODES=(scratch warmstart scratch scratch scratch scratch warmstart scratch)
LEARNING_RATES=(3e-4 3e-4 1e-4 5e-4 3e-4 3e-4 3e-4 3e-4)
EPSILONS=(0.05 0.05 0.05 0.05 0.10 0.05 0.05 0.05)
INCLUDE_SHUFFLED=(0 0 0 0 0 1 0 0)
HEADS_ONLY=(0 0 0 0 0 0 1 0)
ZERO_SEMANTICS=(intrinsic intrinsic intrinsic intrinsic intrinsic intrinsic intrinsic global)

mkdir -p "$OUTPUT_ROOT/logs"
MANIFEST="$OUTPUT_ROOT/launch_manifest.tsv"
COMPLETION_MANIFEST="$OUTPUT_ROOT/completion_manifest.tsv"
if [[ -e "$MANIFEST" || -e "$COMPLETION_MANIFEST" ]]; then
  echo "Refusing to overwrite an existing sweep manifest under: $OUTPUT_ROOT" >&2
  exit 2
fi
printf "trial\tgpu\tinit_mode\tlearning_rate\texploration_epsilon\tinclude_shuffled\theads_only\tzero_target_semantics\tepochs\tpatience\ttrainer_seed\tnegotiation_steps\twarmup_quartets\ttrain_seeds\tvalidation_seeds\ttest_seeds\tpid\trun_dir\tlog\tcheckpoint\tinitial_checkpoint\n" > "$MANIFEST"
printf "trial\tpid\texit_code\tstatus\n" > "$COMPLETION_MANIFEST"

PIDS=()
for index in "${!TRIALS[@]}"; do
  trial="${TRIALS[$index]}"
  gpu="$index"
  init_mode="${INIT_MODES[$index]}"
  learning_rate="${LEARNING_RATES[$index]}"
  epsilon="${EPSILONS[$index]}"
  include_shuffled="${INCLUDE_SHUFFLED[$index]}"
  heads_only="${HEADS_ONLY[$index]}"
  zero_semantics="${ZERO_SEMANTICS[$index]}"
  run_dir="$OUTPUT_ROOT/$trial"
  log="$OUTPUT_ROOT/logs/$trial.log"
  checkpoint="$run_dir/checkpoint_message_v4_5_strict.pt"
  initial_checkpoint=""
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse trial directory: $run_dir" >&2
    exit 2
  fi

  args=("$TRAIN_SCRIPT")
  # Every job receives all eight run directories explicitly.  Resolution by a
  # mutable runs root is deliberately disabled for this paper sweep.
  for warmup_seed in "${RUN_DIR_SEEDS[@]}"; do
    args+=(
      --run-dir
      "$WARMUP_ROOT/strict_depth${DEPTH}_q${WARMUP_QUARTETS}_seed${warmup_seed}"
    )
  done
  args+=(
    --data "$DATA"
    --output "$run_dir"
    --train-seeds "$TRAIN_SEEDS"
    --validation-seeds "$VALIDATION_SEEDS"
    --test-seeds "$TEST_SEEDS"
    --folds "$FOLDS"
    --negotiation-steps "$DEPTH"
    --warmup-quartets "$WARMUP_QUARTETS"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --trainer-seed "$TRAINER_SEED"
    --learning-rate "$learning_rate"
    --exploration-epsilon "$epsilon"
    --zero-target-semantics "$zero_semantics"
    --device cuda:0
  )
  if [[ "$init_mode" == "warmstart" ]]; then
    initial_checkpoint="$WARMSTART_CHECKPOINT"
    args+=(--initial-checkpoint "$WARMSTART_CHECKPOINT")
  fi
  if [[ "$include_shuffled" -eq 1 ]]; then args+=(--include-shuffled); fi
  if [[ "$heads_only" -eq 1 ]]; then args+=(--heads-only); fi

  if [[ $PLAN_ONLY -eq 1 ]]; then
    pid=0
    printf "PLAN gpu=%s trial=%s command=" "$gpu" "$trial"
    printf "%q " env "CUDA_VISIBLE_DEVICES=$gpu" "$PYTHON_BIN" "${args[@]}"
    printf "\n"
  else
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export OMP_NUM_THREADS=20
      export MKL_NUM_THREADS=20
      export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
      cd "$PROJECT_ROOT" || exit 97
      exec "$PYTHON_BIN" "${args[@]}"
    ) > "$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    echo "launched gpu=$gpu trial=$trial pid=$pid log=$log"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$trial" "$gpu" "$init_mode" "$learning_rate" "$epsilon" \
    "$include_shuffled" "$heads_only" "$zero_semantics" "$EPOCHS" \
    "$PATIENCE" "$TRAINER_SEED" "$DEPTH" "$WARMUP_QUARTETS" \
    "$TRAIN_SEEDS" "$VALIDATION_SEEDS" "$TEST_SEEDS" "$pid" \
    "$run_dir" "$log" "$checkpoint" "$initial_checkpoint" >> "$MANIFEST"
done

if [[ $PLAN_ONLY -eq 1 ]]; then
  for trial in "${TRIALS[@]}"; do
    printf "%s\t0\t0\tplanned_not_launched\n" "$trial" >> "$COMPLETION_MANIFEST"
  done
  echo "plan_manifest=$MANIFEST"
  exit 0
fi

status=0
for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  trial="${TRIALS[$index]}"
  if wait "$pid"; then
    rc=0
    state="complete"
    echo "completed trial=$trial pid=$pid"
  else
    rc=$?
    state="failed"
    status=1
    echo "failed trial=$trial pid=$pid rc=$rc" >&2
  fi
  printf "%s\t%s\t%s\t%s\n" "$trial" "$pid" "$rc" "$state" >> "$COMPLETION_MANIFEST"
done

echo "launch_manifest=$MANIFEST"
echo "completion_manifest=$COMPLETION_MANIFEST"
exit "$status"
