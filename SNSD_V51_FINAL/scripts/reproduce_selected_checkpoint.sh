#!/usr/bin/env bash
# Re-run the exact validation-selected training cell recorded by the formal
# model manifest. This reconstructs the frozen training contract; it does not
# promise bitwise-identical floating-point output across environments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="$PROJECT_ROOT/data/C3.csv"
WARMUP_ROOT="$PROJECT_ROOT/checkpoints/provenance/warmup_teacher_strict"
INITIAL_CHECKPOINT="$PROJECT_ROOT/checkpoints/provenance/formal_warmstart_message_v3/checkpoint_message_v3.pt"
OUTPUT=""
DEVICE="${DEVICE:-cuda:0}"
PLAN_ONLY=0

usage() {
  echo "Usage: $0 --output DIR [--python FILE] [--data C3.csv] [--warmup-root DIR] [--initial-checkpoint FILE] [--device DEVICE] [--plan-only]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --warmup-root) WARMUP_ROOT="$2"; shift 2 ;;
    --initial-checkpoint) INITIAL_CHECKPOINT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then usage >&2; exit 2; fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi
for required in "$DATA" "$INITIAL_CHECKPOINT" "$PROJECT_ROOT/train_gnn.py"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

args=("$PROJECT_ROOT/train_gnn.py")
for seed in 11 22 33 44 55 66 77 88; do
  run_dir="$WARMUP_ROOT/strict_depth12_q7_seed$seed"
  if [[ ! -f "$run_dir/warmup_manifest.json" ]]; then
    echo "Missing formal warm-up manifest: $run_dir/warmup_manifest.json" >&2
    exit 2
  fi
  args+=(--run-dir "$run_dir")
done
args+=(
  --data "$DATA"
  --output "$OUTPUT"
  --train-seeds 44,55,66,77,88
  --validation-seeds 33
  --test-seeds 11,22
  --folds 2
  --negotiation-steps 12
  --warmup-quartets 7
  --epochs 12
  --learning-rate 3e-4
  --weight-decay 1e-5
  --patience 4
  --gradient-clip 5.0
  --trainer-seed 20260803
  --exploration-epsilon 0.05
  --real-weight 1.0
  --zero-weight 0.25
  --shuffled-weight 0.10
  --zero-target-semantics intrinsic
  --initial-checkpoint "$INITIAL_CHECKPOINT"
  --device "$DEVICE"
)

if [[ $PLAN_ONLY -eq 1 ]]; then
  printf "FORMAL_SELECTED_COMMAND="
  printf "%q " "$PYTHON_BIN" "${args[@]}"
  printf "\n"
  exit 0
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "${args[@]}"
