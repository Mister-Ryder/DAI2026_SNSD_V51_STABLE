#!/usr/bin/env bash
# Portable 8-seed x 8-method formal matrix launcher for the frozen v5.1 core.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FINAL_ROOT="$(cd "$ABLATION_ROOT/../SNSD_V51_FINAL" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="$FINAL_ROOT/data/C3.csv"
CHECKPOINT="$FINAL_ROOT/checkpoints/checkpoint_message_v4_5_strict.pt"
OUTPUT_ROOT=""
MODE="learned"
SEEDS_CSV="11,22,33,44,55,66,77,88"
THREADS_PER_JOB=2
PLAN_ONLY=0

usage() {
  echo "Usage: $0 --output-root NEW_DIR [--mode learned|oracle] [--seeds CSV] [--data C3.csv] [--checkpoint FILE] [--python FILE] [--threads-per-job N] [--plan-only]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --threads-per-job) THREADS_PER_JOB="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$OUTPUT_ROOT" ]] || { usage >&2; exit 2; }
[[ "$MODE" == "learned" || "$MODE" == "oracle" ]] || {
  echo "--mode must be learned or oracle" >&2; exit 2;
}
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "formal output must be a new path: $OUTPUT_ROOT" >&2; exit 2;
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
METHODS=(
  Full
  FrozenAnchorComm
  SelectionOnly
  FrozenAnchorSelectionOnly
  AlternatingConflictBenefit
  OneShotComm
  ShuffledComm
  NoPeerFinalShield
)
[[ ${#SEEDS[@]} -eq 8 && ${#METHODS[@]} -eq 8 ]] || {
  echo "formal protocol requires exactly 8 seeds and 8 methods" >&2; exit 2;
}

RUNNER="$ABLATION_ROOT/run_ablation.py"
SUMMARIZER="$ABLATION_ROOT/analysis/summarize.py"
FINALIZER="$ABLATION_ROOT/analysis/finalize_audit.py"
CHECKER="$FINAL_ROOT/scripts/check_checkpoint.py"

if [[ $PLAN_ONLY -eq 1 ]]; then
  echo "mode=$MODE"
  echo "data=$DATA"
  echo "checkpoint=$CHECKPOINT"
  echo "output_root=$OUTPUT_ROOT"
  echo "seeds=${SEEDS[*]}"
  echo "methods=${METHODS[*]}"
  echo "cells=64 paid_worker_calls_per_cell=288 total_paid_worker_calls=18432"
  exit 0
fi

for path in "$DATA" "$RUNNER" "$SUMMARIZER"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "missing Python: $PYTHON_BIN" >&2; exit 2;
}
if [[ "$MODE" == "learned" ]]; then
  [[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
  "$PYTHON_BIN" "$CHECKER" "$CHECKPOINT" > "$OUTPUT_ROOT.checkpoint_preflight.json"
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[[ "$GPU_COUNT" -ge 8 ]] || { echo "eight visible GPUs are required" >&2; exit 2; }

"$PYTHON_BIN" "$RUNNER"   --data "$DATA" --output "$OUTPUT_ROOT/.preflight"   --method Full --seed "${SEEDS[0]}" --preflight-only   > "$OUTPUT_ROOT.preflight.json"

mkdir -p "$OUTPUT_ROOT/logs"
printf "seed	method	gpu	mode	paid_calls	pid	output	log
" > "$OUTPUT_ROOT/launch_manifest.tsv"
PIDS=()
index=0
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    gpu=$((index % 8))
    run_dir="$OUTPUT_ROOT/seed${seed}/${method}"
    log="$OUTPUT_ROOT/logs/seed${seed}_${method}.log"
    args=(
      "$RUNNER" --data "$DATA" --output "$run_dir"
      --method "$method" --seed "$seed" --device cuda:0
    )
    if [[ "$MODE" == "learned" ]]; then
      args+=(--checkpoint "$CHECKPOINT")
    else
      args+=(--oracle)
    fi
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export OMP_NUM_THREADS="$THREADS_PER_JOB"
      export MKL_NUM_THREADS="$THREADS_PER_JOB"
      exec "$PYTHON_BIN" "${args[@]}"
    ) > "$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    printf "%s	%s	%s	%s	288	%s	%s	%s
"       "$seed" "$method" "$gpu" "$MODE" "$pid" "$run_dir" "$log"       >> "$OUTPUT_ROOT/launch_manifest.tsv"
    index=$((index + 1))
  done
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done

SUMMARY="$OUTPUT_ROOT/formal_summary.json"
REPORT="$OUTPUT_ROOT/formal_report.md"
LABEL="Learned"
[[ "$MODE" == "oracle" ]] && LABEL="Oracle Mechanism Diagnostic"
"$PYTHON_BIN" "$SUMMARIZER"   --root "$OUTPUT_ROOT" --output "$SUMMARY" --report "$REPORT"   --evaluation-label "$LABEL" --seeds "$SEEDS_CSV" || status=1
if [[ "$MODE" == "learned" ]]; then
  "$PYTHON_BIN" "$FINALIZER"     --root "$OUTPUT_ROOT" --checkpoint "$CHECKPOINT" --summary "$SUMMARY" || status=1
fi
exit "$status"
