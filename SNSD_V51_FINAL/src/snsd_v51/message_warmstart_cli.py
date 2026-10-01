"""CLI for cross-fitted Real/Zero message warm-start training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import torch

from snsd_v51.warmstart_cli import _parse_seeds, _resolve_run_dirs
from snsd_v51.training.message_teacher import (
    build_crossfit_message_folds,
    write_crossfit_manifest,
)
from snsd_v51.training.offline_dataset import (
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    OfflineDataError,
    load_offline_dataset,
)
from snsd_v51.training.message_warmstart_trainer import (
    MessageTrainerConfig,
    train_message_warmstart,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a strict-compatible v1.5 checkpoint on cross-fitted exact "
            "Real/Zero boundary messages"
        )
    )
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--generated-runs-dir", default="runs/warmup_teacher"
    )
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--train-seeds", type=_parse_seeds, default=DEFAULT_TRAIN_SEEDS
    )
    parser.add_argument(
        "--validation-seeds",
        type=_parse_seeds,
        default=DEFAULT_VALIDATION_SEEDS,
    )
    parser.add_argument(
        "--test-seeds", type=_parse_seeds, default=DEFAULT_TEST_SEEDS
    )
    parser.add_argument("--materialize-missing", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--trainer-seed", type=int, default=20260803)
    parser.add_argument("--zero-weight", type=float, default=1.0)
    parser.add_argument("--shuffled-weight", type=float, default=0.25)
    parser.add_argument("--paired-delta-weight", type=float, default=1.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train_seeds = tuple(args.train_seeds)
    validation_seeds = tuple(args.validation_seeds)
    test_seeds = tuple(args.test_seeds)
    if set(train_seeds).intersection(validation_seeds) or set(train_seeds).intersection(test_seeds) or set(validation_seeds).intersection(test_seeds):
        raise OfflineDataError("train/validation/test seed groups must be disjoint")
    requested = train_seeds + validation_seeds + test_seeds
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise OfflineDataError("CUDA requested but unavailable")
    run_dirs = _resolve_run_dirs(
        explicit=args.run_dir,
        runs_root=Path(args.runs_root).resolve(),
        generated_root=Path(args.generated_runs_dir).resolve(),
        requested=requested,
        data_path=args.data,
        materialize_missing=bool(args.materialize_missing),
    )
    offline = load_offline_dataset(
        [str(path) for path in run_dirs], data_path=args.data, warmup_quartets=4
    )
    split = offline.split_by_seed(
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        require_all=True,
    )
    folds = build_crossfit_message_folds(offline, folds=2)
    output = Path(args.output).resolve()
    artifacts = offline.write_artifacts(str(output))
    split_path = output / "split_manifest.json"
    split.write_manifest(str(split_path))
    crossfit_path = output / "crossfit_message_teacher_manifest.json"
    write_crossfit_manifest(folds, str(crossfit_path))
    if args.audit_only:
        payload = {
            "status": "audit_only",
            "seed_count": len(offline.seeds),
            "fold_count": len(folds),
            "crossfit_target_candidates": sum(
                len(item.target_examples) for item in folds
            ),
            "candidate_objective_in_own_message": False,
            "crossfit_manifest": str(crossfit_path),
            "dataset_artifacts": artifacts,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    config = MessageTrainerConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        gradient_clip=float(args.gradient_clip),
        seed=int(args.trainer_seed),
        device=str(args.device),
        include_shuffled=bool(args.include_shuffled),
        zero_weight=float(args.zero_weight),
        shuffled_weight=float(args.shuffled_weight),
        paired_delta_weight=float(args.paired_delta_weight),
    )
    result = train_message_warmstart(
        offline=offline,
        folds=folds,
        split=split,
        output_dir=str(output),
        config=config,
    )
    payload = result.to_dict()
    payload.update(
        {
            "status": "complete",
            "crossfit_manifest": str(crossfit_path),
            "split_manifest": str(split_path),
            "dataset_artifacts": artifacts,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
