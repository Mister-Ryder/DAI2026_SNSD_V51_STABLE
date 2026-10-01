"""CLI for auditable v1.5 exact-factor teacher and head warm-start training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import torch

from snsd_v51.training.offline_dataset import (
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    OfflineDataError,
    load_offline_dataset,
    materialize_warmup_manifest,
)
from snsd_v51.training.warmstart_trainer import TrainerConfig, train_head_warmstart


def _parse_seeds(value: str) -> Tuple[int, ...]:
    result = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("seed lists cannot contain duplicates")
    return result


def _seed_hint(path: Path) -> Optional[int]:
    warmup = path / "warmup_manifest.json"
    if warmup.exists():
        try:
            with warmup.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("seed") is not None:
                return int(payload["seed"])
        except (OSError, ValueError, TypeError):
            pass
    for filename in (
        "shared_manifest.json",
        "comparison_config.resolved.json",
        "config.resolved.json",
        "teacher_config.json",
    ):
        candidate = path / filename
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("seed") is not None:
                return int(payload["seed"])
        except (OSError, ValueError, TypeError):
            continue
    match = re.search(r"seed[_-]?(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _resolve_run_dirs(
    explicit: Sequence[str],
    runs_root: Path,
    generated_root: Path,
    requested: Sequence[int],
    data_path: Optional[str],
    materialize_missing: bool,
) -> Tuple[Path, ...]:
    selected = {}  # type: Dict[int, Path]
    for raw in explicit:
        path = Path(raw).resolve()
        seed = _seed_hint(path)
        if seed is None:
            raise OfflineDataError("cannot determine seed for explicit run %s" % path)
        if seed in selected:
            raise OfflineDataError("duplicate explicit run for seed %d" % seed)
        selected[seed] = path
    for seed in requested:
        if seed in selected:
            continue
        preferred = (
            runs_root / ("comparison_c3_final_seed%d" % seed),
            generated_root / ("warmup_teacher_seed%d" % seed),
        )
        for candidate in preferred:
            if (candidate / "warmup_manifest.json").exists():
                selected[seed] = candidate.resolve()
                break
    missing = [seed for seed in requested if seed not in selected]
    if missing and materialize_missing:
        if data_path is None:
            raise OfflineDataError("--data is required with --materialize-missing")
        for seed in missing:
            run_dir = (generated_root / ("warmup_teacher_seed%d" % seed)).resolve()
            materialize_warmup_manifest(
                seed=seed,
                data_path=data_path,
                output_dir=str(run_dir),
                moves=500,
                restarts=2,
                boundary_top_h=8,
                warmup_quartets=4,
                num_agents=6,
            )
            selected[seed] = run_dir
    missing = [seed for seed in requested if seed not in selected]
    if missing:
        raise OfflineDataError(
            "missing warmup runs for seeds %s; pass --materialize-missing and --data"
            % missing
        )
    return tuple(selected[seed] for seed in requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact-chain warmup teacher labels and train a strict-compatible "
            "v1.5 frozen-encoder head checkpoint"
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
        "--train-seeds",
        type=_parse_seeds,
        default=DEFAULT_TRAIN_SEEDS,
    )
    parser.add_argument(
        "--validation-seeds",
        type=_parse_seeds,
        default=DEFAULT_VALIDATION_SEEDS,
    )
    parser.add_argument(
        "--test-seeds",
        type=_parse_seeds,
        default=DEFAULT_TEST_SEEDS,
    )
    parser.add_argument("--materialize-missing", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--trainer-seed", type=int, default=20260803)
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
        raise OfflineDataError("CUDA device requested but torch.cuda.is_available() is false")
    runs = _resolve_run_dirs(
        explicit=args.run_dir,
        runs_root=Path(args.runs_root).resolve(),
        generated_root=Path(args.generated_runs_dir).resolve(),
        requested=requested,
        data_path=args.data,
        materialize_missing=bool(args.materialize_missing),
    )
    offline = load_offline_dataset(
        run_dirs=[str(path) for path in runs],
        data_path=args.data,
        warmup_quartets=4,
    )
    split = offline.split_by_seed(
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        require_all=True,
    )
    output = Path(args.output).resolve()
    artifacts = offline.write_artifacts(str(output))
    split_path = output / "split_manifest.json"
    split.write_manifest(str(split_path))
    if args.audit_only:
        result = {
            "status": "audit_only",
            "artifacts": artifacts,
            "split_manifest": str(split_path),
            "candidate_count": len(offline.examples),
            "pressure_identifiable": offline.receipt["pressure_identifiable"],
            "pressure_total": offline.receipt["pressure_total"],
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    trainer_config = TrainerConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        gradient_clip=float(args.gradient_clip),
        seed=int(args.trainer_seed),
        device=str(args.device),
        mode="Zero",
    )
    training = train_head_warmstart(
        offline=offline,
        split=split,
        output_dir=str(output),
        config=trainer_config,
    )
    result = training.to_dict()
    result.update(
        {
            "status": "complete",
            "dataset_artifacts": artifacts,
            "split_manifest": str(split_path),
        }
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
