"""CLI for strict end-to-end deep Real/Zero rolling training."""

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
from snsd_v51.training.strict_teacher import (
    build_strict_crossfit_rolling_trajectories,
    write_strict_rolling_teacher_manifest,
)
from snsd_v51.training.offline_dataset import (
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    OfflineDataError,
    load_offline_dataset,
    materialize_warmup_manifest,
)
from snsd_v51.training.gnn_trainer import (
    StrictRollingMessageTrainerConfig,
    train_strict_rolling_message_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train strict end-to-end v1.5 communication on two on-policy "
            "deep Real/Zero branches (12 transitions by default)"
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
    parser.add_argument("--heads-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--negotiation-steps", type=int, default=12)
    parser.add_argument(
        "--warmup-quartets",
        type=int,
        default=0,
        help="0 chooses enough distinct candidates for folds*negotiation_steps",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--trainer-seed", type=int, default=20260803)
    parser.add_argument("--exploration-epsilon", type=float, default=0.05)
    parser.add_argument("--real-weight", type=float, default=1.0)
    parser.add_argument("--zero-weight", type=float, default=0.25)
    parser.add_argument("--shuffled-weight", type=float, default=0.10)
    parser.add_argument(
        "--zero-target-semantics",
        choices=("intrinsic", "global"),
        default="intrinsic",
        help=(
            "intrinsic uses the exact information-restricted local marginal; "
            "global is a same-global-label sensitivity analysis"
        ),
    )
    parser.add_argument("--initial-checkpoint")
    parser.add_argument(
        "--late-stage-weighted",
        action="store_true",
        help=(
            "weight exact step losses 1x/2x/3x over steps 0-5/6-8/9-11 "
            "and prioritize exact-label near-incumbent/hard-negative pairs"
        ),
    )
    parser.add_argument("--priority-pair-weight", type=float, default=1.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train_seeds = tuple(args.train_seeds)
    validation_seeds = tuple(args.validation_seeds)
    test_seeds = tuple(args.test_seeds)
    if (
        set(train_seeds).intersection(validation_seeds)
        or set(train_seeds).intersection(test_seeds)
        or set(validation_seeds).intersection(test_seeds)
    ):
        raise OfflineDataError("train/validation/test seed groups must be disjoint")
    if int(args.negotiation_steps) < 6:
        raise OfflineDataError("strict v4.5 requires at least six steps")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise OfflineDataError("CUDA requested but unavailable")
    requested = train_seeds + validation_seeds + test_seeds
    # One safe action and the factor-selected incumbent can be distinct, so a
    # 4Q+1 warm-up table may expose only 4Q-1 cross-fit-eligible candidates.
    minimum_quartets = (
        int(args.folds) * int(args.negotiation_steps) + 4
    ) // 4
    warmup_quartets = (
        int(args.warmup_quartets)
        if int(args.warmup_quartets) > 0
        else minimum_quartets
    )
    if warmup_quartets < minimum_quartets:
        raise OfflineDataError(
            "warmup_quartets=%d cannot supply %d folds x %d steps; need >=%d"
            % (
                warmup_quartets,
                int(args.folds),
                int(args.negotiation_steps),
                minimum_quartets,
            )
        )
    if args.materialize_missing and not args.run_dir:
        if args.data is None:
            raise OfflineDataError("--data is required with --materialize-missing")
        generated = Path(args.generated_runs_dir).resolve()
        resolved = []
        for seed in requested:
            run_dir = generated / (
                "strict_depth%d_q%d_seed%d"
                % (int(args.negotiation_steps), warmup_quartets, int(seed))
            )
            if not (run_dir / "warmup_manifest.json").exists():
                materialize_warmup_manifest(
                    seed=int(seed),
                    data_path=str(args.data),
                    output_dir=str(run_dir),
                    moves=500,
                    restarts=2,
                    boundary_top_h=8,
                    warmup_quartets=warmup_quartets,
                    num_agents=6,
                )
            resolved.append(run_dir.resolve())
        run_dirs = tuple(resolved)
    else:
        run_dirs = _resolve_run_dirs(
            explicit=args.run_dir,
            runs_root=Path(args.runs_root).resolve(),
            generated_root=Path(args.generated_runs_dir).resolve(),
            requested=requested,
            data_path=args.data,
            materialize_missing=False,
        )
    available_by_run = []
    for run_dir in run_dirs:
        manifest_path = Path(run_dir) / "warmup_manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                warmup_payload = json.load(stream)
            quartet_tables = warmup_payload.get("quartets_by_agent", {})
            if quartet_tables:
                available_quartets = min(
                    len(value) for value in quartet_tables.values()
                )
            else:
                signature_tables = warmup_payload.get("signatures_by_agent", {})
                available_quartets = min(
                    max(0, (len(value) - 1) // 4)
                    for value in signature_tables.values()
                )
        except (OSError, TypeError, ValueError):
            available_quartets = -1
        available_by_run.append((run_dir, available_quartets))
    if (
        not args.materialize_missing
        and int(args.warmup_quartets) == 0
        and available_by_run
    ):
        available_values = {value for _, value in available_by_run}
        if len(available_values) == 1 and next(iter(available_values)) >= minimum_quartets:
            warmup_quartets = next(iter(available_values))
    for run_dir, available_quartets in available_by_run:
        if available_quartets != warmup_quartets:
            raise OfflineDataError(
                "run %s has warmup_quartets=%d but depth %d requires %d; "
                "rerun with --materialize-missing (and without legacy --run-dir)"
                % (
                    run_dir,
                    available_quartets,
                    int(args.negotiation_steps),
                    warmup_quartets,
                )
            )
    offline = load_offline_dataset(
        [str(path) for path in run_dirs],
        data_path=args.data,
        warmup_quartets=warmup_quartets,
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
        observations = build_strict_crossfit_rolling_trajectories(
            offline,
            folds=int(args.folds),
            negotiation_steps=int(args.negotiation_steps),
        )
        teacher_path = output / "rolling_teacher_audit_manifest_v4_5_strict.json"
        write_strict_rolling_teacher_manifest(observations, str(teacher_path))
        payload = {
            "status": "audit_only",
            "schema_version": "SNSD-v1.5-e2e-on-policy-deep-real-zero-v4.5-strict",
            "seed_count": len(offline.seeds),
            "observation_count": len(observations),
            "negotiation_steps": int(args.negotiation_steps),
            "warmup_quartets": warmup_quartets,
            "consumed_message_observations": sum(
                item.consumes_message for item in observations
            ),
            "all_observations_full_agent_activity": all(
                bool(item.receipt.get("full_agent_activity"))
                for item in observations
            ),
            "candidate_objective_in_own_message": False,
            "teacher_manifest": str(teacher_path),
            "dataset_artifacts": artifacts,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    config = StrictRollingMessageTrainerConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        gradient_clip=float(args.gradient_clip),
        seed=int(args.trainer_seed),
        device=str(args.device),
        folds=int(args.folds),
        negotiation_steps=int(args.negotiation_steps),
        heads_only=bool(args.heads_only),
        include_shuffled=bool(args.include_shuffled),
        exploration_epsilon=float(args.exploration_epsilon),
        real_weight=float(args.real_weight),
        zero_weight=float(args.zero_weight),
        shuffled_weight=float(args.shuffled_weight),
        zero_target_semantics=str(args.zero_target_semantics),
        initial_checkpoint=args.initial_checkpoint,
        late_stage_weighted=bool(args.late_stage_weighted),
        priority_pair_weight=float(args.priority_pair_weight),
    )
    result = train_strict_rolling_message_policy(
        offline=offline,
        split=split,
        output_dir=str(output),
        config=config,
    )
    payload = result.to_dict()
    payload.update(
        {
            "status": "complete",
            "split_manifest": str(split_path),
            "dataset_artifacts": artifacts,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
