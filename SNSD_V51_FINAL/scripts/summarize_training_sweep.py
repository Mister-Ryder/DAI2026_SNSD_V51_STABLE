"""Summarize the strict v4.5 depth-12 training sweep without test selection.

The winner is locked using ``best_validation_loss`` only.  Test metrics are
attached afterwards and are explicitly labelled post-selection diagnostics.
This ordering is intentional and regression-tested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "SNSD-v1.5-strict-training-sweep-summary-v1"
MODEL_MANIFEST = "model_manifest_message_v4_5_strict.json"
REQUIRED_LAUNCH_FIELDS = (
    "trial",
    "gpu",
    "init_mode",
    "learning_rate",
    "exploration_epsilon",
    "include_shuffled",
    "heads_only",
    "zero_target_semantics",
    "pid",
    "run_dir",
    "log",
    "checkpoint",
    "train_seeds",
    "validation_seeds",
    "test_seeds",
    "negotiation_steps",
)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_launch_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in REQUIRED_LAUNCH_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                "launch manifest misses required fields: {}".format(
                    ", ".join(missing)
                )
            )
        rows = [dict(row) for row in reader]
    if len(rows) != 8:
        raise ValueError("strict sweep must contain exactly eight trials")
    trials = [row["trial"] for row in rows]
    if len(set(trials)) != len(trials):
        raise ValueError("launch manifest contains duplicate trial names")
    if {row["gpu"] for row in rows} != {str(value) for value in range(8)}:
        raise ValueError("strict sweep must map one trial to each GPU 0..7")
    for row in rows:
        if row["train_seeds"] != "44,55,66,77,88":
            raise ValueError("unexpected training split for {}".format(row["trial"]))
        if row["validation_seeds"] != "33":
            raise ValueError("unexpected validation split for {}".format(row["trial"]))
        if row["test_seeds"] != "11,22":
            raise ValueError("unexpected test split for {}".format(row["trial"]))
        if int(row["negotiation_steps"]) != 12:
            raise ValueError("unexpected negotiation depth for {}".format(row["trial"]))
    return rows


def _validation_step_rows(
    trial: str, rollout: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = rollout.get("per_step", {})
    if not isinstance(raw, Mapping):
        raise ValueError("validation_rollout.per_step must be an object")
    rows = []
    total_changes = 0
    total_slots = 0
    for key in sorted(raw):
        value = raw[key]
        if not isinstance(value, Mapping) or "/step_" not in str(key):
            raise ValueError("invalid per-step validation row: {}".format(key))
        variant, step_text = str(key).split("/step_", 1)
        observations = int(value.get("observations", 0))
        changes = int(value.get("anchor_change_events", 0))
        row = {
            "trial": trial,
            "variant": variant,
            "step": int(step_text),
            "observations": observations,
            "candidate_samples": int(value.get("candidate_samples", 0)),
            "minimum_candidates_per_agent": int(
                value.get("minimum_candidates_per_agent", 0)
            ),
            "revealed_candidates": int(value.get("revealed_candidates", 0)),
            "anchor_change_events": changes,
            "anchor_change_rate": _finite_or_none(
                value.get("anchor_change_rate")
            ),
            "full_agent_activity": bool(value.get("full_agent_activity", False)),
        }
        rows.append(row)
        total_changes += changes
        total_slots += observations * 6
    return rows, {
        "validation_step_rows": len(rows),
        "validation_full_activity_steps": sum(
            bool(row["full_agent_activity"]) for row in rows
        ),
        "validation_all_steps_full_agent_activity": bool(rows)
        and all(bool(row["full_agent_activity"]) for row in rows),
        "validation_anchor_change_events": total_changes,
        "validation_anchor_change_rate": (
            float(total_changes) / float(total_slots) if total_slots else None
        ),
    }


def _base_trial_row(
    launch: Mapping[str, str], graph_layers: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    validation = graph_layers.get("validation_metrics", {})
    if not isinstance(validation, Mapping):
        raise ValueError("validation_metrics must be an object")
    validation_rollout = graph_layers.get("validation_rollout", {})
    if not isinstance(validation_rollout, Mapping):
        raise ValueError("validation_rollout must be an object")
    step_rows, activity = _validation_step_rows(
        launch["trial"], validation_rollout
    )
    best_validation_loss = _finite_or_none(graph_layers.get("best_validation_loss"))
    if best_validation_loss is None:
        raise ValueError(
            "{} has no finite best_validation_loss".format(launch["trial"])
        )
    row = {
        "trial": launch["trial"],
        "status": "complete",
        "gpu": int(launch["gpu"]),
        "init_mode": launch["init_mode"],
        "learning_rate": float(launch["learning_rate"]),
        "exploration_epsilon": float(launch["exploration_epsilon"]),
        "include_shuffled": bool(int(launch["include_shuffled"])),
        "heads_only": bool(int(launch["heads_only"])),
        "zero_target_semantics": launch["zero_target_semantics"],
        "checkpoint": launch["checkpoint"],
        "checkpoint_hash": graph_layers.get("checkpoint_hash"),
        "best_epoch": _int_or_none(graph_layers.get("best_epoch")),
        "best_validation_loss": best_validation_loss,
        "validation_real_rank_accuracy": _finite_or_none(
            validation.get("real_rank_accuracy")
        ),
        "validation_zero_rank_accuracy": _finite_or_none(
            validation.get("zero_rank_accuracy")
        ),
        "validation_shuffled_rank_accuracy": _finite_or_none(
            validation.get("shuffled_rank_accuracy")
        ),
        "parameter_count": _int_or_none(graph_layers.get("parameter_count")),
        "trainable_parameter_count": _int_or_none(
            graph_layers.get("trainable_parameter_count")
        ),
        "encoder_trainable_parameter_count": _int_or_none(
            graph_layers.get("encoder_trainable_parameter_count")
        ),
    }
    row.update(activity)
    return row, step_rows


def _attach_postselection_test_metrics(
    row: Dict[str, Any], graph_layers: Mapping[str, Any]
) -> None:
    test = graph_layers.get("test_metrics", {})
    if not isinstance(test, Mapping):
        test = {}
    # This function is called only after the validation-only winner is locked.
    row["postselection_test_loss"] = _finite_or_none(test.get("loss"))
    row["postselection_test_real_rank_accuracy"] = _finite_or_none(
        test.get("real_rank_accuracy")
    )
    row["postselection_test_zero_rank_accuracy"] = _finite_or_none(
        test.get("zero_rank_accuracy")
    )


def summarize_strict_training_sweep(
    output_root: Any, require_complete: bool = False
) -> Dict[str, Any]:
    root = Path(output_root).resolve()
    launches = read_launch_manifest(root / "launch_manifest.tsv")
    complete = []
    incomplete = []
    models = {}
    step_rows = []
    for launch in launches:
        model_path = Path(launch["run_dir"]) / MODEL_MANIFEST
        if not model_path.is_file():
            incomplete.append(
                {
                    "trial": launch["trial"],
                    "status": "missing_model_manifest",
                    "model_manifest": str(model_path),
                    "log": launch["log"],
                }
            )
            continue
        graph_layers = _read_json(model_path)
        row, trial_steps = _base_trial_row(launch, graph_layers)
        complete.append(row)
        models[launch["trial"]] = graph_layers
        step_rows.extend(trial_steps)
    if require_complete and incomplete:
        raise ValueError(
            "strict sweep is incomplete: {}".format(
                ", ".join(row["trial"] for row in incomplete)
            )
        )

    # Lock before reading or attaching any test metric.  Tie-breaks use only a
    # validation metric and then the stable trial name, never a test outcome.
    ranked = sorted(
        complete,
        key=lambda row: (
            float(row["best_validation_loss"]),
            -float(row["validation_real_rank_accuracy"])
            if row["validation_real_rank_accuracy"] is not None
            else 0.0,
            str(row["trial"]),
        ),
    )
    selected_trial = ranked[0]["trial"] if ranked else None
    for validation_rank, row in enumerate(ranked, 1):
        row["validation_rank"] = validation_rank
    for row in complete:
        _attach_postselection_test_metrics(row, models[row["trial"]])

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not incomplete else "partial",
        "launch_trial_count": len(launches),
        "completed_trial_count": len(complete),
        "incomplete_trial_count": len(incomplete),
        "selection_protocol": {
            "selected_trial": selected_trial,
            "primary_metric": "best_validation_loss",
            "primary_direction": "minimize",
            "validation_seeds": [33],
            "test_seeds": [11, 22],
            "test_metrics_used_for_selection": False,
            "test_metric_role": "post-selection diagnostic only",
        },
        "trials_by_validation_rank": ranked,
        "incomplete_trials": incomplete,
        "validation_per_step": sorted(
            step_rows,
            key=lambda row: (str(row["trial"]), str(row["variant"]), int(row["step"])),
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument("--trials-csv")
    parser.add_argument("--steps-csv")
    args = parser.parse_args(argv)
    result = summarize_strict_training_sweep(
        args.output_root, require_complete=bool(args.require_complete)
    )
    root = Path(args.output_root).resolve()
    json_out = Path(args.json_out).resolve() if args.json_out else root / "training_sweep_summary.json"
    trials_csv = Path(args.trials_csv).resolve() if args.trials_csv else root / "training_sweep_trials.csv"
    steps_csv = Path(args.steps_csv).resolve() if args.steps_csv else root / "training_sweep_validation_steps.csv"
    json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(trials_csv, result["trials_by_validation_rank"])
    _write_csv(steps_csv, result["validation_per_step"])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
