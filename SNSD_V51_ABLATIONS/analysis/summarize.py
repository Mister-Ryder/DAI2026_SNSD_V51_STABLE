"""Summarize and formally validate the v5.1 8x8 Oracle matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


METHODS = (
    "Full",
    "FrozenAnchorComm",
    "SelectionOnly",
    "FrozenAnchorSelectionOnly",
    "AlternatingConflictBenefit",
    "OneShotComm",
    "ShuffledComm",
    "NoPeerFinalShield",
)
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
SEMANTIC_EVENT_TYPE = "v51_paired_semantic_counterfactual"
EXPECTED_SEMANTIC_SCHEDULES = {
    "Full": {"real": 24},
    "FrozenAnchorComm": {"real": 24},
    "SelectionOnly": {"zero": 24},
    "FrozenAnchorSelectionOnly": {"zero": 24},
    "AlternatingConflictBenefit": {"conflict_only": 12, "benefit_only": 12},
    "OneShotComm": {"real": 4, "zero": 20},
    "ShuffledComm": {"shuffled": 24},
    "NoPeerFinalShield": {"zero": 24},
}


def _stats(values):
    values = list(values)
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sum": sum(values) if values else 0.0,
    }


def _paired_stats(values):
    result = _stats(values)
    result.update(
        {
            "positive_seeds": sum(value > 1.0e-9 for value in values),
            "ties": sum(abs(value) <= 1.0e-9 for value in values),
            "negative_seeds": sum(value < -1.0e-9 for value in values),
            "values_by_seed_order": list(values),
        }
    )
    return result


def _number(value):
    return "n/a" if value is None else "{:,.1f}".format(float(value))


def _semantic_schedule_violation(events_path, seed, method):
    """Validate the registered late-event schedule directly from events.jsonl."""

    expected = dict(EXPECTED_SEMANTIC_SCHEDULES[method])
    actual = {}
    event_count = 0
    parse_errors = []
    events_path = Path(events_path)
    if not events_path.is_file():
        return {
            "seed": int(seed),
            "method": str(method),
            "events_path": str(events_path),
            "expected_event_count": 24,
            "actual_event_count": 0,
            "expected_mode_counts": expected,
            "actual_mode_counts": {},
            "errors": ["missing events.jsonl"],
        }

    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if SEMANTIC_EVENT_TYPE not in line:
                    continue
                try:
                    event = json.loads(line)
                except (TypeError, ValueError) as exc:
                    parse_errors.append(
                        "line {}: {}".format(line_number, str(exc))
                    )
                    continue
                if event.get("event_type") != SEMANTIC_EVENT_TYPE:
                    continue
                event_count += 1
                mode = str(event.get("semantic_mode", "<missing>"))
                actual[mode] = int(actual.get(mode, 0)) + 1
    except OSError as exc:
        return {
            "seed": int(seed),
            "method": str(method),
            "events_path": str(events_path),
            "expected_event_count": 24,
            "actual_event_count": int(event_count),
            "expected_mode_counts": expected,
            "actual_mode_counts": actual,
            "errors": ["events.jsonl read failed: {}".format(str(exc))],
        }

    errors = list(parse_errors)
    if int(event_count) != 24:
        errors.append(
            "expected 24 late semantic events, observed {}".format(event_count)
        )
    if actual != expected:
        errors.append(
            "semantic_mode schedule mismatch: expected {}, observed {}".format(
                expected, actual
            )
        )
    if not errors:
        return None
    return {
        "seed": int(seed),
        "method": str(method),
        "events_path": str(events_path),
        "expected_event_count": 24,
        "actual_event_count": int(event_count),
        "expected_mode_counts": expected,
        "actual_mode_counts": actual,
        "errors": errors,
    }


def _parse_seeds(value):
    if value is None:
        return tuple(SEEDS)
    seeds = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("formal v5.1 summary requires exactly eight unique seeds")
    return seeds


def summarize(root, seeds=SEEDS):
    root = Path(root)
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("formal v5.1 summary requires exactly eight unique seeds")
    rows = []
    row_sources = {}
    for path in sorted(root.glob("seed*/*/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("status") == "complete":
            rows.append(row)
            row_sources[(int(row["seed"]), str(row["method"]))] = path

    indexed = {}
    duplicates = []
    for row in rows:
        key = (int(row["seed"]), str(row["method"]))
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    expected_keys = {(seed, method) for seed in seeds for method in METHODS}
    actual_keys = set(indexed)

    budget_violations = []
    pollution_violations = []
    extra_call_violations = []
    semantic_schedule_violations = []
    for key, row in indexed.items():
        calls = int(row.get("online_query_calls", row.get("online_calls", -1)))
        if calls != 288:
            budget_violations.append({"seed": key[0], "method": key[1], "calls": calls})
        pollution = int(row.get("v51_unaccepted_ca_domain_injections", -1))
        if pollution != 0:
            pollution_violations.append(
                {"seed": key[0], "method": key[1], "count": pollution}
            )
        extra = int(row.get("v51_additional_worker_calls", -1))
        if extra != 0:
            extra_call_violations.append(
                {"seed": key[0], "method": key[1], "count": extra}
            )
        source = row_sources.get(key)
        events_path = source.parent / "events.jsonl" if source else root / "missing"
        violation = _semantic_schedule_violation(
            events_path=events_path,
            seed=key[0],
            method=key[1],
        )
        if violation is not None:
            semantic_schedule_violations.append(violation)

    by_method = {}
    for method in METHODS:
        items = [indexed[(seed, method)] for seed in seeds if (seed, method) in indexed]

        def values(field, default=0.0):
            return [float(row.get(field, default)) for row in items]

        by_method[method] = {
            "initial_objective": _stats(values("initial_objective")),
            "final_objective": _stats(values("final_objective")),
            "final_gain": _stats(values("final_gain")),
            "runtime_s": _stats(values("total_runtime_s")),
            "online_worker_calls": _stats(
                [
                    float(row.get("online_query_calls", row.get("online_calls", 0)))
                    for row in items
                ]
            ),
            "late_decisions": _stats(values("v51_late_decisions")),
            "c0_evaluations": _stats(values("v51_c0_evaluations")),
            "ca_evaluations": _stats(values("v51_ca_evaluations")),
            "ca_deduplicated_evaluations": _stats(
                values("v51_ca_deduplicated_evaluations")
            ),
            "semantic_strict_wins": _stats(values("v51_semantic_strict_wins")),
            "semantic_accepts": _stats(values("v51_semantic_accepts")),
            "c0_selections": _stats(values("v51_c0_selections")),
            "ca_minus_c0_mean": _stats(values("v51_ca_minus_c0_mean")),
            "ca_minus_c0_positive_steps": _stats(
                values("v51_ca_minus_c0_positive_steps")
            ),
            "endpoint_difference_steps": _stats(
                values("v51_endpoint_difference_steps")
            ),
            "refill_difference_steps": _stats(values("v51_refill_difference_steps")),
            "distinct_microresponses_c0": _stats(
                values("v51_distinct_microresponses_c0")
            ),
            "distinct_microresponses_ca": _stats(
                values("v51_distinct_microresponses_ca")
            ),
            "payload_bytes_c0": _stats(values("v51_payload_bytes_c0")),
            "payload_bytes_ca": _stats(values("v51_payload_bytes_ca")),
            "ca_domain_injections": _stats(values("v51_ca_domain_injections")),
            "unaccepted_ca_domain_injections": _stats(
                values("v51_unaccepted_ca_domain_injections")
            ),
        }

    def paired(left, right):
        return _paired_stats(
            [
                float(indexed[(seed, left)]["final_gain"])
                - float(indexed[(seed, right)]["final_gain"])
                for seed in seeds
                if (seed, left) in indexed and (seed, right) in indexed
            ]
        )

    effects = {
        "A_given_B": paired("Full", "SelectionOnly"),
        "A_without_B": paired("FrozenAnchorComm", "FrozenAnchorSelectionOnly"),
        "B_given_A": paired("Full", "FrozenAnchorComm"),
        "B_without_A": paired("SelectionOnly", "FrozenAnchorSelectionOnly"),
        "Full_minus_A0B0": paired("Full", "FrozenAnchorSelectionOnly"),
        "Full_minus_OneShot": paired("Full", "OneShotComm"),
        "Full_minus_Shuffled": paired("Full", "ShuffledComm"),
        "Full_minus_NoPeerFinalShield": paired("Full", "NoPeerFinalShield"),
    }
    interaction = []
    for seed in seeds:
        keys = (
            (seed, "Full"),
            (seed, "FrozenAnchorComm"),
            (seed, "SelectionOnly"),
            (seed, "FrozenAnchorSelectionOnly"),
        )
        if all(key in indexed for key in keys):
            interaction.append(
                float(indexed[keys[0]]["final_gain"])
                - float(indexed[keys[1]]["final_gain"])
                - float(indexed[keys[2]]["final_gain"])
                + float(indexed[keys[3]]["final_gain"])
            )
    effects["A_x_B_interaction"] = _paired_stats(interaction)

    per_seed = []
    for seed in seeds:
        for method in METHODS:
            row = indexed.get((seed, method))
            if row is None:
                continue
            per_seed.append(
                {
                    "seed": seed,
                    "method": method,
                    "initial_objective": float(row["initial_objective"]),
                    "final_objective": float(row["final_objective"]),
                    "final_gain": float(row["final_gain"]),
                    "semantic_strict_wins": int(
                        row.get("v51_semantic_strict_wins", 0)
                    ),
                    "semantic_accepts": int(row.get("v51_semantic_accepts", 0)),
                    "ca_minus_c0_mean": float(row.get("v51_ca_minus_c0_mean", 0.0)),
                    "endpoint_difference_steps": int(
                        row.get("v51_endpoint_difference_steps", 0)
                    ),
                    "refill_difference_steps": int(
                        row.get("v51_refill_difference_steps", 0)
                    ),
                }
            )

    complete = (
        len(indexed) == len(seeds) * len(METHODS)
        and not duplicates
        and actual_keys == expected_keys
        and not budget_violations
        and not pollution_violations
        and not extra_call_violations
        and not semantic_schedule_violations
    )
    return {
        "schema_version": "SNSD-v1.5-v5.1-formal-summary-v2",
        "source_root": str(root.resolve()),
        "completed_runs": len(indexed),
        "expected_runs": len(seeds) * len(METHODS),
        "complete_matrix": bool(complete),
        "formal_contract": {
            "complete_c3_non_smoke": True,
            "seeds": list(seeds),
            "methods": list(METHODS),
            "negotiation_depth": 12,
            "macro_rounds": 4,
            "paid_worker_calls_per_run": 288,
            "same_state_paired_c0_ca": True,
            "missing_cells": [list(key) for key in sorted(expected_keys - actual_keys)],
            "unexpected_cells": [list(key) for key in sorted(actual_keys - expected_keys)],
            "duplicate_cells": [list(key) for key in duplicates],
            "budget_violations": budget_violations,
            "unaccepted_ca_pollution_violations": pollution_violations,
            "additional_worker_call_violations": extra_call_violations,
            "semantic_schedule_checked_runs": len(indexed),
            "expected_semantic_schedules": EXPECTED_SEMANTIC_SCHEDULES,
            "semantic_schedule_violations": semantic_schedule_violations,
        },
        "methods": by_method,
        "paired_effects": effects,
        "per_seed": per_seed,
    }


def render_report(result, evaluation_label="Oracle"):
    lines = [
        "# v5.1 Paired Semantic-Counterfactual Safeguard: Formal C3 {} Results".format(
            str(evaluation_label)
        ),
        "",
        "The formal matrix contains {}/{} complete non-smoke runs. Every cell uses "
        "the same 12-step x 4-macro protocol and 288 paid worker calls.".format(
            result["completed_runs"], result["expected_runs"]
        ),
        "",
        "## Method-level results",
        "",
        "| Method | Mean gain +/- SD | Mean final objective | Semantic strict wins | "
        "Semantic accepts | Mean CA-C0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = result["methods"][method]
        lines.append(
            "| {} | {} +/- {} | {} | {} | {} | {} |".format(
                method,
                _number(row["final_gain"]["mean"]),
                _number(row["final_gain"]["sample_std"]),
                _number(row["final_objective"]["mean"]),
                _number(row["semantic_strict_wins"]["mean"]),
                _number(row["semantic_accepts"]["mean"]),
                _number(row["ca_minus_c0_mean"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired factorial effects",
            "",
            "| Effect | Mean +/- SD | Positive / Tie / Negative seeds | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    labels = (
        ("A_given_B", "A \\| B"),
        ("A_without_B", "A \\| not B"),
        ("B_given_A", "B \\| A"),
        ("B_without_A", "B \\| not A"),
        ("A_x_B_interaction", "A x B"),
        ("Full_minus_A0B0", "Full - A0B0"),
        ("Full_minus_OneShot", "Full - OneShot"),
        ("Full_minus_Shuffled", "Full - Shuffled"),
        ("Full_minus_NoPeerFinalShield", "Full - NoPeer"),
    )
    for key, label in labels:
        row = result["paired_effects"][key]
        lines.append(
            "| {} | {} +/- {} | {}/{}/{} | {} | {} |".format(
                label,
                _number(row["mean"]),
                _number(row["sample_std"]),
                row["positive_seeds"],
                row["ties"],
                row["negative_seeds"],
                _number(row["min"]),
                _number(row["max"]),
            )
        )
    contract = result["formal_contract"]
    lines.extend(
        [
            "",
            "## Formal safeguards",
            "",
            "- Complete matrix: `{}`.".format(result["complete_matrix"]),
            "- Worker-budget violations: `{}`.".format(
                len(contract["budget_violations"])
            ),
            "- Unaccepted-CA domain-pollution violations: `{}`.".format(
                len(contract["unaccepted_ca_pollution_violations"])
            ),
            "- Additional paid worker-call violations: `{}`.".format(
                len(contract["additional_worker_call_violations"])
            ),
            "- Semantic-schedule violations from `events.jsonl`: `{}`.".format(
                len(contract["semantic_schedule_violations"])
            ),
            "",
        ]
    )
    if contract["semantic_schedule_violations"]:
        lines.extend(["## Semantic-schedule violations", ""])
        for violation in contract["semantic_schedule_violations"]:
            lines.append(
                "- Seed `{}` / `{}`: expected 24 events with `{}`, observed {} "
                "events with `{}`; {}.".format(
                    violation["seed"],
                    violation["method"],
                    violation["expected_mode_counts"],
                    violation["actual_event_count"],
                    violation["actual_mode_counts"],
                    "; ".join(violation["errors"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--evaluation-label", default="Oracle")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="exactly eight comma-separated formal seeds",
    )
    args = parser.parse_args(argv)
    result = summarize(args.root, seeds=_parse_seeds(args.seeds))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            render_report(result, evaluation_label=args.evaluation_label),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["complete_matrix"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
