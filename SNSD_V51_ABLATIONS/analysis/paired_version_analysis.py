#!/usr/bin/env python3
"""Reproducible paired analysis for the formal v5.0/v5.1 C3 matrices.

The script deliberately works from per-seed rows rather than differences of
aggregate means.  It also reports an exact sign-flip randomization p-value so
that the small eight-seed sample is not presented as stronger evidence than it
is.  Python 3.7 compatibility is required by the base_protocol server.
"""

from __future__ import print_function

import argparse
import itertools
import json
import math
from pathlib import Path


def _mean(values):
    return sum(values) / float(len(values)) if values else float("nan")


def _sample_std(values):
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _two_sided_sign_p(values):
    nonzero = [value for value in values if abs(value) > 1.0e-9]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(value > 0.0 for value in nonzero)
    tail = min(positive, n - positive)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / float(2 ** n)
    return min(1.0, probability)


def _two_sided_signflip_mean_p(values):
    """Exact paired randomization test based on the absolute mean delta."""

    nonzero = [float(value) for value in values if abs(value) > 1.0e-9]
    n = len(nonzero)
    if n == 0:
        return 1.0
    observed = abs(_mean(nonzero))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        permuted = abs(_mean([sign * value for sign, value in zip(signs, nonzero)]))
        if permuted + 1.0e-12 >= observed:
            extreme += 1
        total += 1
    return extreme / float(total)


def _stats(values):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": _mean(values),
        "sample_std": _sample_std(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "positive": sum(value > 1.0e-9 for value in values),
        "ties": sum(abs(value) <= 1.0e-9 for value in values),
        "negative": sum(value < -1.0e-9 for value in values),
        "two_sided_sign_p": _two_sided_sign_p(values),
        "two_sided_signflip_mean_p": _two_sided_signflip_mean_p(values),
        "values": values,
    }


def _load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("complete_matrix") is not True:
        raise RuntimeError("formal matrix is not complete: {}".format(path))
    rows = result.get("per_seed")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("missing per_seed rows: {}".format(path))
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), str(row["method"]))
        if key in indexed:
            raise RuntimeError("duplicate row {} in {}".format(key, path))
        indexed[key] = row
    return result, indexed


def _paired_delta(left, right, left_method, right_method, metric="final_gain"):
    left_seeds = {seed for seed, method in left if method == left_method}
    right_seeds = {seed for seed, method in right if method == right_method}
    seeds = sorted(left_seeds & right_seeds)
    if not seeds:
        raise RuntimeError("no paired seeds for {} and {}".format(left_method, right_method))
    values = [
        float(left[(seed, left_method)][metric]) - float(right[(seed, right_method)][metric])
        for seed in seeds
    ]
    result = _stats(values)
    result["seeds"] = seeds
    return result


def _method_names(indexed):
    return sorted({method for _, method in indexed})


def _method_stats(indexed, metric="final_gain"):
    result = {}
    for method in _method_names(indexed):
        seeds = sorted(seed for seed, candidate_method in indexed if candidate_method == method)
        values = [float(indexed[(seed, method)][metric]) for seed in seeds]
        result[method] = _stats(values)
        result[method]["seeds"] = seeds
    return result


def _merge_disjoint_indexes(first, second):
    overlap = set(first) & set(second)
    if overlap:
        raise RuntimeError("pooled matrices contain duplicate seed/method cells: {}".format(sorted(overlap)))
    result = dict(first)
    result.update(second)
    return result


def _factor_effects(indexed, metric="final_gain"):
    required = (
        "Full",
        "FrozenAnchorComm",
        "SelectionOnly",
        "FrozenAnchorSelectionOnly",
    )
    seed_sets = [
        {seed for seed, method in indexed if method == required_method}
        for required_method in required
    ]
    seeds = sorted(set.intersection(*seed_sets))
    effects = {
        "A_given_B": [],
        "A_without_B": [],
        "B_given_A": [],
        "B_without_A": [],
        "A_x_B_interaction": [],
    }
    for seed in seeds:
        full = float(indexed[(seed, "Full")][metric])
        frozen = float(indexed[(seed, "FrozenAnchorComm")][metric])
        selection = float(indexed[(seed, "SelectionOnly")][metric])
        a0b0 = float(indexed[(seed, "FrozenAnchorSelectionOnly")][metric])
        effects["A_given_B"].append(full - selection)
        effects["A_without_B"].append(frozen - a0b0)
        effects["B_given_A"].append(full - frozen)
        effects["B_without_A"].append(selection - a0b0)
        effects["A_x_B_interaction"].append(full - frozen - selection + a0b0)
    return seeds, effects


def _factor_stats(indexed):
    seeds, effects = _factor_effects(indexed)
    result = {}
    for name, values in effects.items():
        result[name] = _stats(values)
        result[name]["seeds"] = seeds
    return result


def _factor_delta_stats(left, right):
    left_seeds, left_effects = _factor_effects(left)
    right_seeds, right_effects = _factor_effects(right)
    if left_seeds != right_seeds:
        raise RuntimeError("factorial matrices do not use identical paired seeds")
    result = {}
    for name in sorted(left_effects):
        values = [
            left_value - right_value
            for left_value, right_value in zip(left_effects[name], right_effects[name])
        ]
        result[name] = _stats(values)
        result[name]["seeds"] = left_seeds
    return result


def _fmt(stats):
    return "{:.1f} +/- {:.1f} ({}/{}/{})".format(
        stats["mean"],
        stats["sample_std"],
        stats["positive"],
        stats["ties"],
        stats["negative"],
    )


def analyze(
    v50_learned_path,
    v51_learned_path,
    v50_oracle_path=None,
    v51_oracle_path=None,
    v51_confirmatory_path=None,
):
    v50_learned, v50_l_rows = _load(v50_learned_path)
    v51_learned, v51_l_rows = _load(v51_learned_path)

    learned_version_deltas = {}
    for method in sorted(set(_method_names(v50_l_rows)) & set(_method_names(v51_l_rows))):
        learned_version_deltas[method] = _paired_delta(
            v51_l_rows, v50_l_rows, method, method
        )

    learned_v51_contrasts = {}
    for label, left_method, right_method in (
        ("Alternating_minus_Full", "AlternatingConflictBenefit", "Full"),
        ("Alternating_minus_OneShot", "AlternatingConflictBenefit", "OneShotComm"),
        ("Alternating_minus_Shuffled", "AlternatingConflictBenefit", "ShuffledComm"),
        ("Full_minus_OneShot", "Full", "OneShotComm"),
        ("Full_minus_Shuffled", "Full", "ShuffledComm"),
        ("Full_minus_A0B0", "Full", "FrozenAnchorSelectionOnly"),
        ("Full_minus_NoPeer", "Full", "NoPeerFinalShield"),
    ):
        learned_v51_contrasts[label] = _paired_delta(
            v51_l_rows, v51_l_rows, left_method, right_method
        )

    output = {
        "schema_version": "SNSD-v5.0-v5.1-paired-analysis-v1",
        "inputs": {
            "v50_learned": str(Path(v50_learned_path)),
            "v51_learned": str(Path(v51_learned_path)),
        },
        "learned_version_deltas_v51_minus_v50": learned_version_deltas,
        "learned_v51_within_matrix_contrasts": learned_v51_contrasts,
        "learned_v51_factor_effects": _factor_stats(v51_l_rows),
        "learned_factor_effect_deltas_v51_minus_v50": _factor_delta_stats(
            v51_l_rows, v50_l_rows
        ),
        "source_contracts": {
            "v50_learned_complete": v50_learned.get("complete_matrix"),
            "v51_learned_complete": v51_learned.get("complete_matrix"),
            "v51_learned_formal_contract": v51_learned.get("formal_contract"),
        },
    }

    if v50_oracle_path and v51_oracle_path:
        v50_oracle, v50_o_rows = _load(v50_oracle_path)
        v51_oracle, v51_o_rows = _load(v51_oracle_path)
        oracle_version_deltas = {}
        for method in sorted(set(_method_names(v50_o_rows)) & set(_method_names(v51_o_rows))):
            oracle_version_deltas[method] = _paired_delta(
                v51_o_rows, v50_o_rows, method, method
            )
        output["inputs"].update(
            {
                "v50_oracle": str(Path(v50_oracle_path)),
                "v51_oracle": str(Path(v51_oracle_path)),
            }
        )
        output["oracle_version_deltas_v51_minus_v50"] = oracle_version_deltas
        output["source_contracts"].update(
            {
                "v50_oracle_complete": v50_oracle.get("complete_matrix"),
                "v51_oracle_complete": v51_oracle.get("complete_matrix"),
                "v51_oracle_formal_contract": v51_oracle.get("formal_contract"),
            }
        )

    if v51_confirmatory_path:
        confirmatory, confirmatory_rows = _load(v51_confirmatory_path)
        confirmatory_contrasts = {}
        for label, left_method, right_method in (
            ("Alternating_minus_Full", "AlternatingConflictBenefit", "Full"),
            ("Alternating_minus_SelectionOnly", "AlternatingConflictBenefit", "SelectionOnly"),
            ("Alternating_minus_OneShot", "AlternatingConflictBenefit", "OneShotComm"),
            ("Alternating_minus_Shuffled", "AlternatingConflictBenefit", "ShuffledComm"),
            ("Full_minus_OneShot", "Full", "OneShotComm"),
            ("Full_minus_Shuffled", "Full", "ShuffledComm"),
        ):
            confirmatory_contrasts[label] = _paired_delta(
                confirmatory_rows, confirmatory_rows, left_method, right_method
            )

        pooled_rows = _merge_disjoint_indexes(v51_l_rows, confirmatory_rows)
        pooled_contrasts = {}
        for label, left_method, right_method in (
            ("Alternating_minus_Full", "AlternatingConflictBenefit", "Full"),
            ("Alternating_minus_SelectionOnly", "AlternatingConflictBenefit", "SelectionOnly"),
            ("Alternating_minus_OneShot", "AlternatingConflictBenefit", "OneShotComm"),
            ("Alternating_minus_Shuffled", "AlternatingConflictBenefit", "ShuffledComm"),
            ("Full_minus_OneShot", "Full", "OneShotComm"),
            ("Full_minus_Shuffled", "Full", "ShuffledComm"),
        ):
            pooled_contrasts[label] = _paired_delta(
                pooled_rows, pooled_rows, left_method, right_method
            )

        output["inputs"]["v51_confirmatory"] = str(Path(v51_confirmatory_path))
        output["confirmatory_method_stats"] = _method_stats(confirmatory_rows)
        output["confirmatory_factor_effects"] = _factor_stats(confirmatory_rows)
        output["confirmatory_protocol_contrasts"] = confirmatory_contrasts
        output["pooled_16seed_method_stats"] = _method_stats(pooled_rows)
        output["pooled_16seed_factor_effects"] = _factor_stats(pooled_rows)
        output["pooled_16seed_protocol_contrasts"] = pooled_contrasts
        output["source_contracts"].update(
            {
                "v51_confirmatory_complete": confirmatory.get("complete_matrix"),
                "v51_confirmatory_formal_contract": confirmatory.get("formal_contract"),
                "v51_confirmatory_evaluation_mode": confirmatory.get("evaluation_mode"),
                "v51_confirmatory_oracle_mode": confirmatory.get("oracle_mode"),
                "v51_confirmatory_checkpoint_loaded_runs": confirmatory.get(
                    "checkpoint_loaded_runs"
                ),
            }
        )
    return output


def render_markdown(result):
    lines = [
        "# v5.0--v5.1 formal paired analysis",
        "",
        "All deltas are computed seed by seed. `+/-` is sample SD; the triplet is positive/tie/negative seeds.",
        "",
        "## Learned: v5.1 minus v5.0",
        "",
        "| Method | Paired gain delta | Sign p | Sign-flip mean p |",
        "|---|---:|---:|---:|",
    ]
    for method, stats in result["learned_version_deltas_v51_minus_v50"].items():
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} |".format(
                method,
                _fmt(stats),
                stats["two_sided_sign_p"],
                stats["two_sided_signflip_mean_p"],
            )
        )

    lines.extend(
        [
            "",
            "## Learned v5.1: factorial effects",
            "",
            "| Effect | Paired gain effect | Sign p | Sign-flip mean p |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, stats in result["learned_v51_factor_effects"].items():
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} |".format(
                label.replace("_", "\\_"),
                _fmt(stats),
                stats["two_sided_sign_p"],
                stats["two_sided_signflip_mean_p"],
            )
        )

    lines.extend(
        [
            "",
            "## Change in learned factorial effects: v5.1 minus v5.0",
            "",
            "| Effect | Paired effect delta | Sign p | Sign-flip mean p |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, stats in result["learned_factor_effect_deltas_v51_minus_v50"].items():
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} |".format(
                label.replace("_", "\\_"),
                _fmt(stats),
                stats["two_sided_sign_p"],
                stats["two_sided_signflip_mean_p"],
            )
        )

    lines.extend(
        [
            "",
            "## Learned v5.1: paired protocol contrasts",
            "",
            "| Contrast | Paired gain delta | Sign p | Sign-flip mean p |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, stats in result["learned_v51_within_matrix_contrasts"].items():
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} |".format(
                label.replace("_", "\\_"),
                _fmt(stats),
                stats["two_sided_sign_p"],
                stats["two_sided_signflip_mean_p"],
            )
        )

    if "oracle_version_deltas_v51_minus_v50" in result:
        lines.extend(
            [
                "",
                "## Analytic mechanism diagnostic: v5.1 minus v5.0",
                "",
                "| Method | Paired gain delta | Sign p | Sign-flip mean p |",
                "|---|---:|---:|---:|",
            ]
        )
        for method, stats in result["oracle_version_deltas_v51_minus_v50"].items():
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} |".format(
                    method,
                    _fmt(stats),
                    stats["two_sided_sign_p"],
                    stats["two_sided_signflip_mean_p"],
                )
            )

    if "confirmatory_method_stats" in result:
        lines.extend(
            [
                "",
                "## Fresh-seed confirmation specified before execution",
                "",
                "| Method | Gain mean +/- SD |",
                "|---|---:|",
            ]
        )
        for method, stats in result["confirmatory_method_stats"].items():
            lines.append("| {} | {:.1f} +/- {:.1f} |".format(method, stats["mean"], stats["sample_std"]))
        lines.extend(
            [
                "",
                "| Confirmatory contrast | Paired gain delta | Sign p | Sign-flip mean p |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, stats in result["confirmatory_protocol_contrasts"].items():
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} |".format(
                    label.replace("_", "\\_"),
                    _fmt(stats),
                    stats["two_sided_sign_p"],
                    stats["two_sided_signflip_mean_p"],
                )
            )
        lines.extend(
            [
                "",
                "## Descriptive pooled 16-seed view",
                "",
                "The pooled view combines the original and the fresh seed set specified before its execution. It is descriptive; the fresh set remains the confirmatory decision source.",
                "",
                "| Method | Gain mean +/- SD |",
                "|---|---:|",
            ]
        )
        for method, stats in result["pooled_16seed_method_stats"].items():
            lines.append("| {} | {:.1f} +/- {:.1f} |".format(method, stats["mean"], stats["sample_std"]))
        lines.extend(
            [
                "",
                "| Pooled contrast | Paired gain delta | Sign p | Sign-flip mean p |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, stats in result["pooled_16seed_protocol_contrasts"].items():
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} |".format(
                    label.replace("_", "\\_"),
                    _fmt(stats),
                    stats["two_sided_sign_p"],
                    stats["two_sided_signflip_mean_p"],
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The original eight seeds are appropriate for paired ablations but are too small for strong distributional claims.",
            "- Selecting the highest method from the original matrix is exploratory; the separately executed fresh seed set is the confirmatory check.",
            "- The analytic oracle is a mechanism diagnostic scaffold, not an upper bound and not directly comparable as a learned policy ranking.",
            "- Mean advantages with mixed seed signs must be reported together with the sign count and variability.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v50-learned", required=True)
    parser.add_argument("--v51-learned", required=True)
    parser.add_argument("--v50-oracle")
    parser.add_argument("--v51-oracle")
    parser.add_argument("--v51-confirmatory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    result = analyze(
        args.v50_learned,
        args.v51_learned,
        v50_oracle_path=args.v50_oracle,
        v51_oracle_path=args.v51_oracle,
        v51_confirmatory_path=args.v51_confirmatory,
    )
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
