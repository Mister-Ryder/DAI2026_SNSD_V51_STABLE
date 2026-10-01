#!/usr/bin/env python3
"""Offline structural, training, evidence, and artifact audit for SNSD-v5.1."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "SNSD_V51_FINAL"
ABLATIONS = ROOT / "SNSD_V51_ABLATIONS"
FINAL_SRC = FINAL / "src"
ABLATION_SRC = ABLATIONS / "src"
RESULTS = ABLATIONS / "results" / "formal"
HISTORY = (
    ROOT.parent
    / "snsd_v15"
    / "archive"
    / "DAI2026_SNSD_V51_STABLE_MIXED_SRC_20260802"
)
EXPECTED_METHODS = (
    "Full",
    "FrozenAnchorComm",
    "SelectionOnly",
    "FrozenAnchorSelectionOnly",
    "AlternatingConflictBenefit",
    "OneShotComm",
    "ShuffledComm",
    "NoPeerFinalShield",
)
EXPECTED_CHECKPOINT_SHA256 = (
    "a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59"
)
EXPECTED_C3_SHA256 = (
    "ec95f50c11d800f051e218aa1e414df873ddd12e1f71ce911da3ba28adff647e"
)
EXPECTED_PARAMETERS = 456075


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, tol=1e-6):
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol),
        "expected %s, got %s" % (expected, actual),
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_python_sources():
    roots = (FINAL, ABLATIONS, ROOT / "tools")
    paths = sorted(
        {path.resolve() for base in roots for path in base.rglob("*.py")}
    )
    require(paths, "no Python sources found")
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return len(paths)


def audit_project_boundaries():
    require((FINAL / "run.py").is_file(), "final public entry missing")
    require((FINAL / "train_gnn.py").is_file(), "GNN training entry missing")
    require(
        (ABLATIONS / "run_ablation.py").is_file(),
        "ablation public entry missing",
    )
    require((HISTORY / "src").is_dir(), "historical source archive missing")
    require(not (ROOT / "src").exists(), "mixed root src directory still exists")
    require(not (ROOT / "data").exists(), "mixed root data directory still exists")
    require(
        not (ROOT / "results").exists(),
        "mixed root results directory still exists",
    )

    forbidden_names = []
    for pattern in ("experiment_v*.py", "run_v15*.py"):
        forbidden_names.extend(FINAL.rglob(pattern))
    require(
        not forbidden_names,
        "historical version files remain in final project: %s"
        % [str(path) for path in forbidden_names],
    )

    forbidden_references = ("snsd_v15", "snsd_v1.snsd", 'default="snsd_v1/runs"')
    violations = []
    for path in FINAL.rglob("*.py"):
        body = path.read_text(encoding="utf-8")
        for token in forbidden_references:
            if token in body:
                violations.append("%s: %s" % (path, token))
    require(
        not violations,
        "old package/path references remain in final source: %s" % violations,
    )

    require(
        not (ABLATIONS / "src" / "snsd_v51_ablations" / "model.py").exists(),
        "ablation project duplicates the final model",
    )
    require(
        not (ABLATIONS / "src" / "snsd_v51_ablations" / "training").exists(),
        "ablation project duplicates the training implementation",
    )


def audit_imports_and_contracts():
    for path in (FINAL_SRC, ABLATION_SRC):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    final_cli = importlib.import_module("snsd_v51.cli")
    training_cli = importlib.import_module("snsd_v51.training_cli")
    importlib.import_module("snsd_v51.training.gnn_trainer")
    ablation_runner = importlib.import_module("snsd_v51_ablations.runner")
    model_module = importlib.import_module("snsd_v51.model")

    require(
        final_cli.PRIMARY_METHOD == "AlternatingConflictBenefit",
        "final method is not locked to Phased",
    )
    require(final_cli.NEGOTIATION_DEPTH == 12, "final depth changed")
    require(final_cli.MACRO_ROUNDS == 4, "final macro rounds changed")

    final_destinations = {
        action.dest for action in final_cli.build_parser()._actions
    }
    require("method" not in final_destinations, "final CLI exposes --method")
    require("oracle" not in final_destinations, "final CLI exposes --oracle")

    method_action = next(
        action
        for action in ablation_runner.build_parser()._actions
        if action.dest == "method"
    )
    require(
        tuple(method_action.choices) == EXPECTED_METHODS,
        "ablation method contract changed",
    )
    ablation_destinations = {
        action.dest for action in ablation_runner.build_parser()._actions
    }
    require("oracle" in ablation_destinations, "ablation Oracle switch missing")

    training_destinations = {
        action.dest for action in training_cli.build_parser()._actions
    }
    for field in (
        "run_dir",
        "train_seeds",
        "validation_seeds",
        "test_seeds",
        "negotiation_steps",
        "epochs",
        "initial_checkpoint",
    ):
        require(field in training_destinations, "training CLI missing %s" % field)

    model = model_module.TriGraphModel(model_module.TriGraphConfig())
    parameters = sum(int(value.numel()) for value in model.parameters())
    require(parameters == EXPECTED_PARAMETERS, "model parameter count changed")
    return parameters


def audit_training_closure():
    required = (
        "train_warmstart.py",
        "train_message_warmstart.py",
        "train_gnn.py",
        "src/snsd_v51/graph_layers.py",
        "src/snsd_v51/model.py",
        "src/snsd_v51/runtime.py",
        "src/snsd_v51/training/offline_dataset.py",
        "src/snsd_v51/training/message_teacher.py",
        "src/snsd_v51/training/rolling_teacher.py",
        "src/snsd_v51/training/strict_teacher.py",
        "src/snsd_v51/training/warmstart_trainer.py",
        "src/snsd_v51/training/message_warmstart_trainer.py",
        "src/snsd_v51/training/gnn_trainer.py",
        "scripts/train_gnn_8gpu.sh",
        "scripts/reproduce_selected_checkpoint.sh",
        "scripts/select_checkpoint.py",
        "scripts/check_checkpoint.py",
    )
    missing = [name for name in required if not (FINAL / name).is_file()]
    require(not missing, "incomplete GNN training closure: %s" % missing)
    return len(required)


def audit_formal_evidence():
    expected_json = (
        "V51_ORACLE_V2_FORMAL_AUDITED_SUMMARY.json",
        "V51_LEARNED_V2_FORMAL_AUDITED_SUMMARY.json",
        "V51_LEARNED_CONFIRMATORY_FRESH_SEEDS101_808_SUMMARY.json",
        "v51_final_formal_paired_analysis.json",
    )
    for name in expected_json:
        path = RESULTS / name
        require(path.is_file(), "formal evidence missing: %s" % name)
        json.loads(path.read_text(encoding="utf-8"))

    paired = json.loads(
        (RESULTS / "v51_final_formal_paired_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    contracts = paired["source_contracts"]
    for prefix in ("v51_learned", "v51_confirmatory", "v51_oracle"):
        require(contracts["%s_complete" % prefix] is True, "%s incomplete" % prefix)
        contract = contracts["%s_formal_contract" % prefix]
        require(contract["complete_c3_non_smoke"] is True, "%s is smoke" % prefix)
        require(contract["paid_worker_calls_per_run"] == 288, "budget mismatch")
        require(contract["same_state_paired_c0_ca"] is True, "not same-state")
        for field in (
            "missing_cells",
            "unexpected_cells",
            "duplicate_cells",
            "budget_violations",
            "additional_worker_call_violations",
            "semantic_schedule_violations",
            "unaccepted_ca_pollution_violations",
        ):
            require(contract[field] == [], "%s: nonempty %s" % (prefix, field))

    phased = paired["pooled_16seed_method_stats"]["AlternatingConflictBenefit"]
    close(phased["mean"], 35884.625)
    close(phased["sample_std"], 6026.438106377597)

    b_a = paired["pooled_16seed_factor_effects"]["B_given_A"]
    close(b_a["mean"], 20214.0625)
    require(b_a["positive"] == 16 and b_a["negative"] == 0, "B|A signs")

    b_no_a = paired["pooled_16seed_factor_effects"]["B_without_A"]
    close(b_no_a["mean"], 16424.1875)
    require(
        b_no_a["positive"] == 16 and b_no_a["negative"] == 0,
        "B|not A signs",
    )

    phased_sel = paired["pooled_16seed_protocol_contrasts"][
        "Alternating_minus_SelectionOnly"
    ]
    close(phased_sel["mean"], 5033.5625)
    require(phased_sel["positive"] == 14, "Phased-SelectionOnly signs")

    phased_one = paired["pooled_16seed_protocol_contrasts"][
        "Alternating_minus_OneShot"
    ]
    close(phased_one["mean"], 4810.6875)
    require(phased_one["positive"] == 14, "Phased-OneShot signs")


def audit_artifacts():
    c3 = FINAL / "data" / "C3.csv"
    require(c3.stat().st_size == 4682663, "C3 byte size changed")
    require(file_sha256(c3) == EXPECTED_C3_SHA256, "C3 SHA-256 changed")

    method = ROOT / "paper" / "SNSD_V51_METHOD_AND_EVIDENCE_ZH.md"
    body = method.read_text(encoding="utf-8")
    for token in ("20,214.1", "16,424.2", "5,033.6", "4,810.7", "16/16"):
        require(token in body, "paper evidence token missing: %s" % token)

    pdf = ROOT / "output" / "pdf" / "SNSD_V51_METHOD_ZH_ACM_SIGCONF.pdf"
    require(pdf.read_bytes().startswith(b"%PDF-"), "final PDF missing or invalid")

    checkpoint = FINAL / "checkpoints" / "checkpoint_message_v4_5_strict.pt"
    if checkpoint.is_file():
        require(
            file_sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256,
            "checkpoint SHA-256 mismatch",
        )
        return True
    return False


def audit_training_provenance():
    provenance = FINAL / "checkpoints" / "provenance"
    selected = provenance / "formal_selected_run"
    warmstart = provenance / "formal_warmstart_message_v3"
    warmups = provenance / "warmup_teacher_strict"
    training_root = provenance / "formal_training_root"
    environment = provenance / "server_environment"
    launch_root = (
        ABLATIONS
        / "results"
        / "formal"
        / "provenance"
        / "launch_manifests"
    )

    selected_files = (
        "checkpoint_message_v4_5_strict.pt",
        "model_manifest_message_v4_5_strict.json",
        "training_history_message_v4_5_strict.jsonl",
        "dataset_receipt.json",
        "split_manifest.json",
        "prequery_candidates.jsonl",
        "teacher_labels.jsonl",
        "arc_pressures.jsonl",
    )
    warmstart_files = (
        "checkpoint_message_v3.pt",
        "model_manifest_message_v3.json",
        "training_history_message_v3.jsonl",
        "crossfit_message_teacher_manifest.json",
        "dataset_receipt.json",
        "split_manifest.json",
        "prequery_candidates.jsonl",
        "teacher_labels.jsonl",
        "arc_pressures.jsonl",
    )
    warmup_files = tuple(
        warmups / ("strict_depth12_q7_seed%s" % seed) / name
        for seed in (11, 22, 33, 44, 55, 66, 77, 88)
        for name in ("warmup_manifest.json", "teacher_config.json")
    )
    selection_files = (
        "selected_fast_main.json",
        "fast_sweep_manifest.tsv",
        "fast_remaining_manifest.tsv",
        "fast_corrected_additions_manifest.tsv",
        "warmstart_e2e_lr3e4_eps05_d12_e12_fast.log",
    )
    environment_files = (
        "conda_explicit.txt",
        "pip_freeze.txt",
        "system_and_gpu.txt",
    )
    launch_files = (
        "learned_original_launch_manifest.tsv",
        "learned_fresh_launch_manifest.tsv",
        "oracle_launch_manifest.tsv",
    )

    available = {
        "formal_selected_run": all(
            (selected / name).is_file() for name in selected_files
        ),
        "formal_v3_warmstart": all(
            (warmstart / name).is_file() for name in warmstart_files
        ),
        "formal_eight_warmups": all(path.is_file() for path in warmup_files),
        "formal_selection_receipts": all(
            (training_root / name).is_file() for name in selection_files
        ),
        "server_environment_snapshot": all(
            (environment / name).is_file() for name in environment_files
        ),
        "formal_experiment_launch_manifests": all(
            (launch_root / name).is_file() for name in launch_files
        ),
    }
    prelaunch_source_snapshot = (
        provenance / "prelaunch_training_source_snapshot"
    ).is_dir()
    return available, prelaunch_source_snapshot


def main():
    source_count = audit_python_sources()
    audit_project_boundaries()
    parameters = audit_imports_and_contracts()
    training_files = audit_training_closure()
    audit_formal_evidence()
    checkpoint_bundled = audit_artifacts()
    provenance, prelaunch_source_snapshot = audit_training_provenance()
    missing_receipts = sorted(
        name for name, available in provenance.items() if not available
    )

    if not checkpoint_bundled:
        status = "PASS_WITH_BINARY_PROVENANCE_GAP"
    elif missing_receipts:
        status = "PASS_WITH_TRAINING_PROVENANCE_GAP"
    elif not prelaunch_source_snapshot:
        status = "PASS_WITH_TRAINING_SOURCE_SNAPSHOT_GAP"
    else:
        status = "PASS"

    print(
        json.dumps(
            {
                "status": status,
                "final_public_method": "AlternatingConflictBenefit",
                "final_cli_exposes_method_or_oracle": False,
                "ablation_methods": list(EXPECTED_METHODS),
                "python_sources": source_count,
                "gnn_training_files": training_files,
                "model_parameters": parameters,
                "formal_matrices": 3,
                "formal_runs": 192,
                "paid_worker_calls": 55296,
                "history_archive_present": HISTORY.is_dir(),
                "checkpoint_bundled": checkpoint_bundled,
                "checkpoint_expected_sha256": EXPECTED_CHECKPOINT_SHA256,
                "missing_training_provenance_receipts": missing_receipts,
                "prelaunch_training_source_snapshot_bundled": (
                    prelaunch_source_snapshot
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
