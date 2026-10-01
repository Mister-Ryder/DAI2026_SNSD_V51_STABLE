"""Wait for the cached strict sweep and select one validation-only main model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from snsd_v51.model import TriGraphConfig, TriGraphModel
from snsd_v51.training.warmstart_trainer import _state_dict_hash


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    main = [row for row in rows if row.get("role") == "main_candidate"]
    if len(main) != 8:
        raise RuntimeError("expected 8 E2E main candidates, found %d" % len(main))
    if any(row.get("heads_only") != "0" for row in main):
        raise RuntimeError("heads-only row entered the E2E main candidate set")
    if any(row.get("zero_target_semantics") != "intrinsic" for row in main):
        raise RuntimeError("non-intrinsic row entered the E2E main candidate set")
    return main


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _strict_load(checkpoint_path: Path) -> Dict[str, Any]:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("strict checkpoint payload has no state_dict")
    config = TriGraphConfig(**dict(payload.get("model_config", {})))
    model = TriGraphModel(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    observed_hash = _state_dict_hash(model.state_dict())
    recorded_hash = str(payload.get("checkpoint_hash", ""))
    if observed_hash != recorded_hash:
        raise RuntimeError("strict checkpoint state hash mismatch")
    trained_depth = int(payload.get("trained_negotiation_steps", -1))
    if trained_depth != 12:
        raise RuntimeError("selected checkpoint depth is %d, expected 12" % trained_depth)
    observed_parameters = sum(int(value.numel()) for value in model.parameters())
    recorded_parameters = int(payload.get("parameter_count", -1))
    if observed_parameters != recorded_parameters:
        raise RuntimeError("strict checkpoint parameter-count mismatch")
    return {
        "strict_load_verified": True,
        "checkpoint_hash": observed_hash,
        "trained_negotiation_steps": trained_depth,
        "parameter_count": observed_parameters,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise RuntimeError("refusing to overwrite %s" % output_path)
    rows = _load_rows(manifest_path)
    last_complete = -1
    while True:
        available = []
        for row in rows:
            run_dir = Path(row["output"]).resolve()
            model_manifest = run_dir / "model_manifest_message_v4_5_strict.json"
            checkpoint = run_dir / "checkpoint_message_v4_5_strict.pt"
            if model_manifest.exists() and checkpoint.exists():
                available.append((row, model_manifest, checkpoint))
        if len(available) != last_complete:
            print(
                json.dumps(
                    {
                        "event": "fast_selector_wait",
                        "complete": len(available),
                        "required": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_complete = len(available)
        if len(available) == len(rows):
            break
        time.sleep(max(1.0, float(args.poll_seconds)))

    candidates = []
    for row, model_manifest_path, checkpoint_path in available:
        with model_manifest_path.open("r", encoding="utf-8") as stream:
            model_manifest = json.load(stream)
        validation_loss = float(model_manifest["best_validation_loss"])
        candidates.append(
            {
                "trial": row["trial"],
                "validation_loss": validation_loss,
                "best_epoch": int(model_manifest["best_epoch"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_file_sha256": _file_digest(checkpoint_path),
                "model_manifest": str(model_manifest_path),
                "model_manifest_sha256": _file_digest(model_manifest_path),
            }
        )
    candidates.sort(key=lambda item: (item["validation_loss"], item["trial"]))
    selected = candidates[0]
    strict_load = _strict_load(Path(selected["checkpoint"]))
    result = {
        "schema_version": "SNSD-v1.5-fast-validation-only-selection-v1",
        "selection_metric": "best_validation_loss",
        "selection_direction": "minimize",
        "test_metrics_used_for_selection": False,
        "heads_only_eligible": False,
        "candidate_count": len(candidates),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _file_digest(manifest_path),
        "selected_trial": selected["trial"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_validation_loss": selected["validation_loss"],
        "selected_best_epoch": selected["best_epoch"],
        "strict_load": strict_load,
        "candidates": candidates,
    }
    _atomic_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
