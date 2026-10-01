"""Attach the strict learned-checkpoint contract to an audited v5.1 summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

FINAL_SRC = Path(__file__).resolve().parents[2] / "SNSD_V51_FINAL" / "src"
if str(FINAL_SRC) not in sys.path:
    sys.path.insert(0, str(FINAL_SRC))

import torch

from snsd_v51.model import TriGraphConfig, TriGraphModel


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(root, checkpoint, summary_path):
    root = Path(root).resolve()
    checkpoint = Path(checkpoint).resolve()
    summary_path = Path(summary_path).resolve()
    checkpoint_hash = _sha256(checkpoint)

    payload = torch.load(str(checkpoint), map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    graph_layers = TriGraphModel(TriGraphConfig())
    incompatible = graph_layers.load_state_dict(state, strict=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError("strict checkpoint load reported incompatible keys")
    parameter_count = sum(int(value.numel()) for value in graph_layers.parameters())

    rows = []
    for path in sorted(root.glob("seed*/*/summary.json")):
        with path.open("r", encoding="utf-8") as stream:
            rows.append(json.load(stream))
    if len(rows) != 64:
        raise RuntimeError("learned audit requires 64 complete run summaries")
    for row in rows:
        label = "seed={} method={}".format(row.get("seed"), row.get("method"))
        if row.get("oracle_mode") is not False:
            raise RuntimeError("oracle row detected in learned matrix: " + label)
        if row.get("checkpoint_loaded") is not True:
            raise RuntimeError("checkpoint not loaded: " + label)
        if row.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("checkpoint file hash mismatch: " + label)
        if int(row.get("online_query_calls", row.get("online_calls", -1))) != 288:
            raise RuntimeError("paid worker-query mismatch: " + label)
        if int(row.get("v51_unaccepted_ca_domain_injections", -1)) != 0:
            raise RuntimeError("unaccepted CA domain pollution: " + label)

    with summary_path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    contract = result.get("formal_contract", {})
    if result.get("complete_matrix") is not True:
        raise RuntimeError("base audited matrix is incomplete")
    if contract.get("semantic_schedule_violations"):
        raise RuntimeError("semantic schedule violations remain")

    result.update(
        {
            "schema_version": "SNSD-v1.5-v5.1-learned-formal-summary-v2",
            "evaluation_mode": "learned-validation-selected",
            "oracle_mode": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_loaded_runs": 64,
            "online_worker_calls_per_run": 288,
            "strict_checkpoint_audit": {
                "strict_load": True,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "parameter_count": int(parameter_count),
                "checkpoint_file_sha256": checkpoint_hash,
                "row_hash_matches": 64,
                "row_checkpoint_loaded": 64,
                "row_oracle_false": 64,
            },
        }
    )
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    result = finalize(args.root, args.checkpoint, args.summary)
    print(
        json.dumps(
            {
                "complete_matrix": result["complete_matrix"],
                "checkpoint_loaded_runs": result["checkpoint_loaded_runs"],
                "oracle_mode": result["oracle_mode"],
                "semantic_schedule_violations": len(
                    result["formal_contract"]["semantic_schedule_violations"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
