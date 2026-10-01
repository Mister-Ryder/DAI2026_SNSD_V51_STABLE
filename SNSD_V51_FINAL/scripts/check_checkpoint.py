"""Strict-load and hash audit for a v1.5 TriGraph checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1] / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch

from snsd_v51.model import TriGraphConfig, TriGraphModel
from snsd_v51.training.warmstart_trainer import EXPECTED_PARAMETER_COUNT


def state_hash(state_dict):
    digest = hashlib.sha256()
    digest.update(b"SNSD-V15-OFFLINE-CHECKPOINT-v2\0")
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args(argv)
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model = TriGraphModel(TriGraphConfig())
    incompatible = model.load_state_dict(state, strict=True)
    parameters = sum(int(value.numel()) for value in model.parameters())
    if parameters != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("parameter count mismatch")
    digest = state_hash(model.state_dict())
    declared = payload.get("checkpoint_hash") if isinstance(payload, dict) else None
    if declared is not None and declared != digest:
        raise RuntimeError("checkpoint hash mismatch")
    print(
        json.dumps(
            {
                "strict_load": True,
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "parameter_count": parameters,
                "checkpoint_hash": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
