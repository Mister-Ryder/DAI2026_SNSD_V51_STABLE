#!/usr/bin/env python3
"""Build deterministic file hashes for the reorganized SNSD-v5.1 snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifest"
TRANSIENT_SUFFIXES = {
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
    ".xdv",
    ".pyc",
}


def is_included(path):
    rel = path.relative_to(ROOT)
    if rel.parts and rel.parts[0] == ".git":
        return False
    if rel.parts and rel.parts[0] == "manifest":
        return False
    if "__pycache__" in rel.parts:
        return False
    if (
        len(rel.parts) >= 3
        and rel.parts[:3]
        == ("SNSD_V51_ABLATIONS", "results", "formal")
    ):
        return True
    if (
        len(rel.parts) >= 3
        and rel.parts[:3]
        == ("SNSD_V51_FINAL", "checkpoints", "provenance")
    ):
        return True
    name = path.name.lower()
    return not any(name.endswith(suffix) for suffix in TRANSIENT_SUFFIXES)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(rel):
    return rel.parts[0] if rel.parts else "root"


def main():
    files = sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and is_included(path)
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    records = []
    for path in files:
        rel = path.relative_to(ROOT)
        records.append(
            {
                "path": rel.as_posix(),
                "category": category(rel),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    checkpoint = (
        ROOT
        / "SNSD_V51_FINAL"
        / "checkpoints"
        / "checkpoint_message_v4_5_strict.pt"
    )
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "snsd-v5.1-reorganized-stable-file-manifest-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root_name": ROOT.name,
        "projects": {
            "final": "SNSD_V51_FINAL",
            "ablations": "SNSD_V51_ABLATIONS",
            "history_archive": (
                "../snsd_v15/archive/"
                "DAI2026_SNSD_V51_STABLE_MIXED_SRC_20260802"
            ),
        },
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "checkpoint_bundled": checkpoint.is_file(),
        "checkpoint_expected_sha256": (
            "a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59"
        ),
        "files": records,
    }

    json_path = MANIFEST_DIR / "STABLE_FILE_MANIFEST.json"
    sums_path = MANIFEST_DIR / "SHA256SUMS.txt"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums_path.write_text(
        "".join(
            "%s  %s\n" % (record["sha256"], record["path"])
            for record in records
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "schema": payload["schema"],
                "file_count": len(records),
                "total_bytes": payload["total_bytes"],
                "checkpoint_bundled": payload["checkpoint_bundled"],
                "manifest": str(json_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
