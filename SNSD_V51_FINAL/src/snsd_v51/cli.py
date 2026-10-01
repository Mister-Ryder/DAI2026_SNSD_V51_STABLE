"""Single public CLI for the frozen SNSD-v5.1 paper method."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .semantic_safeguard_protocol import (
    preflight_v51_full_c3,
    run_semantic_safeguard_experiment,
)

PRIMARY_METHOD = "AlternatingConflictBenefit"
NEGOTIATION_DEPTH = 12
MACRO_ROUNDS = 4
PACKET_TOP_K = 4
MICROCYCLES = 3
MAX_PACKET_TOP_K = 6
MAX_MICROCYCLES = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen learned SNSD-v5.1 Phased protocol. Ablation and "
            "Oracle switches intentionally live in SNSD_V51_ABLATIONS."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.preflight_only:
        result = preflight_v51_full_c3(
            data_path=args.data,
            method=PRIMARY_METHOD,
            seed=args.seed,
            packet_top_k=PACKET_TOP_K,
            microcycles=MICROCYCLES,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required for the frozen learned method")
    result = run_semantic_safeguard_experiment(
        data_path=args.data,
        output_dir=args.output,
        method=PRIMARY_METHOD,
        seed=args.seed,
        oracle=False,
        checkpoint=args.checkpoint,
        device=args.device,
        negotiation_depth=NEGOTIATION_DEPTH,
        macro_rounds=MACRO_ROUNDS,
        packet_top_k=PACKET_TOP_K,
        microcycles=MICROCYCLES,
        max_packet_top_k=MAX_PACKET_TOP_K,
        max_microcycles=MAX_MICROCYCLES,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "PRIMARY_METHOD",
    "NEGOTIATION_DEPTH",
    "MACRO_ROUNDS",
    "build_parser",
    "main",
]
