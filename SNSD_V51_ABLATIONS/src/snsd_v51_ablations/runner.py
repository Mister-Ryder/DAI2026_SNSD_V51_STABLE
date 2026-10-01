"""CLI exposing only the paper's learned/oracle comparison matrix."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from snsd_v51.semantic_safeguard_protocol import (
    FORMAL_METHODS,
    preflight_v51_full_c3,
    run_semantic_safeguard_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one SNSD-v5.1 ablation cell")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=FORMAL_METHODS, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.oracle and args.checkpoint:
        parser.error("Oracle cells must not receive a learned checkpoint")
    if not args.oracle and not args.preflight_only and not args.checkpoint:
        parser.error("learned cells require --checkpoint")
    if args.preflight_only:
        result = preflight_v51_full_c3(
            data_path=args.data,
            method=args.method,
            seed=args.seed,
            packet_top_k=4,
            microcycles=3,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    result = run_semantic_safeguard_experiment(
        data_path=args.data,
        output_dir=args.output,
        method=args.method,
        seed=args.seed,
        oracle=args.oracle,
        checkpoint=args.checkpoint,
        device=args.device,
        negotiation_depth=12,
        macro_rounds=4,
        packet_top_k=4,
        microcycles=3,
        max_packet_top_k=6,
        max_microcycles=4,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0
