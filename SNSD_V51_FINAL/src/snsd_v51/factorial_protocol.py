"""Strict 2x2 communication-factorial runner for rolling v4.5.

The two manipulated coordination channels are:

* A: factor-pressure messages consumed by the next proposal step.
* B: the joint Max-Sum working selection consumed as the next proposal anchor.

All four factorial cells retain exact current-domain chain Max-Sum selection.
``NoPeerComm`` remains available only as an additional system baseline because
its independent selector plus feasibility shield changes more than A and B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base_protocol import ExperimentResult
from .rolling_protocol import ROLLING_METHODS, RollingExperimentConfig
from .barrier_protocol import BarrierCompleteProtocol


FROZEN_SELECTION_METHOD = "FrozenAnchorSelectionOnly"
FACTORIAL_METHODS = tuple(ROLLING_METHODS) + (FROZEN_SELECTION_METHOD,)
STRICT_FACTORIAL_CELLS = (
    "Full",
    "FrozenAnchorComm",
    "SelectionOnly",
    FROZEN_SELECTION_METHOD,
)


@dataclass(frozen=True)
class FactorialExperimentConfig(RollingExperimentConfig):
    """Rolling config with an unambiguous label for the A0B0 factorial cell."""

    frozen_anchor_selection_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.frozen_anchor_selection_only:
            if self.method != "SelectionOnly" or not self.frozen_anchor_comm:
                raise ValueError(
                    "FrozenAnchorSelectionOnly requires method=SelectionOnly "
                    "and frozen_anchor_comm=True"
                )

    @property
    def reported_method(self) -> str:
        if self.frozen_anchor_selection_only:
            return FROZEN_SELECTION_METHOD
        return super().reported_method


class FactorialNegotiationProtocol(BarrierCompleteProtocol):
    config: FactorialExperimentConfig

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        reported = self.config.reported_method
        pressure_enabled = bool(self.behavior.wave_a_message_pass)
        feedback_enabled = bool(
            not self.config.frozen_anchor_comm
            and self.behavior.final_selection != "independent"
        )
        is_factorial = reported in STRICT_FACTORIAL_CELLS
        fields.update(
            {
                "schema_version": "SNSD-v1.5-rolling-recurrent-run-v4.5-factorial",
                "strict_two_by_two_factorial": bool(is_factorial),
                "factorial_cell": reported if is_factorial else None,
                "factor_A_pressure_messages_enabled": pressure_enabled,
                "factor_B_joint_anchor_feedback_enabled": feedback_enabled,
                "factorial_selector": (
                    "exact_current_domain_chain_max_sum" if is_factorial else None
                ),
                "no_peer_is_additional_system_baseline": reported == "NoPeerComm",
                "no_peer_is_factorial_cell": False,
            }
        )
        return fields


def _factorial_config(
    data_path: str,
    output_dir: str,
    method: str,
    seed: int,
    smoke: bool,
    oracle: bool,
    checkpoint: Optional[str],
    device: str,
    allow_untrained_smoke: bool,
) -> FactorialExperimentConfig:
    if method not in FACTORIAL_METHODS:
        raise ValueError("unknown factorial method: {}".format(method))
    one_shot = method == "OneShotComm"
    frozen_selection = method == FROZEN_SELECTION_METHOD
    frozen = method == "FrozenAnchorComm" or frozen_selection
    if frozen_selection:
        base_method = "SelectionOnly"
    elif one_shot or method == "FrozenAnchorComm":
        base_method = "Full"
    else:
        base_method = method
    return FactorialExperimentConfig(
        data_path=data_path,
        output_dir=output_dir,
        method=base_method,
        seed=seed,
        smoke=smoke,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=allow_untrained_smoke,
        one_shot_comm=one_shot,
        frozen_anchor_comm=frozen,
        frozen_anchor_selection_only=frozen_selection,
    )


def run_factorial_protocol(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    allow_untrained_smoke: bool = False,
) -> ExperimentResult:
    config = _factorial_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        smoke=smoke,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=allow_untrained_smoke,
    )
    return FactorialNegotiationProtocol(config).run()


__all__ = [
    "FACTORIAL_METHODS",
    "STRICT_FACTORIAL_CELLS",
    "FROZEN_SELECTION_METHOD",
    "FactorialExperimentConfig",
    "FactorialNegotiationProtocol",
    "run_factorial_protocol",
]
