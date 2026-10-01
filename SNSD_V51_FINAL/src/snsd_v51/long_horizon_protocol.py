"""Configurable long-horizon recurrent negotiation for SNSD v1.5.

The default schedule deliberately keeps the agents negotiating after the
initial solution: four macro rounds, twelve proposal/factor steps per macro,
six agents and one paid query per agent-step.  This is exactly 288 online
worker calls.  There is no convergence or no-improvement early stopping.

``AlternatingConflictBenefit`` is a channel-scheduling treatment.  Step 0 of
each macro observes public/static state only.  Starting at step 1, odd steps
consume conflict-only messages and even steps consume benefit-only messages.
The ordinary ``Full`` treatment is unchanged and consumes all registered
message channels at every counterproposal step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from .base_protocol import ExperimentResult, MethodBehavior
from .factorial_protocol import FactorialExperimentConfig
from .strict_protocol import (
    STRICT_METHODS,
    StrictNegotiationProtocol,
)


ALTERNATING_METHOD = "AlternatingConflictBenefit"
LONG_NEGOTIATION_DEPTHS = (6, 12, 18)
LONG_METHODS = tuple(STRICT_METHODS) + (ALTERNATING_METHOD,)


@dataclass(frozen=True)
class LongNegotiationConfig(FactorialExperimentConfig):
    """Fixed-budget long negotiation schedule.

    Field names intentionally differ from inherited read-only properties.
    This keeps compatibility with every earlier config while making the two
    base_protocol controls explicit at the CLI boundary.
    """

    negotiation_depth: int = 12
    macro_round_count: int = 4
    alternating_conflict_benefit: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        depth = int(self.negotiation_depth)
        macros = int(self.macro_round_count)
        if depth not in LONG_NEGOTIATION_DEPTHS:
            raise ValueError(
                "negotiation_depth must be one of {}".format(
                    LONG_NEGOTIATION_DEPTHS
                )
            )
        if macros <= 0:
            raise ValueError("macro_round_count must be positive")
        if self.alternating_conflict_benefit and self.method != "Full":
            raise ValueError(
                "AlternatingConflictBenefit must use Full as its base method"
            )
        object.__setattr__(self, "negotiation_depth", depth)
        object.__setattr__(self, "macro_round_count", macros)

    @property
    def negotiation_steps(self) -> int:
        return int(self.negotiation_depth)

    @property
    def macro_rounds(self) -> int:
        return int(self.macro_round_count)

    @property
    def reported_method(self) -> str:
        if self.alternating_conflict_benefit:
            return ALTERNATING_METHOD
        return super().reported_method


def _phase_for_method_step(method: str, step: int) -> str:
    """Return the auditable message phase consumed by a proposal step."""

    step = int(step)
    if step < 0:
        raise ValueError("negotiation step cannot be negative")
    if step == 0:
        return "public_initial"
    if method == ALTERNATING_METHOD:
        return "conflict_only" if step % 2 == 1 else "benefit_only"
    if method == "Full" or method == "FrozenAnchorComm":
        return "full_conflict_and_benefit"
    if method == "OneShotComm":
        return "full_conflict_and_benefit" if step == 1 else "strict_zero"
    if method == "ConflictMessageOnly":
        return "conflict_only"
    if method == "BenefitMessageOnly":
        return "benefit_only"
    if method == "ShuffledComm":
        return "shuffled_full"
    if method == "NoGlobalContext":
        return "full_without_global_context"
    return "strict_zero"


class LongHorizonProtocol(StrictNegotiationProtocol):
    """Strict v4.5 runner with a longer, configurable fixed schedule."""

    config: LongNegotiationConfig

    def __init__(self, config: LongNegotiationConfig) -> None:
        super().__init__(config)
        self._active_communication_phase = "not_started"
        self._observed_communication_phases = []  # type: List[Dict[str, Any]]

    @staticmethod
    def _alternating_behavior(
        behavior: MethodBehavior, phase: str
    ) -> MethodBehavior:
        if phase == "conflict_only":
            return replace(behavior, peer_pressure="conflict")
        if phase == "benefit_only":
            return replace(behavior, peer_pressure="benefit")
        return behavior

    def _pressure_for_step(self, step: int, pressure_maps: Any):
        pressure, mode = super()._pressure_for_step(step, pressure_maps)
        phase = _phase_for_method_step(self.config.reported_method, int(step))
        self._active_communication_phase = phase
        if self.config.alternating_conflict_benefit and int(step) > 0:
            # This is an audit label only.  The actual channel mask is applied
            # to both proposal generation and policy inference in _query_wave.
            mode = phase
        return pressure, mode

    def _query_wave(self, *args: Any, **kwargs: Any):
        if not self.config.alternating_conflict_benefit:
            return super()._query_wave(*args, **kwargs)
        original = self.behavior
        self.behavior = self._alternating_behavior(
            original, self._active_communication_phase
        )
        try:
            return super()._query_wave(*args, **kwargs)
        finally:
            # Full must remain Full outside the query that consumes the
            # scheduled single-channel message (notably for factor counting).
            self.behavior = original

    def _event(self, event_type: str, **payload: Any) -> None:
        if event_type in (
            "negotiation_step_start",
            "proposal_pool",
            "worker_query",
            "solver_compliance",
            "negotiation_step",
        ):
            payload["communication_phase"] = self._active_communication_phase
            payload["message_phase"] = self._active_communication_phase
        if event_type == "negotiation_step":
            self._observed_communication_phases.append(
                {
                    "macro_round": int(payload["macro_round"]),
                    "negotiation_step": int(payload["negotiation_step"]),
                    "phase": self._active_communication_phase,
                }
            )
        super()._event(event_type, **payload)

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        expected = (
            self.config.macro_rounds
            * self.config.negotiation_steps
            * self.config.num_agents
        )
        fields.update(
            {
                "schema_version": "SNSD-v1.5-long-negotiation-run-v4.6",
                "negotiation_steps_per_macro": self.config.negotiation_steps,
                "macro_rounds": self.config.macro_rounds,
                "fixed_schedule_no_early_stopping": True,
                "configured_online_query_calls": expected,
                "alternating_conflict_benefit": bool(
                    self.config.alternating_conflict_benefit
                ),
                "alternating_phase_rule": (
                    "step0=public_initial; odd=conflict_only; "
                    "even=benefit_only"
                    if self.config.alternating_conflict_benefit
                    else None
                ),
                "full_method_uses_simultaneous_channels": (
                    self.config.reported_method == "Full"
                ),
                "communication_phase_trace": list(
                    self._observed_communication_phases
                ),
            }
        )
        return fields


def _long_config(
    data_path: str,
    output_dir: str,
    method: str,
    seed: int,
    smoke: bool,
    oracle: bool,
    checkpoint: Optional[str],
    device: str,
    allow_untrained_smoke: bool,
    negotiation_depth: int,
    macro_rounds: int,
) -> LongNegotiationConfig:
    if method not in LONG_METHODS:
        raise ValueError("unknown long-negotiation method: {}".format(method))

    alternating = method == ALTERNATING_METHOD
    one_shot = method == "OneShotComm"
    frozen_selection = method == "FrozenAnchorSelectionOnly"
    frozen = method == "FrozenAnchorComm" or frozen_selection
    if alternating:
        base_method = "Full"
    elif frozen_selection:
        base_method = "SelectionOnly"
    elif one_shot or method == "FrozenAnchorComm":
        base_method = "Full"
    else:
        base_method = method

    return LongNegotiationConfig(
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
        negotiation_depth=negotiation_depth,
        macro_round_count=macro_rounds,
        alternating_conflict_benefit=alternating,
    )


def run_long_negotiation_experiment(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    allow_untrained_smoke: bool = False,
    negotiation_depth: int = 12,
    macro_rounds: int = 4,
) -> ExperimentResult:
    config = _long_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        smoke=smoke,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=allow_untrained_smoke,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
    )
    return LongHorizonProtocol(config).run()


__all__ = [
    "ALTERNATING_METHOD",
    "LONG_METHODS",
    "LONG_NEGOTIATION_DEPTHS",
    "LongNegotiationConfig",
    "LongHorizonProtocol",
    "run_long_negotiation_experiment",
]
