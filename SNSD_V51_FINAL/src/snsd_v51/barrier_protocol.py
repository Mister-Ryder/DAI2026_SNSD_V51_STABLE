"""Complete v4.5: NoPeer barriers plus split coordination accounting."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base_protocol import ExperimentResult
from .rolling_protocol import ROLLING_METHODS, RollingExperimentConfig
from .finalized_rolling_protocol import FinalizedRollingProtocol


class BarrierCompleteProtocol(FinalizedRollingProtocol):
    def __init__(self, config: RollingExperimentConfig) -> None:
        super().__init__(config)
        self.joint_anchor_feedback_cycles = 0
        self.joint_anchor_feedback_payload_scalars = 0
        self._last_pressure_payload_scalars = 0

    def _joint_feedback_consumed(self, step: int) -> bool:
        return bool(
            int(step) < self.config.negotiation_steps - 1
            and not self.config.frozen_anchor_comm
            and self.behavior.final_selection != "independent"
        )

    def _factor_pass(self, *args: Any, **kwargs: Any):
        before = int(self.message_payload_scalars)
        result = super()._factor_pass(*args, **kwargs)
        self._last_pressure_payload_scalars = int(self.message_payload_scalars) - before
        return result

    def _event(self, event_type: str, **payload: Any) -> None:
        if event_type == "negotiation_step":
            step = int(payload["negotiation_step"])
            feedback = self._joint_feedback_consumed(step)
            feedback_payload = self.config.num_agents if feedback else 0
            if feedback:
                self.joint_anchor_feedback_cycles += 1
                self.joint_anchor_feedback_payload_scalars += feedback_payload
            payload["pressure_message_consumed"] = bool(payload.get("message_pass"))
            payload["pressure_message_payload_scalars"] = int(
                self._last_pressure_payload_scalars
            )
            payload["joint_anchor_feedback_consumed"] = feedback
            payload["joint_anchor_feedback_payload_scalars"] = feedback_payload
        super()._event(event_type, **payload)

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-rolling-recurrent-run-v4.5-complete",
                "pressure_message_cycles": int(self.message_passes),
                "pressure_message_payload_scalars": int(self.message_payload_scalars),
                "pressure_message_payload_bytes_float64": 8
                * int(self.message_payload_scalars),
                "joint_anchor_feedback_cycles": int(
                    self.joint_anchor_feedback_cycles
                ),
                "joint_anchor_feedback_payload_scalars": int(
                    self.joint_anchor_feedback_payload_scalars
                ),
                "joint_anchor_feedback_payload_bytes_int64": 8
                * int(self.joint_anchor_feedback_payload_scalars),
                "total_dynamic_coordination_cycles": int(self.message_passes)
                + int(self.joint_anchor_feedback_cycles),
                "message_passes_semantics": "pressure_message_cycles_only",
            }
        )
        return fields


def run_barrier_protocol(
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
    if method not in ROLLING_METHODS:
        raise ValueError("unknown rolling method: {}".format(method))
    one_shot = method == "OneShotComm"
    frozen = method == "FrozenAnchorComm"
    base_method = "Full" if one_shot or frozen else method
    return BarrierCompleteProtocol(
        RollingExperimentConfig(
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
        )
    ).run()


__all__ = [
    "ROLLING_METHODS",
    "RollingExperimentConfig",
    "BarrierCompleteProtocol",
    "run_barrier_protocol",
]
