"""Final channel-corrected learned TriGraph base_protocol runner.

The BenefitMessageOnly intervention carries the identifiable local-benefit
pressure and masks conflict/externality channels.  ConflictMessageOnly carries
only conflict pressure.  This keeps the two channel ablations semantically
disjoint while preserving identical static graph/local state.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .base_protocol import METHODS, ExperimentResult
from .checkpoint_policy_protocol import (
    CheckpointPolicyConfig,
    CheckpointPolicyProtocol,
)
from .messages import BoundaryMessage


class ValidatedPolicyProtocol(CheckpointPolicyProtocol):
    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-learned-run-v3.2",
                "benefit_message_channel": "identifiable_local_benefit_pressure",
                "conflict_message_channel": "identifiable_opportunity_cost_pressure",
                "message_channels_disjoint_in_single_channel_ablations": True,
            }
        )
        return fields

    def _message(
        self,
        generator: Any,
        pressure: Mapping[str, Mapping[int, Any]],
    ) -> BoundaryMessage:
        order = tuple(int(node) for node in generator.boundary)
        zeros = {node: 0.0 for node in order}
        mode = self.behavior.peer_pressure

        if mode == "none":
            selected_pressure = dict(zeros)
            conflict_message = dict(zeros)
            opportunity = dict(zeros)
            known = {node: False for node in order}
        elif mode == "benefit":
            selected_pressure = pressure["benefit"]
            conflict_message = dict(zeros)
            opportunity = dict(zeros)
            known = pressure["identifiable_benefit"]
        elif mode == "conflict":
            selected_pressure = pressure["conflict"]
            conflict_message = pressure["conflict"]
            opportunity = dict(zeros)
            known = pressure["identifiable_conflict"]
        else:
            selected_pressure = pressure["full"]
            conflict_message = pressure["conflict"]
            opportunity = pressure["opportunity"]
            known = {
                node: bool(
                    pressure["identifiable_full"][node]
                    and pressure["identifiable_conflict"][node]
                    and pressure["identifiable_opportunity"][node]
                )
                for node in order
            }

        uncertainty = {
            node: 0.0 if bool(known[node]) else 1.0 for node in order
        }
        return BoundaryMessage.from_maps(
            boundary_order=order,
            opportunity_cost=opportunity,
            pressure=selected_pressure,
            neighbor_recovery=opportunity,
            conflict_score=conflict_message,
            uncertainty=uncertainty,
        )


def run_validated_policy_protocol(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
    checkpoint: str = None,
    device: str = "cpu",
    allow_untrained_smoke: bool = False,
) -> ExperimentResult:
    return ValidatedPolicyProtocol(
        CheckpointPolicyConfig(
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
    ).run()


__all__ = [
    "METHODS",
    "CheckpointPolicyConfig",
    "ValidatedPolicyProtocol",
    "run_validated_policy_protocol",
]
