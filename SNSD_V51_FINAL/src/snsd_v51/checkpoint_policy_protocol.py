"""Confound-free learned v3 runner.

This additive revision keeps every v3 invariant while ensuring communication
ablations never erase static/public/local graph features.  Only NoGlobal masks
the full-graph encoder contribution; all other method differences are applied
to ``BoundaryMessage`` channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .base_protocol import METHODS, ExperimentResult
from .learned_policy_protocol import (
    LearnedPolicyConfig,
    LearnedPolicyProtocol,
)
from .model import FeatureMode
from .messages import BoundaryMessage


@dataclass(frozen=True)
class CheckpointPolicyConfig(LearnedPolicyConfig):
    @property
    def encoder_mode(self) -> FeatureMode:
        return (
            FeatureMode.NO_GLOBAL
            if self.method == "NoGlobalContext"
            else FeatureMode.FULL
        )


class _EncoderModePolicy:
    """Delegate policy calls while enforcing a graph-only encoder ablation."""

    def __init__(self, policy: Any, encoder_mode: FeatureMode) -> None:
        self._policy = policy
        self._encoder_mode = encoder_mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["mode"] = self._encoder_mode
        return self._policy.prepare(*args, **kwargs)


class CheckpointPolicyProtocol(LearnedPolicyProtocol):
    config: CheckpointPolicyConfig

    def __init__(self, config: CheckpointPolicyConfig) -> None:
        super().__init__(config)
        if self.policy is not None:
            self.policy = _EncoderModePolicy(self.policy, config.encoder_mode)

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-learned-run-v3.1",
                "encoder_feature_mode": self.config.encoder_mode.value,
                "static_state_paired_across_methods": True,
                "communication_ablation_scope": "BoundaryMessage-only",
                "no_global_ablation_scope": "full-encoder-output-only",
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
            uncertainty = dict(zeros)
        elif mode == "benefit":
            selected_pressure = dict(zeros)
            conflict_message = dict(zeros)
            opportunity = pressure["opportunity"]
            opportunity_known = pressure["identifiable_opportunity"]
            uncertainty = {
                node: 0.0 if bool(opportunity_known[node]) else 1.0
                for node in order
            }
        elif mode == "conflict":
            selected_pressure = pressure["conflict"]
            conflict_message = pressure["conflict"]
            opportunity = dict(zeros)
            conflict_known = pressure["identifiable_conflict"]
            uncertainty = {
                node: 0.0 if bool(conflict_known[node]) else 1.0
                for node in order
            }
        else:
            selected_pressure = pressure["full"]
            conflict_message = pressure["conflict"]
            opportunity = pressure["opportunity"]
            full_known = pressure["identifiable_full"]
            conflict_known = pressure["identifiable_conflict"]
            opportunity_known = pressure["identifiable_opportunity"]
            uncertainty = {
                node: 0.0
                if bool(
                    full_known[node]
                    and conflict_known[node]
                    and opportunity_known[node]
                )
                else 1.0
                for node in order
            }

        return BoundaryMessage.from_maps(
            boundary_order=order,
            opportunity_cost=opportunity,
            pressure=selected_pressure,
            neighbor_recovery=opportunity,
            conflict_score=conflict_message,
            uncertainty=uncertainty,
        )


def run_checkpoint_policy_protocol(
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
    return CheckpointPolicyProtocol(
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
    "CheckpointPolicyProtocol",
    "run_checkpoint_policy_protocol",
]
