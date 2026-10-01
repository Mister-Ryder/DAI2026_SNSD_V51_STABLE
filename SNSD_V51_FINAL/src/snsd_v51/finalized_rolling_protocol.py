"""Final v4.5 NoPeer barrier with separate proposal/factor/public anchors.

This additive file supersedes the initial v4.5 draft, which correctly froze
the policy mask but still passed potentially conflicting local working anchors
to factor incumbent validation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base_protocol import ExperimentResult
from .rolling_protocol import ROLLING_METHODS, RollingExperimentConfig
from .public_incumbent_protocol import PublicIncumbentProtocol


class FinalizedRollingProtocol(PublicIncumbentProtocol):
    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-rolling-recurrent-run-v4.5-final",
                "no_peer_anchor_separation": {
                    "proposal_anchor": "agent-local-working-record",
                    "factor_feasible_anchor": "macro-accepted-incumbent",
                    "policy_public_incumbent": "macro-accepted-incumbent",
                },
            }
        )
        return fields

    def _event(self, event_type: str, **payload: Any) -> None:
        if (
            self.behavior.final_selection == "independent"
            and event_type in ("negotiation_step_start", "negotiation_step")
            and self._macro_public_incumbent is not None
        ):
            proposal = list(payload.get("anchor_signature_hashes", ()))
            feasible = [
                self._macro_public_incumbent[i].candidate_id
                for i in range(self.config.num_agents)
            ]
            payload["proposal_anchor_signature_hashes"] = proposal
            payload["factor_feasible_anchor_signature_hashes"] = feasible
            payload["policy_public_incumbent_signature_hashes"] = feasible
        super()._event(event_type, **payload)

    def _factor_pass(self, records_by_agent, incumbent_ids, *args, **kwargs):
        require_incumbent = kwargs.get("require_incumbent", True)
        if (
            self.behavior.final_selection == "independent"
            and self._macro_public_incumbent is not None
            and require_incumbent
        ):
            incumbent_ids = {
                agent_id: self._macro_public_incumbent[agent_id].candidate_id
                for agent_id in range(self.config.num_agents)
            }
        return super()._factor_pass(
            records_by_agent, incumbent_ids, *args, **kwargs
        )


def run_finalized_rolling_protocol(
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
    return FinalizedRollingProtocol(
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
    "FinalizedRollingProtocol",
    "run_finalized_rolling_protocol",
]
