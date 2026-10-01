"""NoPeer information-barrier correction for rolling negotiation v4.5.

NoPeer agents may roll their own local proposal anchors, but the TriGraph
public/global accepted-incumbent mask is frozen at the macro-start accepted
solution.  Other agents' temporary local anchors therefore cannot become an
implicit communication channel through the full/local/boundary encoders.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .base_protocol import ExperimentResult
from .rolling_protocol import (
    ROLLING_METHODS,
    RollingExperimentConfig,
    RollingNegotiationProtocol,
    _joint_hash,
)


class _PublicIncumbentPolicy:
    def __init__(self, policy: Any, public_incumbent: Mapping[int, Any]) -> None:
        self._policy = policy
        self._public_incumbent = dict(public_incumbent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["incumbent"] = self._public_incumbent
        return self._policy.prepare(*args, **kwargs)


class PublicIncumbentProtocol(RollingNegotiationProtocol):
    def __init__(self, config: RollingExperimentConfig) -> None:
        super().__init__(config)
        self._macro_public_incumbent = None
        self._active_public_incumbent_hash = None
        self._active_public_incumbent_signatures = None

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-rolling-recurrent-run-v4.5",
                "no_peer_global_incumbent_source": "macro-accepted-incumbent",
                "no_peer_local_proposal_anchor": "agent-local-working-record",
                "no_peer_dynamic_message_channels": "strict-zero",
                "no_peer_cross_agent_working_anchor_visible": False,
            }
        )
        return fields

    def _event(self, event_type: str, **payload: Any) -> None:
        if (
            self.behavior.final_selection == "independent"
            and event_type in ("proposal_pool", "worker_query", "solver_compliance")
            and self._active_public_incumbent_hash is not None
        ):
            payload["policy_public_incumbent_hash"] = self._active_public_incumbent_hash
            payload["policy_public_incumbent_signature_hashes"] = list(
                self._active_public_incumbent_signatures or ()
            )
        super()._event(event_type, **payload)

    def _query_wave(self, *args: Any, **kwargs: Any):
        # Positional contract inherited from v3:
        # wave, engine_round_context, generators, engines, anchor_records, ...
        if len(args) >= 5:
            wave = args[0]
            context = int(args[1])
            anchor_records = args[4]
        else:
            wave = kwargs["wave"]
            context = int(kwargs["macro_round"])
            anchor_records = kwargs["incumbent_records"]

        if self.behavior.final_selection != "independent":
            return super()._query_wave(*args, **kwargs)

        if wave == "A" or context % self.config.negotiation_steps == 0:
            # Step 0 is anchored at the newly accepted macro incumbent.
            self._macro_public_incumbent = dict(anchor_records)
        if self._macro_public_incumbent is None:
            raise RuntimeError("NoPeer public incumbent was not initialized")

        self._active_public_incumbent_hash = _joint_hash(
            self._macro_public_incumbent
        )
        self._active_public_incumbent_signatures = tuple(
            self._macro_public_incumbent[i].candidate_id
            for i in range(self.config.num_agents)
        )
        original_policy = self.policy
        if original_policy is not None:
            self.policy = _PublicIncumbentPolicy(
                original_policy, self._macro_public_incumbent
            )
        try:
            return super()._query_wave(*args, **kwargs)
        finally:
            self.policy = original_policy
            self._active_public_incumbent_hash = None
            self._active_public_incumbent_signatures = None


def run_public_incumbent_protocol(
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
    return PublicIncumbentProtocol(
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
    "PublicIncumbentProtocol",
    "run_public_incumbent_protocol",
]
