"""Channel-corrected, consumed-message-accounted recurrent runner v4.2."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .base_protocol import ExperimentResult
from .validated_policy_protocol import ValidatedPolicyProtocol
from .recurrent_protocol import RECURRENT_METHODS, RecurrentNegotiationConfig
from .pressure_relay_protocol import PressureRelayProtocol
from .messages import BoundaryMessage


class FairRelayProtocol(PressureRelayProtocol):
    """v4.1 audit contract plus v3.2 disjoint communication channels."""

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-recurrent-learned-run-v4.2",
                "benefit_message_channel": "identifiable_local_benefit_pressure",
                "conflict_message_channel": "identifiable_opportunity_cost_pressure",
                "message_channels_disjoint_in_single_channel_ablations": True,
                "consumed_message_cycles": self.message_passes,
                "terminal_factor_counted_as_communication": False,
            }
        )
        return fields

    def _message(
        self,
        generator: Any,
        pressure: Mapping[str, Mapping[int, Any]],
    ) -> BoundaryMessage:
        # Reuse the audited v3.2 semantics verbatim.  Calling it explicitly
        # avoids a multiple-inheritance MRO that could silently fall back to
        # the older v3.1 channel mapping.
        return ValidatedPolicyProtocol._message(self, generator, pressure)

    def _count_factor_communication(self, step: int) -> bool:
        step = int(step)
        if not self.behavior.wave_a_message_pass:
            return False
        if self.config.one_shot_comm:
            return step == 0
        # The post-step-5 factor is the terminal selector.  No later proposal
        # consumes its outgoing messages, so it is computation, not a
        # communication cycle.
        return 0 <= step < self.config.negotiation_steps - 1


def run_fair_relay_protocol(
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
    one_shot = method == "OneShotComm"
    base_method = "Full" if one_shot else method
    if method not in RECURRENT_METHODS:
        raise ValueError("unknown recurrent method: {}".format(method))
    return FairRelayProtocol(
        RecurrentNegotiationConfig(
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
        )
    ).run()


__all__ = [
    "RECURRENT_METHODS",
    "RecurrentNegotiationConfig",
    "FairRelayProtocol",
    "run_fair_relay_protocol",
]
