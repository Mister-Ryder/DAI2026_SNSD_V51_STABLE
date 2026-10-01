"""Audit-enriched recurrent v4 runner.

Adds explicit protocol macro/step coordinates and source message hashes to
candidate-purchase events.  Optimization behavior and budgets are identical
to :mod:`recurrent_protocol`.
"""

from __future__ import annotations

from typing import Any, Dict

from .recurrent_protocol import (
    RECURRENT_METHODS,
    RecurrentNegotiationConfig,
    RecurrentNegotiationProtocol,
    run_recurrent_experiment,
)


class PressureRelayProtocol(RecurrentNegotiationProtocol):
    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-recurrent-learned-run-v4.1",
                "step_audit_fields": [
                    "protocol_macro_round",
                    "negotiation_step",
                    "agent_id",
                    "message_hash",
                    "signature_hash",
                    "step_objective",
                ],
            }
        )
        return fields

    def _event(self, event_type: str, **payload: Any) -> None:
        context = payload.get("macro_round")
        wave = payload.get("wave")
        if (
            event_type in ("proposal_pool", "worker_query", "solver_compliance")
            and isinstance(context, int)
            and context >= 0
            and wave in ("A", "B")
        ):
            payload["engine_round_context"] = context
            payload["protocol_macro_round"] = context // 6
            payload["negotiation_step"] = context % 6
        if event_type == "worker_query":
            intent = payload.get("intent")
            message_hash = None
            if isinstance(intent, dict):
                intent_record = intent.get("intent")
                if isinstance(intent_record, dict):
                    message_hash = intent_record.get("source_message_hash")
            payload["message_hash"] = message_hash
            payload["selected_signature_hash"] = payload.get("signature_hash")
        elif event_type == "negotiation_step":
            payload["step_objective"] = payload.get("factor_value")
        super()._event(event_type, **payload)


def run_pressure_relay_protocol(**kwargs: Any):
    method = kwargs.pop("method", "Full")
    one_shot = method == "OneShotComm"
    base_method = "Full" if one_shot else method
    if method not in RECURRENT_METHODS:
        raise ValueError("unknown recurrent method: {}".format(method))
    return PressureRelayProtocol(
        RecurrentNegotiationConfig(
            method=base_method,
            one_shot_comm=one_shot,
            **kwargs
        )
    ).run()


__all__ = [
    "RECURRENT_METHODS",
    "RecurrentNegotiationConfig",
    "PressureRelayProtocol",
    "run_pressure_relay_protocol",
]
