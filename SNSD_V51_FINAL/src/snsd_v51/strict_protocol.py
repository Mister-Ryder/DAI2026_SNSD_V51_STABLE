"""Final strict v4.5 runner with causal feedback timing and NoPeer isolation.

This module is the canonical v4.5 entrypoint.  It keeps the exact current-
domain chain Max-Sum selector in every cell of the registered 2x2 factorial:

* ``Full``: pressure messages and rolling joint-anchor feedback (A1B1),
* ``FrozenAnchorComm``: pressure messages only (A1B0),
* ``SelectionOnly``: rolling joint-anchor feedback only (A0B1), and
* ``FrozenAnchorSelectionOnly``: neither feedback channel (A0B0).

``NoPeerComm`` is deliberately retained only as an additional system
baseline.  Its independent local selector is not a factorial cell.

The strict runner also closes the learned-policy API barrier for NoPeer.  A
policy preparation made for agent ``i`` receives its own rolling local anchor
``W_i`` but receives the macro-start public accepted incumbent ``I_j`` in
every peer ``j`` BoundaryState.  The global incumbent mask and the factor
feasibility anchor likewise use ``I``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .base_protocol import ExperimentResult
from .rolling_protocol import RollingNegotiationProtocol, _joint_hash
from .factorial_protocol import (
    FACTORIAL_METHODS,
    STRICT_FACTORIAL_CELLS,
    FactorialExperimentConfig,
    FactorialNegotiationProtocol,
    _factorial_config,
)
from .finalized_rolling_protocol import FinalizedRollingProtocol
from .messages import BoundaryState


STRICT_METHODS = FACTORIAL_METHODS


def _state_with_public_incumbent(
    state: BoundaryState, selected: Any
) -> BoundaryState:
    """Rebuild ``state`` with an audited incumbent and a matching state hash."""

    return type(state).from_maps(
        agent_id=state.agent_id,
        boundary_order=state.boundary_order,
        incumbent_selected=selected,
        raw_benefit=state.raw_benefit,
        local_marginal=state.local_marginal,
        conflict_score=state.conflict_score,
        prior_opportunity_cost=state.prior_opportunity_cost,
        uncertainty=state.uncertainty,
        prior_pressure=state.prior_pressure,
    )


@dataclass(frozen=True)
class _PerAgentPrepared:
    """Prepared policy caches whose visibility contract differs by agent."""

    by_agent: Mapping[int, Any]


class _NoPeerPerAgentPolicy:
    """Enforce a separate, peer-sanitised ``prepare`` call for every agent."""

    def __init__(self, policy: Any, public_incumbent: Mapping[int, Any]) -> None:
        self._policy = policy
        self._public_incumbent = dict(public_incumbent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def prepare(self, *args: Any, **kwargs: Any) -> _PerAgentPrepared:
        states = dict(kwargs["boundary_states"])
        prepared = {}
        for target_agent in sorted(states):
            visible_states = {}
            for source_agent, state in states.items():
                if int(source_agent) == int(target_agent):
                    # The only dynamic local working anchor exposed to i is W_i.
                    visible_states[int(source_agent)] = state
                else:
                    public_selected = self._public_incumbent[
                        int(source_agent)
                    ].signature.selected_ids
                    visible_states[int(source_agent)] = _state_with_public_incumbent(
                        state, public_selected
                    )
            visible_kwargs = dict(kwargs)
            visible_kwargs["boundary_states"] = visible_states
            # TriGraph's full/local selected mask must be the public feasible I.
            visible_kwargs["incumbent"] = self._public_incumbent
            prepared[int(target_agent)] = self._policy.prepare(
                *args, **visible_kwargs
            )
        return _PerAgentPrepared(by_agent=prepared)

    def rank_intents(
        self,
        prepared: _PerAgentPrepared,
        agent_id: int,
        intents: Any,
    ) -> Any:
        target = int(agent_id)
        return self._policy.rank_intents(
            prepared.by_agent[target], target, intents
        )


class StrictNegotiationProtocol(FactorialNegotiationProtocol):
    """Canonical strict runner for v4.5 mechanism and ablation experiments."""

    config: FactorialExperimentConfig

    def __init__(self, config: FactorialExperimentConfig) -> None:
        super().__init__(config)
        self._strict_channel_uses = {
            "pressure_message": {
                "incoming_consumed_steps": 0,
                "outgoing_emitted_steps": 0,
                "incoming_payload_scalars": 0,
                "outgoing_payload_scalars": 0,
            },
            "joint_anchor_feedback": {
                "incoming_consumed_steps": 0,
                "outgoing_emitted_steps": 0,
                "incoming_payload_scalars": 0,
                "outgoing_payload_scalars": 0,
            },
        }
        self._strict_steps_with_any_feedback = 0
        self._strict_pending_pressure_payload = {}  # type: Dict[Any, int]
        self._strict_pending_anchor_payload = {}  # type: Dict[Any, int]

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        # These inherited names conflate a channel use with wall-clock time or
        # sum two concurrent channels.  Keep the underlying legacy
        # ``message_passes`` counter for artifact compatibility, but make
        # ``channel_uses`` the sole strict causal accounting interface.
        for key in (
            "consumed_message_cycles",
            "pressure_message_cycles",
            "pressure_message_payload_scalars",
            "pressure_message_payload_bytes_float64",
            "joint_anchor_feedback_cycles",
            "joint_anchor_feedback_payload_scalars",
            "joint_anchor_feedback_payload_bytes_int64",
            "total_dynamic_coordination_cycles",
            "message_passes_semantics",
        ):
            fields.pop(key, None)
        fields.update(
            {
                "schema_version": "SNSD-v1.5-rolling-recurrent-run-v4.5-strict",
                "canonical_v45_entrypoint": True,
                "no_peer_per_agent_prepare_barrier": True,
                "no_peer_policy_view": (
                    "own-local-W_i-plus-peer-public-I_j"
                ),
                "channel_uses": {
                    channel: dict(values)
                    for channel, values in self._strict_channel_uses.items()
                },
                "steps_with_any_feedback": int(
                    self._strict_steps_with_any_feedback
                ),
                "negotiation_steps_total": int(self.config.macro_rounds)
                * int(self.config.negotiation_steps),
                "feedback_event_contract": {
                    "incoming": "consumed by this proposal step",
                    "outgoing": "emitted after this step for the next proposal step",
                    "terminal_outgoing": False,
                    "channels_are_not_summed_as_time_cycles": True,
                },
            }
        )
        return fields

    def _query_wave(self, *args: Any, **kwargs: Any):
        if self.behavior.final_selection != "independent":
            return super()._query_wave(*args, **kwargs)

        # Positional contract inherited from learned_policy_protocol._query_wave.
        if len(args) >= 5:
            wave = args[0]
            context = int(args[1])
            proposal_anchor_records = args[4]
        else:
            wave = kwargs["wave"]
            context = int(kwargs["macro_round"])
            proposal_anchor_records = kwargs["incumbent_records"]

        if wave == "A" or context % self.config.negotiation_steps == 0:
            self._macro_public_incumbent = dict(proposal_anchor_records)
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
            self.policy = _NoPeerPerAgentPolicy(
                original_policy, self._macro_public_incumbent
            )
        try:
            # Deliberately bypass public_incumbent_protocol._PublicIncumbentPolicy: that
            # wrapper fixed only the mask, whereas this call also sanitises all
            # peer BoundaryState objects separately for every target agent.
            return RollingNegotiationProtocol._query_wave(self, *args, **kwargs)
        finally:
            self.policy = original_policy
            self._active_public_incumbent_hash = None
            self._active_public_incumbent_signatures = None

    def _pressure_outgoing(self, step: int) -> bool:
        return bool(self._count_factor_communication(int(step)))

    def _pressure_incoming(self, step: int) -> bool:
        return bool(
            int(step) > 0
            and self._count_factor_communication(int(step) - 1)
        )

    def _anchor_outgoing(self, step: int) -> bool:
        return bool(self._joint_feedback_consumed(int(step)))

    def _anchor_incoming(self, step: int) -> bool:
        return bool(
            int(step) > 0
            and self._joint_feedback_consumed(int(step) - 1)
        )

    @staticmethod
    def _timing_fields(
        prefix: str,
        active: bool,
        source_step: Optional[int],
        target_step: Optional[int],
        payload_scalars: int,
    ) -> Dict[str, Any]:
        return {
            prefix: bool(active),
            prefix + "_source_step": source_step if active else None,
            prefix + "_target_step": target_step if active else None,
            prefix + "_payload_scalars": (
                int(payload_scalars) if active else 0
            ),
        }

    def _event(self, event_type: str, **payload: Any) -> None:
        if event_type == "negotiation_step":
            macro = int(payload["macro_round"])
            step = int(payload["negotiation_step"])
            last = int(self.config.negotiation_steps) - 1

            pressure_in = self._pressure_incoming(step)
            pressure_out = self._pressure_outgoing(step)
            anchor_in = self._anchor_incoming(step)
            anchor_out = self._anchor_outgoing(step)

            pressure_in_payload = self._strict_pending_pressure_payload.pop(
                (macro, step), 0
            )
            pressure_out_payload = (
                int(self._last_pressure_payload_scalars)
                if pressure_out
                else 0
            )
            anchor_in_payload = self._strict_pending_anchor_payload.pop(
                (macro, step), 0
            )
            anchor_out_payload = self.config.num_agents if anchor_out else 0

            if pressure_out:
                if step >= last:
                    raise RuntimeError("terminal pressure cannot be emitted")
                self._strict_pending_pressure_payload[(macro, step + 1)] = int(
                    pressure_out_payload
                )
            if anchor_out:
                if step >= last:
                    raise RuntimeError("terminal anchor feedback cannot be emitted")
                self._strict_pending_anchor_payload[(macro, step + 1)] = int(
                    anchor_out_payload
                )

            if pressure_in:
                self._strict_channel_uses["pressure_message"][
                    "incoming_consumed_steps"
                ] += 1
                self._strict_channel_uses["pressure_message"][
                    "incoming_payload_scalars"
                ] += int(pressure_in_payload)
            if pressure_out:
                self._strict_channel_uses["pressure_message"][
                    "outgoing_emitted_steps"
                ] += 1
                self._strict_channel_uses["pressure_message"][
                    "outgoing_payload_scalars"
                ] += int(pressure_out_payload)
            if anchor_in:
                self._strict_channel_uses["joint_anchor_feedback"][
                    "incoming_consumed_steps"
                ] += 1
                self._strict_channel_uses["joint_anchor_feedback"][
                    "incoming_payload_scalars"
                ] += int(anchor_in_payload)
            if anchor_out:
                self._strict_channel_uses["joint_anchor_feedback"][
                    "outgoing_emitted_steps"
                ] += 1
                self._strict_channel_uses["joint_anchor_feedback"][
                    "outgoing_payload_scalars"
                ] += int(anchor_out_payload)
            if pressure_in or anchor_in:
                self._strict_steps_with_any_feedback += 1

            payload.update(
                self._timing_fields(
                    "incoming_pressure_consumed",
                    pressure_in,
                    step - 1 if pressure_in else None,
                    step if pressure_in else None,
                    pressure_in_payload,
                )
            )
            payload.update(
                self._timing_fields(
                    "outgoing_pressure_emitted",
                    pressure_out,
                    step if pressure_out else None,
                    step + 1 if pressure_out else None,
                    pressure_out_payload,
                )
            )
            payload.update(
                self._timing_fields(
                    "incoming_joint_anchor_consumed",
                    anchor_in,
                    step - 1 if anchor_in else None,
                    step if anchor_in else None,
                    anchor_in_payload,
                )
            )
            payload.update(
                self._timing_fields(
                    "outgoing_joint_anchor_emitted",
                    anchor_out,
                    step if anchor_out else None,
                    step + 1 if anchor_out else None,
                    anchor_out_payload,
                )
            )

        # Skip the old Complete._event accounting, whose ``consumed`` label
        # was attached to the outgoing factor pass.  Final._event preserves
        # the audited proposal/factor/public anchor provenance for NoPeer.
        FinalizedRollingProtocol._event(self, event_type, **payload)


def _strict_config(
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
    return _factorial_config(
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


def run_strict_protocol(
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
    config = _strict_config(
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
    return StrictNegotiationProtocol(config).run()


__all__ = [
    "STRICT_METHODS",
    "STRICT_FACTORIAL_CELLS",
    "FactorialExperimentConfig",
    "StrictNegotiationProtocol",
    "run_strict_protocol",
]
