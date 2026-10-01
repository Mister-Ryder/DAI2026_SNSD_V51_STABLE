"""Strict v2 adapter for the runnable SNSD v1.5 base_protocol.

The original :mod:`base_protocol` module is kept intact as a reproducibility
baseline.  This adapter tightens the experimental protocol in four places:

* non-compliant intent projections are never purchased from the worker;
* every worker purchase has a previously unseen signature for that agent;
* peer information lives only in ``BoundaryMessage`` (never in state);
* proposal ranking uses pre-query graph/message proxies, not worker labels.

The adapter is still a deterministic, no-learning mechanism scaffold.  It
does not claim that the learned TriGraph policy has been integrated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from snsd_core.candidates import BoundarySignature, CandidateGenerator
from snsd_core.solver import ConditionedLocalSolver

from .base_protocol import (
    METHODS,
    ExperimentConfig,
    ExperimentResult,
    BaseNegotiationProtocol,
    _RealizedRecord,
)
from .messages_base import (
    ProposalEngine,
    apply_message_intervention,
    materialize_intent,
    solver_compliance,
)


def projection_is_purchasable(materialized: Any) -> bool:
    """Return whether an intent survived projection without changing any bit."""

    compliance = getattr(materialized, "projection_compliance", None)
    return bool(compliance is not None and compliance.full_compliance)


class FairStateProtocol(BaseNegotiationProtocol):
    """Strict, auditable version of the current-domain two-wave runner."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        # A query is a purchased label.  Re-purchasing it would inflate the
        # registered budget without adding evidence, so warmup and online
        # purchases share one ledger.
        self._purchased_signatures = {
            agent_id: set() for agent_id in range(self.config.num_agents)
        }

    def _event(self, event_type: str, **payload: Any) -> None:
        # Correct the legacy metadata before it is serialized and before the
        # base runner writes summary.json.
        if event_type == "run_start":
            payload["oracle"] = bool(self.config.oracle)
            payload["learned_model_used"] = False
            payload["strict_query_protocol"] = True
        elif event_type == "run_complete":
            summary = payload.get("summary")
            if isinstance(summary, dict):
                summary.update(self._strict_summary_fields())
        super()._event(event_type, **payload)

    def _strict_summary_fields(self) -> Dict[str, Any]:
        return {
            "schema_version": "SNSD-v1.5-strict-run-v2",
            "oracle_mode": bool(self.config.oracle),
            "learned_model_used": False,
            "learned_policy_used": False,
            "policy_backend": (
                "exact-pressure-oracle-no-learning"
                if self.config.oracle
                else "deterministic-state-message-proxy-no-learning"
            ),
            "prequery_selection_uses_worker_labels": False,
            "message_source": "revealed-current-domain-factor-values",
            "proposal_projection_full_compliance_required": True,
            "duplicate_signature_queries_allowed": False,
        }

    def _state(
        self,
        agent_id: int,
        generator: CandidateGenerator,
        incumbent: _RealizedRecord,
        arcs: Sequence[Any],
        graph: Any,
        pressure: Optional[Mapping[str, Mapping[int, float]]] = None,
    ):
        # Intentionally discard pressure here.  This makes the state identical
        # under Real/Zero/Shuffled interventions; only BoundaryMessage changes.
        return super()._state(
            agent_id,
            generator,
            incumbent,
            arcs,
            graph,
            pressure=None,
        )

    def _solve_signature(
        self,
        solver: ConditionedLocalSolver,
        graph: Any,
        signature: BoundarySignature,
        phase: str,
        macro_round: int,
        wave: str,
        slot: int,
        intent_payload: Optional[Mapping[str, Any]] = None,
    ) -> _RealizedRecord:
        ledger = self._purchased_signatures[int(signature.agent_id)]
        signature_hash = str(signature.signature_hash)
        if signature_hash in ledger:
            raise RuntimeError(
                "duplicate worker purchase rejected: agent={} signature={} "
                "phase={} macro_round={} wave={}".format(
                    signature.agent_id,
                    signature_hash,
                    phase,
                    macro_round,
                    wave,
                )
            )
        ledger.add(signature_hash)
        try:
            return super()._solve_signature(
                solver,
                graph,
                signature,
                phase,
                macro_round,
                wave,
                slot,
                intent_payload,
            )
        except Exception:
            # A failed call produced no trustworthy purchased label and may be
            # retried after the underlying failure has been addressed.
            ledger.discard(signature_hash)
            raise

    def _fallback_signatures(
        self,
        generator: CandidateGenerator,
        anchor: BoundarySignature,
        macro_round: int,
        wave: str,
        excluded: set,
        needed: int,
    ) -> List[BoundarySignature]:
        """Search deterministic v1 perturbations until ``needed`` are unique."""

        found = []  # type: List[BoundarySignature]
        wave_offset = 0 if wave == "A" else 1
        for attempt in range(8):
            if len(found) >= needed:
                break
            round_id = 2 * int(macro_round) + wave_offset + 1009 * attempt
            target = max(
                needed - len(found),
                self.config.proposal_pool * (4 + 2 * attempt),
            )
            messages_base = generator.messages_base(
                anchor,
                count=target,
                round_id=round_id,
                exclude_hashes=excluded,
            )
            for signature in messages_base:
                signature_hash = str(signature.signature_hash)
                if signature_hash in excluded:
                    continue
                excluded.add(signature_hash)
                found.append(signature)
                if len(found) >= needed:
                    break
        return found

    def _query_wave(
        self,
        wave: str,
        macro_round: int,
        generators: Sequence[CandidateGenerator],
        engines: Sequence[ProposalEngine],
        incumbent_records: Mapping[int, _RealizedRecord],
        solver: ConditionedLocalSolver,
        graph: Any,
        pressure_maps: Optional[Mapping[int, Mapping[str, Mapping[int, float]]]],
    ) -> Dict[int, List[_RealizedRecord]]:
        queried = {agent_id: [] for agent_id in range(self.config.num_agents)}
        for agent_id in range(self.config.num_agents):
            generator = generators[agent_id]
            anchor_record = incumbent_records[agent_id]
            local_pressure = pressure_maps.get(agent_id) if pressure_maps else None
            state = self._state(
                agent_id,
                generator,
                anchor_record,
                self.dataset.arcs,
                graph,
                pressure=local_pressure,
            )

            if wave == "A":
                intents = engines[agent_id].wave_a(
                    state, macro_round, count=self.config.proposal_pool
                )
                intervention = "none"
                message_pressure = {}
            else:
                if local_pressure is None:
                    raise RuntimeError("wave B requires a completed wave-A message")
                real_message = self._message(generator, local_pressure)
                intervention = self.behavior.wave_b_intervention
                intents = engines[agent_id].wave_b(
                    state,
                    real_message,
                    macro_round,
                    intervention=intervention,
                    count=self.config.proposal_pool,
                )
                intervened = apply_message_intervention(
                    real_message,
                    intervention,
                    experiment_seed=self.config.seed,
                    agent_id=agent_id,
                    macro_round=macro_round,
                )
                message_pressure = intervened.value_map("pressure")

            # This proxy is available before purchasing the candidate.  It
            # contains graph/static features and the received message only;
            # no candidate worker objective or exact-local label appears here.
            scoring = {
                node_id: float(state.local_marginal[index])
                - 0.10 * float(state.conflict_score[index])
                + float(message_pressure.get(node_id, 0.0))
                for index, node_id in enumerate(state.boundary_order)
            }

            purchased = self._purchased_signatures[agent_id]
            excluded = set(purchased)
            pool = []
            rejected = []
            for intent in intents:
                materialized = materialize_intent(
                    intent, generator, anchor_signature=anchor_record.signature
                )
                if not projection_is_purchasable(materialized):
                    rejected.append(
                        {
                            "intent_hash": intent.intent_hash,
                            "reason": "projection_noncompliance",
                            "compliance": materialized.projection_compliance.to_dict(),
                        }
                    )
                    continue
                signature_hash = str(materialized.signature.signature_hash)
                if signature_hash in excluded:
                    rejected.append(
                        {
                            "intent_hash": intent.intent_hash,
                            "reason": "duplicate_or_previously_purchased_signature",
                            "signature_hash": signature_hash,
                        }
                    )
                    continue
                excluded.add(signature_hash)
                score = self._intent_score(
                    materialized.signature, anchor_record.signature, scoring
                )
                pool.append(
                    (score, str(intent.intent_hash), materialized.signature, materialized)
                )

            pool.sort(key=lambda item: (-item[0], item[1]))
            selected = list(pool[: self.config.query_budget])
            selected_hashes = {str(item[2].signature_hash) for item in selected}

            # Do not pad with repeats.  Continue through deterministic legal
            # perturbations, or fail clearly before buying a partial budget.
            fallback_excluded = set(purchased).union(selected_hashes)
            needed = self.config.query_budget - len(selected)
            fallbacks = self._fallback_signatures(
                generator,
                anchor_record.signature,
                macro_round,
                wave,
                fallback_excluded,
                needed,
            )
            for fallback_index, signature in enumerate(fallbacks):
                score = self._intent_score(
                    signature, anchor_record.signature, scoring
                )
                key = "fallback-{:02d}-{}".format(
                    fallback_index, signature.signature_hash
                )
                selected.append((score, key, signature, None))

            if len(selected) != self.config.query_budget:
                raise RuntimeError(
                    "unique query budget unavailable: agent={} macro_round={} "
                    "wave={} required={} found={} previously_purchased={}".format(
                        agent_id,
                        macro_round,
                        wave,
                        self.config.query_budget,
                        len(selected),
                        len(purchased),
                    )
                )
            if len({str(item[2].signature_hash) for item in selected}) != len(selected):
                raise RuntimeError("internal error: selected worker queries are not unique")

            self._event(
                "proposal_pool",
                macro_round=macro_round,
                wave=wave,
                agent_id=agent_id,
                intervention=intervention,
                state_hash=state.state_hash,
                prequery_score_source="static-graph-plus-boundary-message",
                pool_size=len(pool),
                rejected=rejected,
                predictions=[
                    {
                        "proposal_key": key,
                        "score": score,
                        "signature_hash": signature.signature_hash,
                        "source": "intent" if materialized is not None else "fallback",
                        "intent": (
                            materialized.intent.to_dict()
                            if materialized is not None
                            else None
                        ),
                    }
                    for score, key, signature, materialized in selected
                ],
            )

            for slot, (_, key, signature, materialized) in enumerate(selected):
                if materialized is None:
                    intent_payload = {
                        "source": "candidate_generator_fallback",
                        "proposal_key": key,
                        "signature": signature.to_dict(),
                    }
                else:
                    intent_payload = materialized.to_dict()
                record = self._solve_signature(
                    solver,
                    graph,
                    signature,
                    phase="online",
                    macro_round=macro_round,
                    wave=wave,
                    slot=slot,
                    intent_payload=intent_payload,
                )
                if materialized is not None:
                    compliance = solver_compliance(materialized, record.result)
                    self._event(
                        "solver_compliance",
                        macro_round=macro_round,
                        wave=wave,
                        agent_id=agent_id,
                        compliance=compliance.to_dict(),
                    )
                    if not compliance.full_compliance:
                        raise RuntimeError(
                            "worker violated purchased boundary intent: agent={} "
                            "signature={}".format(agent_id, signature.signature_hash)
                        )
                queried[agent_id].append(record)
                self.objective_trace.append(float(self.incumbent_objective))
        return queried


def run_fair_state_protocol(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
) -> ExperimentResult:
    return FairStateProtocol(
        ExperimentConfig(
            data_path=data_path,
            output_dir=output_dir,
            method=method,
            seed=seed,
            smoke=smoke,
            oracle=oracle,
        )
    ).run()


__all__ = [
    "METHODS",
    "ExperimentConfig",
    "ExperimentResult",
    "FairStateProtocol",
    "projection_is_purchasable",
    "run_fair_state_protocol",
]
