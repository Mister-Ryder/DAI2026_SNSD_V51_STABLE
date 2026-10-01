"""Learned, leakage-safe TriGraph base_protocol runner (v3).

This is the first runner that purchases candidates according to the learned
TriGraph policy.  The strict v2 accounting, compliance, uniqueness, Max-Sum,
verification, and monotone-acceptance invariants remain inherited unchanged.
Oracle execution is retained as an explicitly separate scaffold mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from snsd_core.candidates import BoundarySignature, CandidateGenerator
from snsd_core.solver import ConditionedLocalSolver

from . import base_protocol as _legacy_experiment
from .base_protocol import METHODS, ExperimentConfig, ExperimentResult, _RealizedRecord, _sha256_file
from .fair_state_protocol import FairStateProtocol
from .model import FeatureMode
from .messages import (
    BoundaryMessage,
    BoundaryState,
    MaterializedProposal,
    ProposalEngine,
    apply_message_intervention,
    materialize_intent,
    solver_compliance,
)
from .runtime import RankedIntent, TriGraphPolicy


_FEATURE_MODES = {
    "Full": FeatureMode.FULL,
    "NoPeerComm": FeatureMode.ZERO,
    "SelectionOnly": FeatureMode.ZERO,
    "ShuffledComm": FeatureMode.SHUFFLED,
    "NoGlobalContext": FeatureMode.NO_GLOBAL,
    "ConflictMessageOnly": FeatureMode.CONFLICT_ONLY,
    "BenefitMessageOnly": FeatureMode.BENEFIT_ONLY,
}


@dataclass(frozen=True)
class LearnedPolicyConfig(ExperimentConfig):
    checkpoint: Optional[str] = None
    device: str = "cpu"
    allow_untrained_smoke: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        checkpoint = str(self.checkpoint).strip() if self.checkpoint else None
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "device", str(self.device).strip() or "cpu")
        if self.oracle and checkpoint:
            raise ValueError("oracle scaffold and learned checkpoint are separate modes")
        if self.allow_untrained_smoke and not self.smoke:
            raise ValueError("--allow-untrained-smoke is valid only together with --smoke")
        if not self.oracle and checkpoint is None and not (
            self.smoke and self.allow_untrained_smoke
        ):
            raise ValueError(
                "non-oracle learned runs require --checkpoint; random weights are "
                "allowed only with --smoke --allow-untrained-smoke"
            )
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise ValueError("checkpoint does not exist: {}".format(checkpoint))

    @property
    def feature_mode(self) -> FeatureMode:
        return _FEATURE_MODES[self.method]


class LearnedPolicyProtocol(FairStateProtocol):
    """Strict v2 protocol with v2 messages_base and learned pre-query ranking."""

    config: LearnedPolicyConfig

    def __init__(self, config: LearnedPolicyConfig) -> None:
        super().__init__(config)
        self.policy = None  # type: Optional[TriGraphPolicy]
        if not config.oracle:
            self.policy = TriGraphPolicy(
                device=config.device,
                checkpoint_path=config.checkpoint,
                seed=config.seed,
            )

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        checkpoint_sha = (
            _sha256_file(Path(self.config.checkpoint).resolve())
            if self.config.checkpoint
            else None
        )
        if self.policy is None:
            metadata = {
                "model_hash": None,
                "parameter_count": 0,
                "trainable_parameter_count": 0,
                "checkpoint_path": None,
                "checkpoint_loaded": False,
                "device": None,
                "dtype": None,
            }
        else:
            metadata = self.policy.metadata.to_dict()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-learned-run-v3",
                "oracle_mode": bool(self.config.oracle),
                "learned_model_used": self.policy is not None,
                "learned_policy_used": self.policy is not None,
                "policy_backend": (
                    "exact-pressure-oracle-scaffold"
                    if self.config.oracle
                    else "trigraph-policy-v2"
                ),
                "proposal_backend": "proposals_v2_common_random_numbers",
                "proposal_rng_depends_on_message_hash": False,
                "feature_mode": self.config.feature_mode.value,
                "checkpoint_path": (
                    str(Path(self.config.checkpoint).resolve())
                    if self.config.checkpoint
                    else None
                ),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_loaded": bool(metadata["checkpoint_loaded"]),
                "untrained_smoke": bool(
                    self.policy is not None and not metadata["checkpoint_loaded"]
                ),
                "model_hash": metadata["model_hash"],
                "model_parameter_count": metadata["parameter_count"],
                "model_trainable_parameter_count": metadata[
                    "trainable_parameter_count"
                ],
                "model_device": metadata["device"],
                "model_dtype": metadata["dtype"],
                "prequery_selection_uses_worker_labels": False,
                "identifiability_mask_persisted": True,
            }
        )
        return fields

    def run(self) -> ExperimentResult:
        # The base runner constructs engines in run().  Bind that constructor
        # to messages for this isolated call and restore it afterwards.
        original = _legacy_experiment.ProposalEngine
        _legacy_experiment.ProposalEngine = ProposalEngine
        try:
            return super().run()
        finally:
            _legacy_experiment.ProposalEngine = original

    @staticmethod
    def _pressure_with_identifiability(
        boundary: Sequence[int],
        records: Sequence[_RealizedRecord],
        values: Sequence[float],
    ) -> Tuple[Dict[int, float], Dict[int, bool]]:
        pressure = {}
        identifiable = {}
        for node_id in boundary:
            on = [
                float(values[index])
                for index, record in enumerate(records)
                if node_id in record.signature.selected_ids
                and math.isfinite(float(values[index]))
            ]
            off = [
                float(values[index])
                for index, record in enumerate(records)
                if node_id not in record.signature.selected_ids
                and math.isfinite(float(values[index]))
            ]
            known = bool(on and off)
            identifiable[int(node_id)] = known
            pressure[int(node_id)] = max(on) - max(off) if known else 0.0
        return pressure, identifiable

    def _pressure_maps(self, generators, ordered_records, factor):
        maps = {}
        for agent_id, generator in enumerate(generators):
            records = ordered_records[agent_id]
            local_values = [float(item.result.objective) for item in records]
            benefit, benefit_known = self._pressure_with_identifiability(
                generator.boundary, records, local_values
            )
            if factor is None:
                full = dict(benefit)
                full_known = dict(benefit_known)
                conflict = {node_id: 0.0 for node_id in generator.boundary}
                conflict_known = {node_id: False for node_id in generator.boundary}
                opportunity = {node_id: 0.0 for node_id in generator.boundary}
                opportunity_known = {
                    node_id: False for node_id in generator.boundary
                }
            else:
                full, full_known = self._pressure_with_identifiability(
                    generator.boundary,
                    records,
                    factor.forced_best_completions[agent_id],
                )
                neg_cost = [
                    -float(value) if math.isfinite(float(value)) else float("-inf")
                    for value in factor.opportunity_costs[agent_id]
                ]
                conflict, conflict_known = self._pressure_with_identifiability(
                    generator.boundary, records, neg_cost
                )
                opportunity = {}
                opportunity_known = {}
                for node_id in generator.boundary:
                    known = bool(full_known[node_id] and benefit_known[node_id])
                    opportunity_known[node_id] = known
                    opportunity[node_id] = (
                        abs(float(full[node_id] - benefit[node_id])) if known else 0.0
                    )
            maps[agent_id] = {
                "full": full,
                "benefit": benefit,
                "conflict": conflict,
                "opportunity": opportunity,
                "identifiable_full": full_known,
                "identifiable_benefit": benefit_known,
                "identifiable_conflict": conflict_known,
                "identifiable_opportunity": opportunity_known,
            }
        return maps

    def _state(
        self,
        agent_id: int,
        generator: CandidateGenerator,
        incumbent: _RealizedRecord,
        arcs: Sequence[Any],
        graph: Any,
        pressure: Optional[Mapping[str, Mapping[int, Any]]] = None,
    ) -> BoundaryState:
        raw = {node_id: float(arcs[node_id].weight) for node_id in generator.boundary}
        local_proxy = {
            node_id: float(arcs[node_id].weight)
            / float(int(graph.indptr[node_id + 1] - graph.indptr[node_id]) + 1)
            for node_id in generator.boundary
        }
        cut_conflict = {node_id: 0.0 for node_id in generator.boundary}
        for raw_u, raw_v in self.partitions.cut_edges:
            u, v = int(raw_u), int(raw_v)
            if u in cut_conflict:
                cut_conflict[u] += 1.0
            if v in cut_conflict:
                cut_conflict[v] += 1.0
        return BoundaryState.from_maps(
            agent_id=agent_id,
            boundary_order=generator.boundary,
            incumbent_selected=incumbent.signature.selected_ids,
            raw_benefit=raw,
            local_marginal=local_proxy,
            conflict_score=cut_conflict,
            prior_opportunity_cost={},
            uncertainty={},
            prior_pressure={},
        )

    def _message(
        self,
        generator: CandidateGenerator,
        pressure: Mapping[str, Mapping[int, Any]],
    ) -> BoundaryMessage:
        mode = self.behavior.peer_pressure
        if mode == "none":
            selected_pressure = {node_id: 0.0 for node_id in generator.boundary}
            opportunity = {node_id: 0.0 for node_id in generator.boundary}
            known = {node_id: False for node_id in generator.boundary}
        else:
            selected_pressure = pressure[mode]
            opportunity = (
                pressure["opportunity"]
                if mode in ("full", "conflict")
                else {node_id: 0.0 for node_id in generator.boundary}
            )
            selected_known = pressure["identifiable_" + mode]
            opportunity_known = pressure["identifiable_opportunity"]
            known = {
                node_id: bool(
                    selected_known[node_id]
                    and (mode == "benefit" or opportunity_known[node_id])
                )
                for node_id in generator.boundary
            }
        uncertainty = {
            node_id: 0.0 if known[node_id] else 1.0
            for node_id in generator.boundary
        }
        return BoundaryMessage.from_maps(
            boundary_order=generator.boundary,
            opportunity_cost=opportunity,
            pressure=selected_pressure,
            neighbor_recovery=opportunity,
            conflict_score=pressure["conflict"],
            uncertainty=uncertainty,
        )

    def _rank_oracle(
        self,
        state: BoundaryState,
        message: Optional[BoundaryMessage],
        materialized: Sequence[MaterializedProposal],
        anchor: BoundarySignature,
    ) -> List[Tuple[float, str, MaterializedProposal, Optional[RankedIntent]]]:
        message_pressure = message.value_map("pressure") if message is not None else {}
        scoring = {
            node_id: float(state.local_marginal[index])
            - 0.10 * float(state.conflict_score[index])
            + float(message_pressure.get(node_id, 0.0))
            for index, node_id in enumerate(state.boundary_order)
        }
        result = [
            (
                self._intent_score(item.signature, anchor, scoring),
                item.intent.intent_hash,
                item,
                None,
            )
            for item in materialized
        ]
        return sorted(result, key=lambda row: (-row[0], row[1]))

    def _query_wave(
        self,
        wave: str,
        macro_round: int,
        generators: Sequence[CandidateGenerator],
        engines: Sequence[ProposalEngine],
        incumbent_records: Mapping[int, _RealizedRecord],
        solver: ConditionedLocalSolver,
        graph: Any,
        pressure_maps: Optional[Mapping[int, Mapping[str, Mapping[int, Any]]]],
    ) -> Dict[int, List[_RealizedRecord]]:
        states = {}
        messages = {}
        materialized_by_agent = {}
        rejected_by_agent = {}

        for agent_id in range(self.config.num_agents):
            generator = generators[agent_id]
            anchor = incumbent_records[agent_id].signature
            local_pressure = pressure_maps.get(agent_id) if pressure_maps else None
            state = self._state(
                agent_id,
                generator,
                incumbent_records[agent_id],
                self.dataset.arcs,
                graph,
                pressure=local_pressure,
            )
            states[agent_id] = state
            if wave == "A":
                intervention = "none"
                intents = engines[agent_id].wave_a(
                    state, macro_round, count=self.config.proposal_pool
                )
            else:
                if local_pressure is None:
                    raise RuntimeError("wave B requires completed wave-A messages")
                real_message = self._message(generator, local_pressure)
                intervention = self.behavior.wave_b_intervention
                intents = engines[agent_id].wave_b(
                    state,
                    real_message,
                    macro_round,
                    intervention=intervention,
                    count=self.config.proposal_pool,
                )
                messages[agent_id] = apply_message_intervention(
                    real_message,
                    intervention,
                    experiment_seed=self.config.seed,
                    agent_id=agent_id,
                    macro_round=macro_round,
                )

            seen = set(self._purchased_signatures[agent_id])
            accepted = []
            rejected = []
            for intent in intents:
                item = materialize_intent(
                    intent,
                    generator,
                    anchor_signature=anchor,
                    strict_compliance=True,
                    seen_signature_hashes=seen,
                )
                if not item.eligible_for_query:
                    rejected.append(item.to_dict())
                    continue
                seen.add(str(item.signature.signature_hash))
                accepted.append(item)
            materialized_by_agent[agent_id] = accepted
            rejected_by_agent[agent_id] = rejected

        prepared = None
        if self.policy is not None:
            prepared = self.policy.prepare(
                self.dataset,
                graph,
                self.partitions,
                generators,
                boundary_states=states,
                boundary_messages=messages if wave == "B" else None,
                incumbent=incumbent_records,
                mode=self.config.feature_mode,
            )

        queried = {agent_id: [] for agent_id in range(self.config.num_agents)}
        for agent_id in range(self.config.num_agents):
            anchor = incumbent_records[agent_id].signature
            accepted = materialized_by_agent[agent_id]
            by_hash = {item.intent.intent_hash: item for item in accepted}
            if self.policy is None:
                ranked = self._rank_oracle(
                    states[agent_id], messages.get(agent_id), accepted, anchor
                )
            else:
                assert prepared is not None
                policy_ranks = self.policy.rank_intents(
                    prepared,
                    agent_id,
                    [item.intent for item in accepted],
                )
                ranked = [
                    (
                        float(row.selection_logit),
                        row.intent.intent_hash,
                        by_hash[row.intent.intent_hash],
                        row,
                    )
                    for row in policy_ranks
                ]

            selected = list(ranked[: self.config.query_budget])
            selected_hashes = {
                str(row[2].signature.signature_hash) for row in selected
            }
            needed = self.config.query_budget - len(selected)
            fallback_signatures = self._fallback_signatures(
                generators[agent_id],
                anchor,
                macro_round,
                wave,
                set(self._purchased_signatures[agent_id]).union(selected_hashes),
                needed,
            )
            fallback_rows = [
                (0.0, "fallback-{}".format(signature.signature_hash), signature, None)
                for signature in fallback_signatures
            ]
            if len(selected) + len(fallback_rows) != self.config.query_budget:
                raise RuntimeError(
                    "unique query budget unavailable before purchase: agent={} "
                    "macro_round={} wave={} required={} learned={} fallback={}".format(
                        agent_id,
                        macro_round,
                        wave,
                        self.config.query_budget,
                        len(selected),
                        len(fallback_rows),
                    )
                )

            intervention = (
                "none" if wave == "A" else self.behavior.wave_b_intervention
            )
            self._event(
                "proposal_pool",
                macro_round=macro_round,
                wave=wave,
                agent_id=agent_id,
                intervention=intervention,
                feature_mode=self.config.feature_mode.value,
                state_hash=states[agent_id].state_hash,
                prequery_score_source=(
                    "oracle-static-message-proxy"
                    if self.policy is None
                    else "TriGraphPolicy.rank_intents"
                ),
                pool_size=len(accepted),
                rejected=rejected_by_agent[agent_id],
                rankings=[
                    row[3].to_dict()
                    if row[3] is not None
                    else {
                        "intent_hash": row[1],
                        "selection_logit": row[0],
                        "oracle_scaffold": True,
                    }
                    for row in ranked
                ],
                selected_signature_hashes=[
                    row[2].signature.signature_hash for row in selected
                ]
                + [row[2].signature_hash for row in fallback_rows],
                fallback_count=len(fallback_rows),
                identifiable_mask=(
                    {
                        str(node): bool(
                            pressure_maps[agent_id]["identifiable_full"][node]
                        )
                        for node in generators[agent_id].boundary
                    }
                    if pressure_maps is not None
                    else None
                ),
            )

            purchases = [
                (score, key, item.signature, item, policy_row)
                for score, key, item, policy_row in selected
            ] + [
                (score, key, signature, None, None)
                for score, key, signature, _ in fallback_rows
            ]
            for slot, (_, key, signature, item, policy_row) in enumerate(purchases):
                payload = (
                    item.to_dict()
                    if item is not None
                    else {
                        "source": "candidate_generator_fallback",
                        "proposal_key": key,
                        "signature": signature.to_dict(),
                    }
                )
                if policy_row is not None:
                    payload["policy_score"] = policy_row.to_dict()
                record = self._solve_signature(
                    solver,
                    graph,
                    signature,
                    phase="online",
                    macro_round=macro_round,
                    wave=wave,
                    slot=slot,
                    intent_payload=payload,
                )
                if item is not None:
                    compliance = solver_compliance(item, record.result)
                    self._event(
                        "solver_compliance",
                        macro_round=macro_round,
                        wave=wave,
                        agent_id=agent_id,
                        compliance=compliance.to_dict(),
                    )
                    if not compliance.full_compliance:
                        raise RuntimeError("worker violated purchased boundary intent")
                    if str(record.result.signature_hash) != str(signature.signature_hash):
                        raise RuntimeError("worker returned a different signature hash")
                queried[agent_id].append(record)
                self.objective_trace.append(float(self.incumbent_objective))
        return queried


def run_learned_experiment(
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
    return LearnedPolicyProtocol(
        LearnedPolicyConfig(
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
    "LearnedPolicyConfig",
    "LearnedPolicyProtocol",
    "run_learned_experiment",
]
