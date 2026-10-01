"""Communication-conditioned feasible repair closure for SNSD v4.9.

This fallback keeps the fixed v4.6 worker/query schedule and persistent
private elites from v4.7.  Its factorial factor B is changed completely:

* every method repairs the union of the six current private elites;
* B=0 uses the same deterministic raw-value repair in both A cells;
* B=1 ranks conflicting boundary nodes with the method's intervened message
  (real, zero, or shuffled) plus full- and boundary-graph context; and
* the repaired tuple R and current exact chain Max-Sum tuple J are explicitly
  verified and evaluated, after which the higher exact objective is emitted.

Repair records enter a later factor domain only after their evaluation event
has been written and they have been consumed as a proposal anchor.  They can
therefore cross a macro boundary only through the ordinary exact factor and
macro-commit path.  No extra worker call is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from snsd_core.solver import LocalSolveResult
from snsd_core.verifier import verify_selected_ids

from .base_protocol import (
    ExperimentResult,
    _RealizedRecord,
    _deduplicate_records,
)
from .long_horizon_protocol import (
    LONG_METHODS,
    LONG_NEGOTIATION_DEPTHS,
    _long_config,
)
from .private_elite_protocol import (
    PrivateEliteConfig,
    PrivateEliteProtocol,
)
from .messages_base import apply_message_intervention


COMM_REPAIR_METHODS = tuple(LONG_METHODS) + ("NoPeerFinalShield",)
_B0_METHODS = (
    "FrozenAnchorComm",
    "FrozenAnchorSelectionOnly",
    "NoPeerFinalShield",
)


@dataclass(frozen=True)
class CommunicationRepairConfig(PrivateEliteConfig):
    """v4.9 method label and registered repair-factor treatment."""

    comm_repair_method_label: str = "Full"
    communication_conditioned_repair: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.comm_repair_method_label not in COMM_REPAIR_METHODS:
            raise ValueError(
                "unknown v4.9 method: {}".format(
                    self.comm_repair_method_label
                )
            )

    @property
    def reported_method(self) -> str:
        return str(self.comm_repair_method_label)


class CommunicationRepairProtocol(PrivateEliteProtocol):
    """Persistent-private-elite negotiation with feasible repair feedback."""

    config: CommunicationRepairConfig

    def __init__(self, config: CommunicationRepairConfig) -> None:
        super().__init__(config)
        # NoPeer keeps its zero-message worker treatment, but its final choice
        # is placed on the same exact J-versus-R path as every other method.
        if self.behavior.final_selection == "independent":
            self.behavior = replace(self.behavior, final_selection="max_sum")
        self._repair_generators = None  # type: Optional[Sequence[Any]]
        self._latest_repair_pressure = None  # type: Optional[Mapping[int, Any]]
        self._repair_trace = []  # type: List[Dict[str, Any]]
        self._repair_evaluations = 0
        self._repair_selected = 0
        self._joint_selected = 0
        self._value_repair_selected = 0
        self._message_repair_selected = 0
        self._extra_coordination_evaluations = 0
        self._repair_feedback_changes = 0
        self._repair_domain_injections = 0
        self._pending_repair_records = None  # type: Optional[Dict[int, List[_RealizedRecord]]]

    def _pressure_maps(
        self,
        generators: Sequence[Any],
        ordered_records: Sequence[Sequence[Any]],
        factor: Any,
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        maps = super()._pressure_maps(generators, ordered_records, factor)
        self._repair_generators = tuple(generators)
        self._latest_repair_pressure = maps
        return maps

    @staticmethod
    def _normalised_map(
        values: Mapping[int, float], nodes: Sequence[int]
    ) -> Dict[int, float]:
        scale = max(
            [abs(float(values.get(int(node), 0.0))) for node in nodes]
            + [1.0e-12]
        )
        return {
            int(node): float(values.get(int(node), 0.0)) / float(scale)
            for node in nodes
        }

    def _repair_messages(
        self,
        macro_round: int,
        step: int,
    ) -> Tuple[Dict[int, Dict[str, Dict[int, float]]], List[str]]:
        if self._repair_generators is None or self._latest_repair_pressure is None:
            raise RuntimeError("repair messages requested before factor pressure")
        channels = {}
        hashes = []
        context_round = (
            int(macro_round) * int(self.config.negotiation_steps) + int(step)
        )
        for agent_id, generator in enumerate(self._repair_generators):
            raw = self._message(
                generator, self._latest_repair_pressure[agent_id]
            )
            message = apply_message_intervention(
                raw,
                self.behavior.wave_b_intervention,
                experiment_seed=self.config.seed,
                agent_id=agent_id,
                macro_round=context_round,
            )
            channels[agent_id] = {
                "pressure": message.value_map("pressure"),
                "conflict": message.value_map("conflict_score"),
                "opportunity": message.value_map("opportunity_cost"),
            }
            hashes.append(str(message.message_hash))
        return channels, hashes

    def _repair_private_elite_union(
        self,
        private_elites: Mapping[int, Any],
        macro_round: int,
        step: int,
        communication_conditioned: bool,
    ) -> Tuple[Tuple[int, ...], Dict[int, _RealizedRecord], Dict[str, Any]]:
        if self._repair_generators is None:
            raise RuntimeError("repair generators are unavailable")

        selected = set()
        for record in private_elites.values():
            selected.update(int(node) for node in record.result.selected_ids)
        pre_ids = tuple(sorted(selected))
        pre_objective = self._objective(pre_ids)

        conflict_edges = [
            (int(raw_u), int(raw_v))
            for raw_u, raw_v in self.partitions.cut_edges
            if int(raw_u) in selected and int(raw_v) in selected
        ]
        conflict_nodes = sorted(
            {node for edge in conflict_edges for node in edge}
        )
        cut_degree = {node: 0.0 for node in conflict_nodes}
        for raw_u, raw_v in self.partitions.cut_edges:
            u, v = int(raw_u), int(raw_v)
            if u in cut_degree:
                cut_degree[u] += 1.0
            if v in cut_degree:
                cut_degree[v] += 1.0

        message_hashes = []  # type: List[str]
        score = {}  # type: Dict[int, float]
        score_parts = {}  # type: Dict[int, Dict[str, float]]
        mode = (
            "communication_conditioned"
            if communication_conditioned
            else "common_value_only"
        )
        if communication_conditioned and conflict_nodes:
            channels, message_hashes = self._repair_messages(
                macro_round, step
            )
            pressure = {}
            conflict = {}
            opportunity = {}
            graph_value = {}
            for node in conflict_nodes:
                agent_id = int(self.partitions.partition_id[node])
                pressure[node] = float(
                    channels[agent_id]["pressure"].get(node, 0.0)
                )
                conflict[node] = float(
                    channels[agent_id]["conflict"].get(node, 0.0)
                )
                opportunity[node] = float(
                    channels[agent_id]["opportunity"].get(node, 0.0)
                )
                full_degree = int(
                    self.graph.indptr[node + 1] - self.graph.indptr[node]
                )
                graph_value[node] = float(self.dataset.arcs[node].weight) / (
                    1.0 + math.log1p(float(full_degree))
                )
            n_value = self._normalised_map(graph_value, conflict_nodes)
            n_pressure = self._normalised_map(pressure, conflict_nodes)
            n_conflict = self._normalised_map(conflict, conflict_nodes)
            n_opportunity = self._normalised_map(
                opportunity, conflict_nodes
            )
            n_cut = self._normalised_map(cut_degree, conflict_nodes)
            for node in conflict_nodes:
                # Positive conflict pressure means that selecting this node
                # relieves factor opportunity cost; opportunity magnitude is
                # separately treated as sensitivity/risk.
                parts = {
                    "full_graph_value": n_value[node],
                    "message_pressure": 0.50 * n_pressure[node],
                    "message_conflict_relief": 0.35 * n_conflict[node],
                    "message_opportunity_risk": -0.25
                    * abs(n_opportunity[node]),
                    "boundary_degree_risk": -0.20 * n_cut[node],
                }
                score[node] = float(sum(parts.values()))
                score_parts[node] = parts
        else:
            for node in conflict_nodes:
                score[node] = float(self.dataset.arcs[node].weight)
                score_parts[node] = {
                    "raw_value": float(self.dataset.arcs[node].weight)
                }

        removed = []
        decisions = []
        for u, v in sorted(conflict_edges):
            if u not in selected or v not in selected:
                continue
            pu = int(self.partitions.partition_id[u])
            pv = int(self.partitions.partition_id[v])
            priority_u = (
                float(score[u]),
                float(self.dataset.arcs[u].weight),
                -pu,
                -u,
            )
            priority_v = (
                float(score[v]),
                float(self.dataset.arcs[v].weight),
                -pv,
                -v,
            )
            loser = v if priority_u >= priority_v else u
            winner = u if loser == v else v
            selected.remove(loser)
            removed.append(loser)
            decisions.append(
                {
                    "edge": [u, v],
                    "winner": int(winner),
                    "removed": int(loser),
                    "winner_score": float(score[winner]),
                    "removed_score": float(score[loser]),
                }
            )

        final_ids = tuple(sorted(selected))
        check = verify_selected_ids(self.graph, final_ids)
        if not check.feasible:
            raise RuntimeError(
                "communication repair failed: {}".format(check.message)
            )
        post_objective = self._objective(final_ids)
        records = {}
        for agent_id, generator in enumerate(self._repair_generators):
            local_nodes = tuple(
                node
                for node in final_ids
                if int(self.partitions.partition_id[node]) == agent_id
            )
            boundary = set(generator.boundary)
            signature = generator.make_signature(
                [node for node in local_nodes if node in boundary]
            )
            result = LocalSolveResult(
                agent_id=agent_id,
                signature_hash=signature.signature_hash,
                status="feasible",
                selected_ids=local_nodes,
                objective=self._objective(local_nodes),
                runtime_s=0.0,
                attempted_moves=0,
                seed=self.config.seed,
                message=(
                    "explicit non-optimizing communication repair evaluation"
                ),
            )
            records[agent_id] = _RealizedRecord(
                signature,
                result,
                "comm_repair_evaluated",
                int(macro_round),
                "coordination_step_{}".format(step),
            )
        metrics = {
            "repair_mode": mode,
            "pre_objective": float(pre_objective),
            "post_objective": float(post_objective),
            "repair_loss": float(pre_objective - post_objective),
            "pre_selected_count": len(pre_ids),
            "post_selected_count": len(final_ids),
            "conflicting_cut_edges": len(conflict_edges),
            "removed_ids": sorted(removed),
            "removed_weight": self._objective(removed),
            "decisions": decisions,
            "message_hashes": message_hashes,
            "score_parts": {
                str(node): score_parts[node] for node in conflict_nodes
            },
        }
        return final_ids, records, metrics

    def _soft_anchor_decision(
        self,
        factor: Any,
        ordered_records: Sequence[Sequence[Any]],
        private_elites: Mapping[int, Any],
        macro_round: int,
        step: int,
    ) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
        joint_records = self._chosen_records(factor, ordered_records)
        joint_ids = self._selected_union(joint_records)
        joint_check = verify_selected_ids(self.graph, joint_ids)
        if not joint_check.feasible:
            raise RuntimeError(
                "exact factor tuple infeasible: {}".format(
                    joint_check.message
                )
            )
        joint_objective = self._objective(joint_ids)
        value_ids, value_records, value_repair = (
            self._repair_private_elite_union(
                private_elites,
                macro_round,
                step,
                communication_conditioned=False,
            )
        )
        value_objective = self._objective(value_ids)
        message_ids = None
        message_records = None
        message_repair = None
        message_objective = None
        if self.config.communication_conditioned_repair:
            message_ids, message_records, message_repair = (
                self._repair_private_elite_union(
                    private_elites,
                    macro_round,
                    step,
                    communication_conditioned=True,
                )
            )
            message_objective = self._objective(message_ids)
            self._extra_coordination_evaluations += 1

        # Safeguarded candidate union.  Ties retain the earlier common
        # candidate, so the message branch wins only by strict exact gain.
        selected_kind = "exact_joint"
        selected_records = joint_records
        selected_objective = float(joint_objective)
        if value_objective > selected_objective + 1.0e-9:
            selected_kind = "common_value_repair"
            selected_records = value_records
            selected_objective = float(value_objective)
        if (
            message_objective is not None
            and message_objective > selected_objective + 1.0e-9
        ):
            selected_kind = "message_conditioned_repair"
            assert message_records is not None
            selected_records = message_records
            selected_objective = float(message_objective)

        # Both explicitly evaluated repair candidates are injected into the
        # next current factor domain.  This is the candidate-union safeguard:
        # Rm can add an option but cannot erase the R0 components available to
        # the corresponding B=0 cell.  It costs no worker query.
        pending = {
            agent_id: [value_records[agent_id]]
            for agent_id in range(self.config.num_agents)
        }
        if message_records is not None:
            for agent_id in range(self.config.num_agents):
                pending[agent_id].append(message_records[agent_id])
        self._pending_repair_records = pending

        self._repair_evaluations += 1
        self._repair_selected += int(selected_kind != "exact_joint")
        self._joint_selected += int(selected_kind == "exact_joint")
        self._value_repair_selected += int(
            selected_kind == "common_value_repair"
        )
        self._message_repair_selected += int(
            selected_kind == "message_conditioned_repair"
        )
        changed = sum(
            1
            for agent_id in range(self.config.num_agents)
            if selected_records[agent_id].candidate_id
            != private_elites[agent_id].candidate_id
        )
        self._repair_feedback_changes += int(changed)
        compact = {
            "macro_round": int(macro_round),
            "source_step": int(step),
            "target_step": int(step) + 1,
            "repair_mode": (
                "safeguarded_R0_union_Rm"
                if message_records is not None
                else "common_R0_only"
            ),
            "value_repair_objective_R0": float(value_objective),
            "message_repair_objective_Rm": (
                float(message_objective)
                if message_objective is not None
                else None
            ),
            "exact_joint_objective": float(joint_objective),
            "selected_kind": selected_kind,
            "selected_objective": float(selected_objective),
            "repair_selected": bool(selected_kind != "exact_joint"),
            "changed_agents_from_private_elite": int(changed),
            "value_conflicting_cut_edges": int(
                value_repair["conflicting_cut_edges"]
            ),
            "value_removed_count": len(value_repair["removed_ids"]),
            "message_conflicting_cut_edges": (
                int(message_repair["conflicting_cut_edges"])
                if message_repair is not None
                else None
            ),
            "message_removed_count": (
                len(message_repair["removed_ids"])
                if message_repair is not None
                else None
            ),
        }
        self._repair_trace.append(compact)
        self._event(
            "comm_repair_evaluation",
            method=self.config.reported_method,
            private_elite_candidate_ids=[
                private_elites[i].candidate_id
                for i in range(self.config.num_agents)
            ],
            exact_joint_candidate_ids=[
                joint_records[i].candidate_id
                for i in range(self.config.num_agents)
            ],
            repair_candidate_ids=[
                value_records[i].candidate_id
                for i in range(self.config.num_agents)
            ],
            value_repair_candidate_ids=[
                value_records[i].candidate_id
                for i in range(self.config.num_agents)
            ],
            message_repair_candidate_ids=(
                [
                    message_records[i].candidate_id
                    for i in range(self.config.num_agents)
                ]
                if message_records is not None
                else None
            ),
            value_repair_feasible=True,
            message_repair_feasible=(
                True if message_records is not None else None
            ),
            exact_joint_feasible=True,
            value_repair=value_repair,
            message_repair=message_repair,
            additional_worker_calls=0,
            additional_coordination_evaluations=(
                1 if message_records is not None else 0
            ),
            **compact
        )

        rows = []
        for agent_id in range(self.config.num_agents):
            rows.append(
                {
                    "macro_round": int(macro_round),
                    "source_step": int(step),
                    "target_step": int(step) + 1,
                    "agent_id": int(agent_id),
                    "elite_candidate_id": private_elites[agent_id].candidate_id,
                    "joint_candidate_id": joint_records[agent_id].candidate_id,
                    "repair_candidate_id": value_records[agent_id].candidate_id,
                    "value_repair_candidate_id": value_records[
                        agent_id
                    ].candidate_id,
                    "message_repair_candidate_id": (
                        message_records[agent_id].candidate_id
                        if message_records is not None
                        else None
                    ),
                    "selected_anchor_candidate_id": selected_records[
                        agent_id
                    ].candidate_id,
                    "selected_kind": selected_kind,
                    # Compatibility field consumed by the inherited event
                    # counter; v4.9 reports its own repair metrics below.
                    "joint_adopted": bool(
                        selected_kind == "exact_joint"
                        and joint_records[agent_id].candidate_id
                        != private_elites[agent_id].candidate_id
                    ),
                }
            )
        return dict(selected_records), rows

    def _query_wave(
        self,
        wave: str,
        macro_round: int,
        generators: Sequence[Any],
        engines: Sequence[Any],
        incumbent_records: Mapping[int, _RealizedRecord],
        solver: Any,
        graph: Any,
        pressure_maps: Optional[Mapping[int, Mapping[str, Mapping[int, float]]]],
    ) -> Dict[int, List[_RealizedRecord]]:
        queried = super()._query_wave(
            wave,
            macro_round,
            generators,
            engines,
            incumbent_records,
            solver,
            graph,
            pressure_maps,
        )
        injected = []
        injected_candidate_ids = {}
        pending = self._pending_repair_records
        for agent_id in range(self.config.num_agents):
            anchor = incumbent_records[agent_id]
            extras = list(pending.get(agent_id, [])) if pending else []
            queried[agent_id] = _deduplicate_records(
                [anchor] + extras + list(queried[agent_id])
            )
            repair_candidates = _deduplicate_records(
                ([anchor] if anchor.source == "comm_repair_evaluated" else [])
                + extras
            )
            if repair_candidates:
                injected.append(agent_id)
                injected_candidate_ids[str(agent_id)] = [
                    record.candidate_id for record in repair_candidates
                ]
        self._pending_repair_records = None
        if injected:
            injected_count = sum(
                len(values) for values in injected_candidate_ids.values()
            )
            self._repair_domain_injections += injected_count
            self._event(
                "comm_repair_domain_injection",
                method=self.config.reported_method,
                engine_round_context=int(macro_round),
                agent_ids=injected,
                candidate_ids_by_agent=injected_candidate_ids,
                injected_candidate_components=int(injected_count),
                worker_calls_added=0,
                explicitly_evaluated_before_injection=True,
            )
        return queried

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        rate = (
            float(self._repair_selected) / float(self._repair_evaluations)
            if self._repair_evaluations
            else 0.0
        )
        fields.update(
            {
                "schema_version": "SNSD-v1.5-comm-repair-run-v4.9",
                "soft_anchor_prototype": False,
                "communication_repair_closure": True,
                "factor_A_definition": (
                    "recurrent proposal communication under registered "
                    "real/zero/shuffled intervention"
                ),
                "factor_B_definition": (
                    "safeguarded candidate union J_union_R0_union_Rm"
                    if self.config.communication_conditioned_repair
                    else "common candidate union J_union_R0"
                ),
                "communication_conditioned_repair": bool(
                    self.config.communication_conditioned_repair
                ),
                "common_private_elite_union_repair_opportunity": True,
                "repair_score_formula": (
                    "norm(weight/(1+log1p(full_degree))) "
                    "+0.50*norm(message_pressure) "
                    "+0.35*norm(message_conflict_relief) "
                    "-0.25*abs(norm(message_opportunity)) "
                    "-0.20*norm(boundary_cut_degree)"
                    if self.config.communication_conditioned_repair
                    else "raw node weight; deterministic common tie break"
                ),
                "repair_selector": (
                    "B1:argmax_exact(J,R0,Rm); B0:argmax_exact(J,R0)"
                ),
                "message_branch_only_adds_candidate": True,
                "common_value_repair_preserved_in_B1": bool(
                    self.config.communication_conditioned_repair
                ),
                "repair_explicit_feasibility_and_objective_evaluation": True,
                "repair_cross_macro_only_after_recorded_factor_entry": True,
                "repair_additional_worker_calls": 0,
                "private_elite_memory": "persistent_across_macro_rounds",
                "repair_evaluations": int(self._repair_evaluations),
                "repair_selected": int(self._repair_selected),
                "exact_joint_selected": int(self._joint_selected),
                "common_value_repair_selected": int(
                    self._value_repair_selected
                ),
                "message_conditioned_repair_selected": int(
                    self._message_repair_selected
                ),
                "extra_coordination_evaluations": int(
                    self._extra_coordination_evaluations
                ),
                "repair_selected_rate": float(rate),
                "repair_feedback_changed_agent_slots": int(
                    self._repair_feedback_changes
                ),
                "repair_domain_injections": int(
                    self._repair_domain_injections
                ),
                "repair_trace": list(self._repair_trace),
            }
        )
        return fields


def _comm_repair_config(
    data_path: str,
    output_dir: str,
    method: str,
    seed: int,
    smoke: bool,
    oracle: bool,
    checkpoint: Optional[str],
    device: str,
    allow_untrained_smoke: bool,
    negotiation_depth: int,
    macro_rounds: int,
) -> CommunicationRepairConfig:
    aliases = {
        "A0B0": "FrozenAnchorSelectionOnly",
        "NoPeer": "NoPeerFinalShield",
    }
    requested = aliases.get(method, method)
    if requested not in COMM_REPAIR_METHODS:
        raise ValueError("unknown v4.9 method: {}".format(method))
    if requested in ("FrozenAnchorComm",):
        base_method = "Full"
    elif requested == "FrozenAnchorSelectionOnly":
        base_method = "SelectionOnly"
    elif requested == "NoPeerFinalShield":
        base_method = "NoPeerComm"
    elif requested == "OneShotComm":
        base_method = "Full"
    elif requested == "AlternatingConflictBenefit":
        base_method = "Full"
    else:
        base_method = requested

    base = _long_config(
        data_path=data_path,
        output_dir=output_dir,
        method=base_method,
        seed=seed,
        smoke=smoke,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=allow_untrained_smoke,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
    )
    values = dict(base.__dict__)
    # v4.9 B is repair conditioning, not frozen-anchor feedback.  Every cell
    # therefore traverses the same repair opportunity and exact comparison.
    values.update(
        {
            "frozen_anchor_comm": False,
            "frozen_anchor_selection_only": False,
            "one_shot_comm": requested == "OneShotComm",
            "alternating_conflict_benefit": (
                requested == "AlternatingConflictBenefit"
            ),
            "comm_repair_method_label": requested,
            "communication_conditioned_repair": requested
            not in _B0_METHODS,
            "no_peer_final_shield": False,
        }
    )
    return CommunicationRepairConfig(**values)


def run_comm_repair_experiment(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    allow_untrained_smoke: bool = False,
    negotiation_depth: int = 12,
    macro_rounds: int = 4,
) -> ExperimentResult:
    config = _comm_repair_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        smoke=smoke,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=allow_untrained_smoke,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
    )
    return CommunicationRepairProtocol(config).run()


__all__ = [
    "COMM_REPAIR_METHODS",
    "LONG_NEGOTIATION_DEPTHS",
    "CommunicationRepairConfig",
    "CommunicationRepairProtocol",
    "run_comm_repair_experiment",
]
