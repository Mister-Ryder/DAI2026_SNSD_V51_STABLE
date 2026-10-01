"""SNSD v5.0 structured tri-graph primal-dual coalition protocol.

The first six paid proposal steps remain private-elite exploitation.  Each of
the final six steps performs structured boundary-state lifting followed by at
least three coordination-only microcycles:

    local Top-K BoundaryPacket response
      -> boundary dual-price update
      -> sparse 2/3-agent coalition Max-Sum
      -> counterfactual local response.

Packets preserve candidate IDs, boundary node/arc sets, typed relations,
local keep/drop/refill/slack responses and a full-graph residual credit.  They
are not scalar message summaries.  A coalition executes a deterministic
delete--replace--refill closure inside the affected subgraphs and therefore
creates new realized local records rather than merely changing a next-step
anchor.  The complete graph verifies every candidate and the emitted tuple is

    argmax_exact {current reference, J, R0, Rm, coalition refill}.

Thus coordination can add candidates but cannot lower the safeguarded exact
objective.  Factor A controls semantic packet/price/credit messages.  Factor
B controls whether the structured coalition refill enters the candidate
union; B0 retains the common R0 repair.  All cells keep the same 288 paid
worker-query budget.  The dual prices are a discrete non-convex coordination
heuristic; no ADMM convergence claim is made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from snsd_core.data import load_arcs
from snsd_core.graph import build_conflict_graph, partition_equal_time
from snsd_core.solver import LocalSolveResult
from snsd_core.verifier import verify_selected_ids

from .base_protocol import (
    ExperimentResult,
    _RealizedRecord,
    _atomic_json,
    _sha256_file,
)
from .communication_repair_protocol import (
    CommunicationRepairConfig,
    CommunicationRepairProtocol,
    _comm_repair_config,
)
from .boundary_joint import run_structured_boundary_cycle


V50_FORMAL_METHODS = (
    "Full",
    "FrozenAnchorComm",
    "SelectionOnly",
    "FrozenAnchorSelectionOnly",
    "AlternatingConflictBenefit",
    "OneShotComm",
    "ShuffledComm",
    "NoPeerFinalShield",
)
_B0_METHODS = (
    "FrozenAnchorComm",
    "FrozenAnchorSelectionOnly",
    "NoPeerFinalShield",
)


@dataclass(frozen=True)
class JointNegotiationConfig(CommunicationRepairConfig):
    packet_top_k: int = 4
    microcycles: int = 3
    max_packet_top_k: int = 6
    max_microcycles: int = 4
    private_exploit_steps: int = 6
    max_coalition_deletions: int = 2
    structured_coalition_exchange: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.smoke:
            raise ValueError("v5.0 is registered only for full non-smoke C3")
        if int(self.negotiation_depth) != 12 or int(self.macro_round_count) != 4:
            raise ValueError("v5.0 formal protocol requires depth=12 and macros=4")
        if int(self.packet_top_k) <= 0:
            raise ValueError("packet_top_k must be positive")
        if int(self.max_packet_top_k) < int(self.packet_top_k):
            raise ValueError("max_packet_top_k must cover packet_top_k")
        if int(self.microcycles) < 3:
            raise ValueError("v5.0 requires at least three microcycles")
        if int(self.max_microcycles) < int(self.microcycles):
            raise ValueError("max_microcycles must cover microcycles")
        if int(self.max_coalition_deletions) not in (1, 2):
            raise ValueError("max_coalition_deletions must be one or two")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class StructuredJointProtocol(CommunicationRepairProtocol):
    config: JointNegotiationConfig

    def __init__(self, config: JointNegotiationConfig) -> None:
        super().__init__(config)
        self._factor_history = {}  # type: Dict[int, List[float]]
        self._v50_trace = []  # type: List[Dict[str, Any]]
        self._v50_microcycles = 0
        self._v50_payload_scalars = 0
        self._v50_payload_bytes = 0
        self._v50_coalition_proposals = 0
        self._v50_coalition_accepts = 0
        self._v50_refill_net_gain = 0.0
        self._v50_primal_residuals = []  # type: List[float]
        self._v50_dual_residuals = []  # type: List[float]
        self._v50_relation_counts = {}  # type: Dict[str, int]
        self._v50_refill_records = 0
        self._v50_no_downgrade_rejections = 0

    @property
    def _coalition_enabled(self) -> bool:
        return bool(
            self.config.structured_coalition_exchange
            and self.config.reported_method not in _B0_METHODS
        )

    def _semantic_mode(self, step: int) -> str:
        method = self.config.reported_method
        if method in (
            "SelectionOnly",
            "FrozenAnchorSelectionOnly",
            "NoPeerFinalShield",
        ):
            return "zero"
        if method == "ShuffledComm":
            return "shuffled"
        if method == "OneShotComm":
            return (
                "real"
                if int(step) == int(self.config.private_exploit_steps) - 1
                else "zero"
            )
        if method == "AlternatingConflictBenefit":
            return "conflict_only" if int(step) % 2 else "benefit_only"
        return "real"

    @staticmethod
    def _factor_reference_records(
        factor: Any, ordered_records: Sequence[Sequence[_RealizedRecord]]
    ) -> Dict[int, _RealizedRecord]:
        return {
            agent_id: ordered_records[agent_id][
                int(factor.incumbent_indices[agent_id])
            ]
            for agent_id in range(len(ordered_records))
        }

    def _verified_objective(
        self, records: Mapping[int, _RealizedRecord], label: str
    ) -> Tuple[Tuple[int, ...], float]:
        selected = self._selected_union(records)
        check = verify_selected_ids(self.graph, selected)
        if not check.feasible:
            raise RuntimeError("{} candidate infeasible: {}".format(label, check.message))
        return selected, float(self._objective(selected))

    def _records_from_cycle(
        self,
        cycle: Any,
        macro_round: int,
        step: int,
    ) -> Optional[Dict[int, _RealizedRecord]]:
        selected_by_agent = _field(cycle, "candidate_selected_ids_by_agent")
        if not selected_by_agent:
            selected_by_agent = _field(cycle, "selected_ids_by_agent")
        if not selected_by_agent or not bool(_field(cycle, "feasible", False)):
            return None
        records = {}
        union = []
        for agent_id in range(self.config.num_agents):
            raw = selected_by_agent.get(agent_id)
            if raw is None:
                raw = selected_by_agent.get(str(agent_id), ())
            local_ids = tuple(sorted(int(node) for node in raw))
            if any(
                int(self.partitions.partition_id[node]) != agent_id
                for node in local_ids
            ):
                raise RuntimeError("coalition refill returned cross-owner local record")
            union.extend(local_ids)
            generator = self._repair_generators[agent_id]
            boundary = set(int(node) for node in generator.boundary)
            signature = generator.make_signature(
                [node for node in local_ids if node in boundary]
            )
            result = LocalSolveResult(
                agent_id=agent_id,
                signature_hash=signature.signature_hash,
                status="feasible",
                selected_ids=local_ids,
                objective=self._objective(local_ids),
                runtime_s=0.0,
                attempted_moves=0,
                seed=self.config.seed,
                message="v5 deterministic structured coalition refill",
            )
            records[agent_id] = _RealizedRecord(
                signature,
                result,
                "v50_structured_coalition_refill",
                int(macro_round),
                "late_step_{}".format(step),
            )
        check = verify_selected_ids(self.graph, tuple(sorted(union)))
        if not check.feasible:
            raise RuntimeError("v5 coalition union infeasible: {}".format(check.message))
        return records

    def _adapt_depth(self, macro_round: int, factor_value: float) -> Tuple[int, int, bool]:
        history = self._factor_history.setdefault(int(macro_round), [])
        history.append(float(factor_value))
        plateau = False
        if len(history) >= 3:
            improvement = float(history[-1] - history[-3])
            threshold = 5.0e-5 * max(1.0, abs(float(history[-1])))
            plateau = bool(improvement <= threshold)
        top_k = int(self.config.max_packet_top_k if plateau else self.config.packet_top_k)
        micro = int(self.config.max_microcycles if plateau else self.config.microcycles)
        return top_k, micro, plateau

    def _early_private_response(
        self,
        private_elites: Mapping[int, _RealizedRecord],
        macro_round: int,
        step: int,
    ) -> Tuple[Dict[int, _RealizedRecord], List[Dict[str, Any]]]:
        rows = []
        for agent_id in range(self.config.num_agents):
            elite = private_elites[agent_id]
            rows.append(
                {
                    "macro_round": int(macro_round),
                    "source_step": int(step),
                    "target_step": int(step) + 1,
                    "agent_id": int(agent_id),
                    "stage": "private_elite_exploitation",
                    "elite_candidate_id": elite.candidate_id,
                    "joint_candidate_id": elite.candidate_id,
                    "selected_anchor_candidate_id": elite.candidate_id,
                    "joint_adopted": False,
                }
            )
        return dict(private_elites), rows

    def _soft_anchor_decision(
        self,
        factor: Any,
        ordered_records: Sequence[Sequence[_RealizedRecord]],
        private_elites: Mapping[int, _RealizedRecord],
        macro_round: int,
        step: int,
    ) -> Tuple[Dict[int, _RealizedRecord], List[Dict[str, Any]]]:
        target_step = int(step) + 1
        if target_step < int(self.config.private_exploit_steps):
            self._factor_history.setdefault(int(macro_round), []).append(
                float(factor.best_value)
            )
            self._pending_repair_records = None
            return self._early_private_response(private_elites, macro_round, step)

        joint_records = self._chosen_records(factor, ordered_records)
        reference_records = self._factor_reference_records(factor, ordered_records)
        _, reference_objective = self._verified_objective(
            reference_records, "current_reference"
        )
        _, joint_objective = self._verified_objective(joint_records, "exact_joint_J")

        value_ids, value_records, value_metrics = self._repair_private_elite_union(
            private_elites,
            macro_round,
            step,
            communication_conditioned=False,
        )
        value_objective = float(self._objective(value_ids))

        semantic_mode = self._semantic_mode(step)
        message_records = None
        message_objective = None
        message_metrics = None
        channels = None
        message_hashes = []
        if self.config.reported_method != "NoPeerFinalShield":
            channels, message_hashes = self._repair_messages(macro_round, step)
        if self.config.reported_method != "NoPeerFinalShield":
            message_ids, message_records, message_metrics = (
                self._repair_private_elite_union(
                    private_elites,
                    macro_round,
                    step,
                    communication_conditioned=True,
                )
            )
            message_objective = float(self._objective(message_ids))

        top_k, micro_rounds, plateau = self._adapt_depth(
            macro_round, float(factor.best_value)
        )
        cycle = None
        coalition_records = None
        coalition_objective = None
        if self.config.reported_method != "NoPeerFinalShield":
            cycle = run_structured_boundary_cycle(
                graph=self.graph,
                partitions=self.partitions,
                arcs=self.dataset.arcs,
                generators=self._repair_generators,
                private_elites=private_elites,
                current_records=ordered_records,
                base_records=joint_records,
                message_channels=channels,
                semantic_mode=semantic_mode,
                top_k=top_k,
                micro_rounds=micro_rounds,
                max_deletions=int(self.config.max_coalition_deletions),
                seed=int(self.config.seed),
                macro_round=int(macro_round),
                step=int(step),
            )
            coalition_records = self._records_from_cycle(cycle, macro_round, step)
            if coalition_records is not None:
                _, coalition_objective = self._verified_objective(
                    coalition_records, "structured_coalition"
                )

        candidates = [
            ("current_reference", reference_records, reference_objective),
            ("exact_joint_J", joint_records, joint_objective),
        ]
        candidates.append(("common_repair_R0", value_records, value_objective))
        if message_records is not None and message_objective is not None:
            candidates.append(("message_repair_Rm", message_records, message_objective))
        if (
            self._coalition_enabled
            and coalition_records is not None
            and coalition_objective is not None
        ):
            candidates.append(
                ("structured_coalition_C", coalition_records, coalition_objective)
            )

        selected_kind, selected_records, selected_objective = candidates[0]
        for kind, records, objective in candidates[1:]:
            if float(objective) > float(selected_objective) + 1.0e-9:
                selected_kind, selected_records, selected_objective = (
                    kind,
                    records,
                    float(objective),
                )
        if float(selected_objective) + 1.0e-9 < float(reference_objective):
            raise RuntimeError("v5 exact candidate union violated monotonic safeguard")

        pending = {agent_id: [] for agent_id in range(self.config.num_agents)}
        for agent_id in range(self.config.num_agents):
            pending[agent_id].append(value_records[agent_id])
        if message_records is not None:
            for agent_id in range(self.config.num_agents):
                pending[agent_id].append(message_records[agent_id])
        if self._coalition_enabled and coalition_records is not None:
            for agent_id in range(self.config.num_agents):
                pending[agent_id].append(coalition_records[agent_id])
                self._v50_refill_records += 1
        self._pending_repair_records = pending if any(pending.values()) else None

        micro_trace = list(_field(cycle, "micro_trace", []) or []) if cycle is not None else []
        payload_scalars = int(_field(cycle, "payload_scalars", 0) or 0) if cycle is not None else 0
        payload_bytes = int(_field(cycle, "payload_bytes", 8 * payload_scalars) or 0) if cycle is not None else 0
        primal = float(_field(cycle, "primal_residual", 0.0) or 0.0) if cycle is not None else 0.0
        dual = float(_field(cycle, "dual_residual", 0.0) or 0.0) if cycle is not None else 0.0
        messages_base = int(_field(cycle, "proposal_count", 0) or 0) if cycle is not None else 0
        members = list(_field(cycle, "members", []) or []) if cycle is not None else []
        removed_ids = list(_field(cycle, "removed_ids", []) or []) if cycle is not None else []
        added_ids = list(_field(cycle, "added_ids", []) or []) if cycle is not None else []
        refill_ids = list(_field(cycle, "refill_ids", []) or []) if cycle is not None else []
        relation_counts = dict(_field(cycle, "relation_counts", {}) or {}) if cycle is not None else {}
        coalition_delta = (
            float(coalition_objective - joint_objective)
            if coalition_objective is not None
            else 0.0
        )
        coalition_accepted = bool(selected_kind == "structured_coalition_C")
        self._v50_microcycles += len(micro_trace) if micro_trace else (
            micro_rounds if cycle is not None else 0
        )
        self._v50_payload_scalars += payload_scalars
        self._v50_payload_bytes += payload_bytes
        self._v50_coalition_proposals += messages_base
        self._v50_coalition_accepts += int(coalition_accepted)
        self._v50_refill_net_gain += max(0.0, coalition_delta)
        self._v50_primal_residuals.append(primal)
        self._v50_dual_residuals.append(dual)
        for relation, count in relation_counts.items():
            self._v50_relation_counts[str(relation)] = (
                self._v50_relation_counts.get(str(relation), 0) + int(count)
            )
        if coalition_records is not None and not coalition_accepted:
            self._v50_no_downgrade_rejections += 1

        compact = {
            "macro_round": int(macro_round),
            "source_step": int(step),
            "target_step": int(target_step),
            "stage": "late_structured_coordination",
            "semantic_mode": semantic_mode,
            "plateau": bool(plateau),
            "packet_top_k": int(top_k),
            "microcycles": int(micro_rounds),
            "factor_B_coalition_enabled": bool(self._coalition_enabled),
            "reference_objective": float(reference_objective),
            "exact_joint_objective_J": float(joint_objective),
            "common_repair_objective_R0": float(value_objective),
            "message_repair_objective_Rm": message_objective,
            "coalition_objective_C": coalition_objective,
            "selected_kind": selected_kind,
            "selected_objective": float(selected_objective),
            "coalition_members": members,
            "coalition_size": len(members),
            "removed_ids": removed_ids,
            "added_ids": added_ids,
            "refill_ids": refill_ids,
            "coalition_exact_delta_vs_J": float(coalition_delta),
            "coalition_accepted": coalition_accepted,
            "primal_residual": primal,
            "dual_residual": dual,
            "payload_scalars": payload_scalars,
            "payload_bytes": payload_bytes,
            "relation_counts": relation_counts,
            "message_hashes": message_hashes,
        }
        self._v50_trace.append(compact)
        self._event(
            "v50_structured_joint_cycle",
            packets=_field(cycle, "packets", None) if cycle is not None else None,
            micro_trace=micro_trace,
            worker_calls_added=0,
            exact_candidate_union=True,
            query_budget_unchanged=True,
            **compact
        )

        rows = []
        for agent_id in range(self.config.num_agents):
            rows.append(
                {
                    "macro_round": int(macro_round),
                    "source_step": int(step),
                    "target_step": int(target_step),
                    "agent_id": int(agent_id),
                    "stage": "late_structured_coordination",
                    "elite_candidate_id": private_elites[agent_id].candidate_id,
                    "joint_candidate_id": joint_records[agent_id].candidate_id,
                    "selected_anchor_candidate_id": selected_records[agent_id].candidate_id,
                    "selected_kind": selected_kind,
                    "joint_adopted": bool(
                        selected_kind == "exact_joint_J"
                        and selected_records[agent_id].candidate_id
                        != private_elites[agent_id].candidate_id
                    ),
                }
            )
        return dict(selected_records), rows

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        mean_primal = (
            sum(self._v50_primal_residuals) / len(self._v50_primal_residuals)
            if self._v50_primal_residuals
            else 0.0
        )
        mean_dual = (
            sum(self._v50_dual_residuals) / len(self._v50_dual_residuals)
            if self._v50_dual_residuals
            else 0.0
        )
        fields.update(
            {
                "schema_version": "SNSD-v1.5-structured-joint-run-v5.0",
                "structured_boundary_state_lifting": True,
                "tri_graph_primal_dual_protocol": True,
                "factor_A_definition": "structured semantic packet-price-credit messages",
                "factor_B_definition": "late coalitional delete-replace-refill versus common repair",
                "factor_A_enabled": self._semantic_mode(6) != "zero",
                "factor_B_enabled": bool(self._coalition_enabled),
                "private_subgraph_state": "Top-K complete realized local elites",
                "shared_boundary_state": "typed BoundaryPacket substructures and responses",
                "public_global_state": "full-graph residual, exact feasibility, exact objective credit",
                "boundary_relation_types": [
                    "conflict", "complement", "substitution", "coalition"
                ],
                "v50_private_exploit_steps": int(self.config.private_exploit_steps),
                "v50_microcycles": int(self._v50_microcycles),
                "v50_payload_scalars": int(self._v50_payload_scalars),
                "v50_payload_bytes": int(self._v50_payload_bytes),
                "v50_coalition_proposals": int(self._v50_coalition_proposals),
                "v50_coalition_accepts": int(self._v50_coalition_accepts),
                "v50_refill_records": int(self._v50_refill_records),
                "v50_refill_net_gain": float(self._v50_refill_net_gain),
                "v50_primal_residual_mean": float(mean_primal),
                "v50_dual_residual_mean": float(mean_dual),
                "v50_relation_counts": dict(self._v50_relation_counts),
                "v50_no_downgrade_rejections": int(
                    self._v50_no_downgrade_rejections
                ),
                "v50_exact_candidate_union": "argmax(current,J,R0,Rm,coalition)",
                "v50_monotone_exact_commit": True,
                "v50_additional_worker_calls": 0,
                "v50_paid_worker_query_budget": 288,
                "v50_trace": list(self._v50_trace),
                "admm_convergence_claimed": False,
                "algorithm_class": "discrete nonconvex tri-graph primal-dual heuristic",
            }
        )
        return fields

    def run(self) -> ExperimentResult:
        result = super().run()
        receipt_path = Path(result.receipt_path)
        receipt = {
            "schema_version": "SNSD-v1.5-structured-joint-receipt-v5.0",
            "status": "complete",
            "method": self.config.reported_method,
            "seed": self.config.seed,
            "artifacts": {
                "events.jsonl": {
                    "sha256": _sha256_file(Path(result.events_path)),
                    "bytes": Path(result.events_path).stat().st_size,
                },
                "summary.json": {
                    "sha256": _sha256_file(Path(result.summary_path)),
                    "bytes": Path(result.summary_path).stat().st_size,
                },
            },
        }
        _atomic_json(receipt_path, receipt)
        return result


def _structured_joint_config(
    data_path: str,
    output_dir: str,
    method: str,
    seed: int,
    oracle: bool,
    checkpoint: Optional[str],
    device: str,
    negotiation_depth: int,
    macro_rounds: int,
    packet_top_k: int,
    microcycles: int,
    max_packet_top_k: int,
    max_microcycles: int,
) -> JointNegotiationConfig:
    if method not in V50_FORMAL_METHODS:
        raise ValueError("unknown v5.0 method: {}".format(method))
    base = _comm_repair_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        smoke=False,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=False,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
    )
    values = dict(base.__dict__)
    values.update(
        {
            "packet_top_k": int(packet_top_k),
            "microcycles": int(microcycles),
            "max_packet_top_k": int(max_packet_top_k),
            "max_microcycles": int(max_microcycles),
            "structured_coalition_exchange": method not in _B0_METHODS,
        }
    )
    return JointNegotiationConfig(**values)


def run_structured_joint_experiment(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    negotiation_depth: int = 12,
    macro_rounds: int = 4,
    packet_top_k: int = 4,
    microcycles: int = 3,
    max_packet_top_k: int = 6,
    max_microcycles: int = 4,
) -> ExperimentResult:
    config = _structured_joint_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
        packet_top_k=packet_top_k,
        microcycles=microcycles,
        max_packet_top_k=max_packet_top_k,
        max_microcycles=max_microcycles,
    )
    return StructuredJointProtocol(config).run()


def preflight_v50_full_c3(
    data_path: str,
    method: str = "Full",
    seed: int = 11,
    packet_top_k: int = 4,
    microcycles: int = 3,
) -> Dict[str, Any]:
    started = time.perf_counter()
    data = Path(data_path).resolve()
    dataset = load_arcs(str(data))
    graph = build_conflict_graph(dataset.arcs)
    partitions = partition_equal_time(
        dataset.arcs, graph, k=6, horizon_start=0, horizon_end=259200
    )
    config = _structured_joint_config(
        data_path=str(data),
        output_dir="preflight-not-created",
        method=method,
        seed=seed,
        oracle=True,
        checkpoint=None,
        device="cpu",
        negotiation_depth=12,
        macro_rounds=4,
        packet_top_k=packet_top_k,
        microcycles=microcycles,
        max_packet_top_k=max(6, packet_top_k),
        max_microcycles=max(4, microcycles),
    )
    if partitions.has_non_adjacent_cuts:
        raise ValueError("v5.0 complete C3 preflight requires adjacent chain cuts")
    if config.expected_online_calls != 288:
        raise RuntimeError("v5.0 query budget preflight failed")
    return {
        "status": "ready",
        "schema_version": "SNSD-v1.5-v5.0-full-c3-preflight-v1",
        "data_path": str(data),
        "dataset_sha256": _sha256_file(data),
        "method": method,
        "seed_is_launch_metadata_only": int(seed),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "cut_edges": len(partitions.cut_edges),
        "num_agents": 6,
        "negotiation_depth": 12,
        "macro_rounds": 4,
        "expected_online_calls": 288,
        "packet_top_k": int(packet_top_k),
        "microcycles": int(microcycles),
        "non_smoke": True,
        "runtime_s": time.perf_counter() - started,
    }


__all__ = [
    "V50_FORMAL_METHODS",
    "JointNegotiationConfig",
    "StructuredJointProtocol",
    "run_structured_joint_experiment",
    "preflight_v50_full_c3",
]
