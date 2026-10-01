"""Runnable oracle/no-learning TriGraph-CBCR base_protocol loop.

This module is intentionally conservative: it reuses the verified v1 data,
graph, conditioned local worker, and verifier, while all joint decisions use
the current-domain exact chain factor graph.  No historical candidate archive
or repair closure is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from snsd_core.candidates import BoundarySignature, CandidateGenerator
from snsd_core.data import load_arcs
from snsd_core.graph import build_conflict_graph, partition_equal_time
from snsd_core.solver import ConditionedLocalSolver, LocalSolveResult
from snsd_core.verifier import verify_selected_ids

from .factor_graph import CandidateAction, ChainMaxSumResult, solve_chain_max_sum
from .messages_base import (
    BoundaryMessage,
    BoundaryState,
    ProposalEngine,
    apply_message_intervention,
    materialize_intent,
    solver_compliance,
)


METHODS = (
    "Full",
    "NoPeerComm",
    "SelectionOnly",
    "ShuffledComm",
    "NoGlobalContext",
    "ConflictMessageOnly",
    "BenefitMessageOnly",
)


@dataclass(frozen=True)
class MethodBehavior:
    method: str
    peer_pressure: str
    wave_b_intervention: str
    global_context: bool
    final_selection: str
    wave_a_message_pass: bool


_BEHAVIORS = {
    "Full": MethodBehavior("Full", "full", "real", True, "max_sum", True),
    "NoPeerComm": MethodBehavior(
        "NoPeerComm", "none", "zero", True, "independent", False
    ),
    "SelectionOnly": MethodBehavior(
        "SelectionOnly", "none", "zero", True, "max_sum", False
    ),
    "ShuffledComm": MethodBehavior(
        "ShuffledComm", "full", "shuffled", True, "max_sum", True
    ),
    "NoGlobalContext": MethodBehavior(
        "NoGlobalContext", "full", "real", False, "max_sum", True
    ),
    "ConflictMessageOnly": MethodBehavior(
        "ConflictMessageOnly", "conflict", "real", True, "max_sum", True
    ),
    "BenefitMessageOnly": MethodBehavior(
        "BenefitMessageOnly", "benefit", "real", True, "max_sum", True
    ),
}


def normalize_method(value: str) -> str:
    token = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        method.lower().replace("-", "").replace("_", ""): method
        for method in METHODS
    }
    aliases["fullv15"] = "Full"
    if token not in aliases:
        raise ValueError("unknown method {!r}; choose from {}".format(value, METHODS))
    return aliases[token]


def method_behavior(method: str) -> MethodBehavior:
    return _BEHAVIORS[normalize_method(method)]


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: str
    output_dir: str
    method: str = "Full"
    seed: int = 11
    smoke: bool = False
    oracle: bool = False
    num_agents: int = 6
    horizon_start: int = 0
    horizon_end: int = 259200

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_method(self.method))
        if int(self.num_agents) != 6:
            raise ValueError("v1.5 C3 protocol requires exactly six agents")

    @property
    def macro_rounds(self) -> int:
        return 1 if self.smoke else 2

    @property
    def query_budget(self) -> int:
        return 1 if self.smoke else 3

    @property
    def proposal_pool(self) -> int:
        return 4 if self.smoke else 24

    @property
    def expected_online_calls(self) -> int:
        return self.macro_rounds * 2 * self.num_agents * self.query_budget


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: str
    events_path: str
    summary_path: str
    receipt_path: str
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class _RealizedRecord:
    signature: BoundarySignature
    result: LocalSolveResult
    source: str
    macro_round: int
    wave: str

    @property
    def candidate_id(self) -> str:
        return str(self.signature.signature_hash)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(str(temporary), "w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    os.replace(str(temporary), str(path))


class _EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = open(str(path), "w", encoding="utf-8")
        self.count = 0

    def write(self, event_type: str, **payload: Any) -> None:
        row = {"event_index": self.count, "event_type": event_type}
        row.update(payload)
        self._stream.write(
            json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._stream.flush()
        self.count += 1

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


def _deduplicate_records(records: Iterable[_RealizedRecord]) -> List[_RealizedRecord]:
    best = {}  # type: Dict[str, _RealizedRecord]
    for record in records:
        key = record.candidate_id
        previous = best.get(key)
        if previous is None:
            best[key] = record
            continue
        previous_feasible = previous.result.status == "feasible"
        current_feasible = record.result.status == "feasible"
        if current_feasible and (
            not previous_feasible
            or float(record.result.objective) > float(previous.result.objective)
        ):
            best[key] = record
    return sorted(best.values(), key=lambda item: item.candidate_id)


class BaseNegotiationProtocol:
    """Current-domain two-wave v1.5 base_protocol with exact oracle pressure."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.behavior = method_behavior(config.method)
        self.output_dir = Path(config.output_dir).resolve()
        self.events_path = self.output_dir / "events.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.receipt_path = self.output_dir / "receipt.json"
        self.online_calls = 0
        self.warmup_calls = 0
        self.physical_worker_runtime_s = 0.0
        self.message_passes = 0
        self.directed_messages = 0
        self.message_payload_scalars = 0
        self.objective_trace = []  # type: List[float]
        self._event_log = None  # type: Optional[_EventLog]

    def _prepare_output(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError("output directory is not empty: {}".format(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._event_log = _EventLog(self.events_path)

    def _event(self, event_type: str, **payload: Any) -> None:
        assert self._event_log is not None
        self._event_log.write(event_type, **payload)

    @staticmethod
    def _cut_edges_by_link(partitions: Any) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        links = [[] for _ in range(partitions.k - 1)]
        for raw_u, raw_v in partitions.cut_edges:
            u, v = int(raw_u), int(raw_v)
            pu = int(partitions.partition_id[u])
            pv = int(partitions.partition_id[v])
            if abs(pu - pv) != 1:
                raise ValueError("chain Max-Sum cannot accept non-adjacent cut edges")
            if pu < pv:
                links[pu].append((u, v))
            else:
                links[pv].append((v, u))
        return tuple(tuple(sorted(edges)) for edges in links)

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
        result = solver.solve(signature.agent_id, signature)
        self.physical_worker_runtime_s += float(result.runtime_s)
        if phase == "online":
            self.online_calls += 1
        else:
            self.warmup_calls += 1
        if result.status == "feasible":
            check = verify_selected_ids(graph, result.selected_ids)
            if not check.feasible:
                raise RuntimeError("local worker returned invalid solution: {}".format(check.message))
        self._event(
            "worker_query",
            phase=phase,
            macro_round=macro_round,
            wave=wave,
            agent_id=signature.agent_id,
            slot=slot,
            logical_online_call=self.online_calls if phase == "online" else None,
            signature_hash=signature.signature_hash,
            status=result.status,
            local_objective=result.objective,
            selected_count=len(result.selected_ids),
            runtime_s=result.runtime_s,
            intent=intent_payload,
        )
        return _RealizedRecord(signature, result, phase, macro_round, wave)

    def _factor_pass(
        self,
        records_by_agent: Mapping[int, Sequence[_RealizedRecord]],
        incumbent_ids: Mapping[int, str],
        safe_ids: Mapping[int, str],
        cut_edges_by_link: Sequence[Sequence[Tuple[int, int]]],
        label: str,
        count_messages: bool = True,
        require_incumbent: bool = True,
    ) -> Tuple[ChainMaxSumResult, Tuple[Tuple[_RealizedRecord, ...], ...]]:
        ordered_records = []
        action_tables = []
        for agent_id in range(self.config.num_agents):
            records = _deduplicate_records(records_by_agent[agent_id])
            ordered_records.append(tuple(records))
            actions = []
            for record in records:
                actions.append(
                    CandidateAction(
                        agent_id=agent_id,
                        candidate_id=record.candidate_id,
                        local_objective=float(record.result.objective),
                        selected_boundary=frozenset(record.signature.selected_ids),
                        is_incumbent=(
                            require_incumbent
                            and record.candidate_id == incumbent_ids[agent_id]
                        ),
                        is_safe=record.candidate_id == safe_ids[agent_id],
                        is_feasible=record.result.status == "feasible",
                    )
                )
            action_tables.append(actions)
        started = time.perf_counter()
        result = solve_chain_max_sum(
            action_tables,
            cut_edges_by_link,
            require_incumbent=require_incumbent,
            require_safe_anchor=True,
        )
        runtime = time.perf_counter() - started
        scalars = sum(len(table) for table in action_tables[1:]) + sum(
            len(table) for table in action_tables[:-1]
        )
        if count_messages:
            directed = 2 * (self.config.num_agents - 1)
            self.message_passes += 1
            self.directed_messages += directed
            self.message_payload_scalars += scalars
        else:
            directed = 0
        self._event(
            "factor_pass",
            label=label,
            counted_as_communication=count_messages,
            directed_messages=directed,
            payload_scalars=scalars if count_messages else 0,
            candidate_counts=[len(table) for table in action_tables],
            best_value=result.best_value,
            best_candidate_ids=[item.candidate_id for item in result.best_actions],
            runtime_s=runtime,
        )
        return result, tuple(ordered_records)

    @staticmethod
    def _chosen_records(
        factor: ChainMaxSumResult,
        ordered_records: Sequence[Sequence[_RealizedRecord]],
    ) -> Dict[int, _RealizedRecord]:
        return {
            agent_id: ordered_records[agent_id][factor.best_indices[agent_id]]
            for agent_id in range(len(ordered_records))
        }

    @staticmethod
    def _pressure_from_values(
        boundary: Sequence[int],
        records: Sequence[_RealizedRecord],
        values: Sequence[float],
    ) -> Dict[int, float]:
        pressure = {}
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
            pressure[int(node_id)] = max(on) - max(off) if on and off else 0.0
        return pressure

    def _pressure_maps(
        self,
        generators: Sequence[CandidateGenerator],
        ordered_records: Sequence[Sequence[_RealizedRecord]],
        factor: Optional[ChainMaxSumResult],
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        maps = {}
        for agent_id, generator in enumerate(generators):
            records = ordered_records[agent_id]
            local_values = [float(item.result.objective) for item in records]
            benefit = self._pressure_from_values(generator.boundary, records, local_values)
            if factor is None:
                full = dict(benefit)
                conflict = {node_id: 0.0 for node_id in generator.boundary}
                opportunity = {node_id: 0.0 for node_id in generator.boundary}
            else:
                full = self._pressure_from_values(
                    generator.boundary,
                    records,
                    factor.forced_best_completions[agent_id],
                )
                neg_cost = [
                    -float(value) if math.isfinite(float(value)) else float("-inf")
                    for value in factor.opportunity_costs[agent_id]
                ]
                conflict = self._pressure_from_values(
                    generator.boundary, records, neg_cost
                )
                opportunity = {
                    node_id: abs(float(full[node_id] - benefit[node_id]))
                    for node_id in generator.boundary
                }
            maps[agent_id] = {
                "full": full,
                "benefit": benefit,
                "conflict": conflict,
                "opportunity": opportunity,
            }
        return maps

    def _state(
        self,
        agent_id: int,
        generator: CandidateGenerator,
        incumbent: _RealizedRecord,
        arcs: Sequence[Any],
        graph: Any,
        pressure: Optional[Mapping[str, Mapping[int, float]]] = None,
    ) -> BoundaryState:
        raw = {node_id: float(arcs[node_id].weight) for node_id in generator.boundary}
        local = {
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

        if self.behavior.peer_pressure == "conflict":
            raw = {node_id: 0.0 for node_id in generator.boundary}
            local = {node_id: 0.0 for node_id in generator.boundary}
        if not self.behavior.global_context:
            # Raw/local features remain available, but complete-graph cut-degree
            # context is masked in this no-learning adapter.
            cut_conflict = {node_id: 0.0 for node_id in generator.boundary}
        prior = pressure or {}
        return BoundaryState.from_maps(
            agent_id=agent_id,
            boundary_order=generator.boundary,
            incumbent_selected=incumbent.signature.selected_ids,
            raw_benefit=raw,
            local_marginal=local,
            conflict_score=cut_conflict,
            prior_opportunity_cost=prior.get("opportunity", {}),
            prior_pressure=prior.get(self.behavior.peer_pressure, {}),
        )

    def _message(
        self,
        generator: CandidateGenerator,
        pressure: Mapping[str, Mapping[int, float]],
    ) -> BoundaryMessage:
        mode = self.behavior.peer_pressure
        if mode == "none":
            selected_pressure = {node_id: 0.0 for node_id in generator.boundary}
            opportunity = {node_id: 0.0 for node_id in generator.boundary}
        elif mode == "benefit":
            selected_pressure = pressure["benefit"]
            opportunity = {node_id: 0.0 for node_id in generator.boundary}
        elif mode == "conflict":
            selected_pressure = pressure["conflict"]
            opportunity = pressure["opportunity"]
        else:
            selected_pressure = pressure["full"]
            opportunity = pressure["opportunity"]
        return BoundaryMessage.from_maps(
            boundary_order=generator.boundary,
            opportunity_cost=opportunity,
            pressure=selected_pressure,
            neighbor_recovery=opportunity,
            conflict_score=pressure["conflict"],
        )

    @staticmethod
    def _intent_score(
        signature: BoundarySignature,
        anchor: BoundarySignature,
        value_map: Mapping[int, float],
    ) -> float:
        chosen = set(signature.selected_ids)
        anchor_chosen = set(anchor.selected_ids)
        return float(
            sum(float(value_map.get(node, 0.0)) for node in chosen.difference(anchor_chosen))
            - sum(float(value_map.get(node, 0.0)) for node in anchor_chosen.difference(chosen))
        )

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
                local_pressure,
            )
            if wave == "A":
                intents = engines[agent_id].wave_a(
                    state, macro_round, count=self.config.proposal_pool
                )
                intervention = "none"
                scoring = {
                    node_id: float(state.local_marginal[index])
                    - 0.10 * float(state.conflict_score[index])
                    for index, node_id in enumerate(state.boundary_order)
                }
            else:
                assert local_pressure is not None
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
                scoring = {
                    node_id: float(state.local_marginal[index])
                    + float(message_pressure.get(node_id, 0.0))
                    for index, node_id in enumerate(state.boundary_order)
                }

            pool = []
            for intent in intents:
                materialized = materialize_intent(
                    intent, generator, anchor_signature=anchor_record.signature
                )
                score = self._intent_score(
                    materialized.signature, anchor_record.signature, scoring
                )
                pool.append((score, intent.intent_hash, materialized))
            # Degenerate/small smoke graphs may not have enough distinct
            # boundary perturbations.  Repeated anchor calls remain logical
            # calls and keep the registered per-agent budget exact.
            if not pool:
                pool.append((0.0, "anchor-fallback", None))
            pool.sort(key=lambda item: (-item[0], item[1]))
            self._event(
                "proposal_pool",
                macro_round=macro_round,
                wave=wave,
                agent_id=agent_id,
                intervention=intervention,
                state_hash=state.state_hash,
                pool_size=len(pool),
                predictions=[
                    {
                        "intent_hash": key,
                        "score": score,
                        "signature_hash": (
                            item.signature.signature_hash
                            if item is not None
                            else anchor_record.signature.signature_hash
                        ),
                        "intent": item.intent.to_dict() if item is not None else None,
                    }
                    for score, key, item in pool
                ],
            )
            selected = []
            unique_hashes = set()
            for entry in pool:
                signature = (
                    entry[2].signature
                    if entry[2] is not None
                    else anchor_record.signature
                )
                if signature.signature_hash in unique_hashes:
                    continue
                selected.append(entry)
                unique_hashes.add(signature.signature_hash)
                if len(selected) == self.config.query_budget:
                    break
            while len(selected) < self.config.query_budget:
                selected.append(pool[len(selected) % len(pool)])

            for slot, (_, _, materialized) in enumerate(selected):
                signature = (
                    materialized.signature
                    if materialized is not None
                    else anchor_record.signature
                )
                intent_payload = (
                    materialized.to_dict() if materialized is not None else None
                )
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
                    self._event(
                        "solver_compliance",
                        macro_round=macro_round,
                        wave=wave,
                        agent_id=agent_id,
                        compliance=solver_compliance(materialized, record.result).to_dict(),
                    )
                queried[agent_id].append(record)
                self.objective_trace.append(float(self.incumbent_objective))
        return queried

    def _objective(self, selected_ids: Iterable[int]) -> float:
        return float(sum(float(self.dataset.arcs[int(node)].weight) for node in selected_ids))

    def _shield(
        self,
        chosen: Mapping[int, _RealizedRecord],
        generators: Sequence[CandidateGenerator],
    ) -> Tuple[Tuple[int, ...], Dict[int, _RealizedRecord], Dict[str, Any]]:
        selected = set()
        for record in chosen.values():
            selected.update(int(node) for node in record.result.selected_ids)
        pre_objective = self._objective(selected)
        removed = []
        for raw_u, raw_v in sorted(
            (tuple(map(int, edge)) for edge in self.partitions.cut_edges)
        ):
            if raw_u not in selected or raw_v not in selected:
                continue
            pu = int(self.partitions.partition_id[raw_u])
            pv = int(self.partitions.partition_id[raw_v])
            priority_u = (float(self.dataset.arcs[raw_u].weight), -pu, -raw_u)
            priority_v = (float(self.dataset.arcs[raw_v].weight), -pv, -raw_v)
            loser = raw_v if priority_u >= priority_v else raw_u
            if loser in selected:
                selected.remove(loser)
                removed.append(loser)
        final_ids = tuple(sorted(selected))
        check = verify_selected_ids(self.graph, final_ids)
        if not check.feasible:
            raise RuntimeError("safety shield failed: {}".format(check.message))
        post_objective = self._objective(final_ids)
        if not removed:
            updated = dict(chosen)
        else:
            updated = {}
            for agent_id in range(self.config.num_agents):
                local_nodes = [
                    node
                    for node in final_ids
                    if int(self.partitions.partition_id[node]) == agent_id
                ]
                boundary = set(generators[agent_id].boundary)
                signature = generators[agent_id].make_signature(
                    [node for node in local_nodes if node in boundary]
                )
                result = LocalSolveResult(
                    agent_id=agent_id,
                    signature_hash=signature.signature_hash,
                    status="feasible",
                    selected_ids=tuple(local_nodes),
                    objective=self._objective(local_nodes),
                    runtime_s=0.0,
                    attempted_moves=0,
                    seed=self.config.seed,
                    message="deterministic non-optimizing safety shield",
                )
                updated[agent_id] = _RealizedRecord(
                    signature, result, "shield", -1, "commit"
                )
        metrics = {
            "pre_shield_objective": pre_objective,
            "post_shield_objective": post_objective,
            "shield_loss": pre_objective - post_objective,
            "removed_ids": sorted(removed),
            "removed_weight": self._objective(removed),
        }
        return final_ids, updated, metrics

    def run(self) -> ExperimentResult:
        started = time.perf_counter()
        self._prepare_output()
        try:
            data_path = Path(self.config.data_path).resolve()
            self.dataset = load_arcs(str(data_path))
            self.graph = build_conflict_graph(self.dataset.arcs)
            self.partitions = partition_equal_time(
                self.dataset.arcs,
                self.graph,
                k=self.config.num_agents,
                horizon_start=self.config.horizon_start,
                horizon_end=self.config.horizon_end,
            )
            if self.partitions.has_non_adjacent_cuts:
                raise ValueError("C3 v1.5 requires a chain without non-adjacent cuts")
            cut_edges_by_link = self._cut_edges_by_link(self.partitions)
            generators = [
                CandidateGenerator(
                    agent_id=i,
                    arcs=self.dataset.arcs,
                    graph=self.graph,
                    partitions=self.partitions,
                    top_h=64,
                    seed=self.config.seed,
                )
                for i in range(self.config.num_agents)
            ]
            engines = [
                ProposalEngine(i, self.config.seed, hotspot_size=64)
                for i in range(self.config.num_agents)
            ]
            solver = ConditionedLocalSolver(
                self.dataset.arcs,
                self.graph,
                self.partitions,
                moves=10 if self.config.smoke else 500,
                restarts=1 if self.config.smoke else 2,
                seed=self.config.seed,
            )
            self._event(
                "run_start",
                method=self.config.method,
                seed=self.config.seed,
                smoke=self.config.smoke,
                oracle=True,
                oracle_requested=self.config.oracle,
                num_nodes=self.graph.num_nodes,
                num_edges=self.graph.num_edges,
                cut_edges=len(self.partitions.cut_edges),
                history_archive_enabled=False,
            )

            warmup_records = {}  # type: Dict[int, List[_RealizedRecord]]
            safe_ids = {}  # type: Dict[int, str]
            warmup_quartets = 0 if self.config.smoke else 4
            for agent_id, generator in enumerate(generators):
                signatures, _ = generator.warmup(warmup_quartets)
                safe_ids[agent_id] = generator.empty_signature().signature_hash
                warmup_records[agent_id] = [
                    self._solve_signature(
                        solver,
                        self.graph,
                        signature,
                        phase="warmup",
                        macro_round=-1,
                        wave="warmup",
                        slot=slot,
                    )
                    for slot, signature in enumerate(signatures)
                ]
            warmup_factor, warmup_order = self._factor_pass(
                warmup_records,
                incumbent_ids={},
                safe_ids=safe_ids,
                cut_edges_by_link=cut_edges_by_link,
                label="warmup_initialization",
                count_messages=False,
                require_incumbent=False,
            )
            incumbent_records = self._chosen_records(warmup_factor, warmup_order)
            incumbent_ids = {
                agent_id: record.candidate_id
                for agent_id, record in incumbent_records.items()
            }
            safe_records = {}
            for agent_id in range(self.config.num_agents):
                matches = [
                    record
                    for record in warmup_records[agent_id]
                    if record.candidate_id == safe_ids[agent_id]
                ]
                if len(matches) != 1 or matches[0].result.status != "feasible":
                    raise RuntimeError("missing unique feasible safe anchor")
                safe_records[agent_id] = matches[0]
            incumbent_selected = tuple(
                sorted(
                    node
                    for record in incumbent_records.values()
                    for node in record.result.selected_ids
                )
            )
            initial_check = verify_selected_ids(self.graph, incumbent_selected)
            if not initial_check.feasible:
                raise RuntimeError("initial incumbent invalid: {}".format(initial_check.message))
            self.incumbent_objective = self._objective(incumbent_selected)
            initial_objective = self.incumbent_objective

            round_metrics = []
            for macro_round in range(self.config.macro_rounds):
                # Current-domain only: no previous non-incumbent candidate survives.
                current = {
                    i: _deduplicate_records(
                        [incumbent_records[i], safe_records[i]]
                    )
                    for i in range(self.config.num_agents)
                }
                wave_a = self._query_wave(
                    "A",
                    macro_round,
                    generators,
                    engines,
                    incumbent_records,
                    solver,
                    self.graph,
                    pressure_maps=None,
                )
                for i in range(self.config.num_agents):
                    current[i] = _deduplicate_records(current[i] + wave_a[i])

                if self.behavior.wave_a_message_pass:
                    wave_a_factor, wave_a_order = self._factor_pass(
                        current,
                        incumbent_ids,
                        safe_ids,
                        cut_edges_by_link,
                        label="macro_%d_wave_a_pressure" % macro_round,
                    )
                    pressure = self._pressure_maps(
                        generators, wave_a_order, wave_a_factor
                    )
                    wave_a_value = wave_a_factor.best_value
                else:
                    wave_a_order = tuple(
                        tuple(_deduplicate_records(current[i]))
                        for i in range(self.config.num_agents)
                    )
                    pressure = self._pressure_maps(generators, wave_a_order, None)
                    wave_a_value = None

                wave_b = self._query_wave(
                    "B",
                    macro_round,
                    generators,
                    engines,
                    incumbent_records,
                    solver,
                    self.graph,
                    pressure_maps=pressure,
                )
                for i in range(self.config.num_agents):
                    current[i] = _deduplicate_records(current[i] + wave_b[i])

                if self.behavior.final_selection == "independent":
                    chosen = {}
                    for i in range(self.config.num_agents):
                        feasible = [
                            item for item in current[i] if item.result.status == "feasible"
                        ]
                        chosen[i] = max(
                            feasible,
                            key=lambda item: (
                                float(item.result.objective),
                                item.candidate_id,
                            ),
                        )
                    final_domain_value = sum(
                        float(item.result.objective) for item in chosen.values()
                    )
                else:
                    final_factor, final_order = self._factor_pass(
                        current,
                        incumbent_ids,
                        safe_ids,
                        cut_edges_by_link,
                        label="macro_%d_final_selection" % macro_round,
                    )
                    chosen = self._chosen_records(final_factor, final_order)
                    final_domain_value = final_factor.best_value

                proposed_ids, shielded_records, shield = self._shield(chosen, generators)
                proposed_objective = float(shield["post_shield_objective"])
                accepted = proposed_objective > self.incumbent_objective + 1.0e-9
                if accepted:
                    incumbent_records = shielded_records
                    incumbent_ids = {
                        i: incumbent_records[i].candidate_id
                        for i in range(self.config.num_agents)
                    }
                    incumbent_selected = proposed_ids
                    self.incumbent_objective = proposed_objective
                if self.objective_trace:
                    self.objective_trace[-1] = float(self.incumbent_objective)
                row = {
                    "macro_round": macro_round,
                    "wave_a_restricted_value": wave_a_value,
                    "final_current_domain_value": final_domain_value,
                    "proposed_objective": proposed_objective,
                    "accepted": accepted,
                    "incumbent_objective": self.incumbent_objective,
                    "current_domain_counts": [len(current[i]) for i in range(6)],
                    "shield": shield,
                }
                round_metrics.append(row)
                self._event("macro_commit", **row)

            if self.online_calls != self.config.expected_online_calls:
                raise RuntimeError(
                    "online-call invariant failed: expected {}, got {}".format(
                        self.config.expected_online_calls, self.online_calls
                    )
                )
            final_check = verify_selected_ids(self.graph, incumbent_selected)
            if not final_check.feasible:
                raise RuntimeError("final schedule invalid: {}".format(final_check.message))
            gains = [value - initial_objective for value in self.objective_trace]
            summary = {
                "schema_version": "SNSD-v1.5-oracle-run-v1",
                "status": "complete",
                "method": self.config.method,
                "seed": self.config.seed,
                "smoke": self.config.smoke,
                "oracle_mode": True,
                "policy_backend": "exact-pressure-no-learning",
                "data_path": str(data_path),
                "dataset_sha256": _sha256_file(data_path),
                "graph_hash": self.graph.graph_hash,
                "num_nodes": self.graph.num_nodes,
                "num_edges": self.graph.num_edges,
                "cut_edges": len(self.partitions.cut_edges),
                "num_agents": self.config.num_agents,
                "macro_rounds": self.config.macro_rounds,
                "waves_per_macro": 2,
                "query_budget_per_agent_wave": self.config.query_budget,
                "warmup_query_calls": self.warmup_calls,
                "online_query_calls": self.online_calls,
                "expected_online_query_calls": self.config.expected_online_calls,
                "initial_objective": initial_objective,
                "final_objective": self.incumbent_objective,
                "final_gain": self.incumbent_objective - initial_objective,
                "query_normalized_auc": (
                    float(sum(gains)) / float(len(gains)) if gains else 0.0
                ),
                "objective_trace": self.objective_trace,
                "final_selected_ids": list(incumbent_selected),
                "final_feasible": final_check.feasible,
                "rounds": round_metrics,
                "message_passes": self.message_passes,
                "directed_messages": self.directed_messages,
                "message_payload_scalars": self.message_payload_scalars,
                "message_payload_bytes_float64": 8 * self.message_payload_scalars,
                "physical_worker_runtime_s": self.physical_worker_runtime_s,
                "total_runtime_s": time.perf_counter() - started,
                "history_archive_enabled": False,
                "current_domain_only": True,
            }
            self._event("run_complete", summary=summary)
            assert self._event_log is not None
            self._event_log.close()
            _atomic_json(self.summary_path, summary)
            receipt = {
                "schema_version": "SNSD-v1.5-receipt-v1",
                "status": "complete",
                "method": self.config.method,
                "seed": self.config.seed,
                "artifacts": {
                    "events.jsonl": {
                        "sha256": _sha256_file(self.events_path),
                        "bytes": self.events_path.stat().st_size,
                    },
                    "summary.json": {
                        "sha256": _sha256_file(self.summary_path),
                        "bytes": self.summary_path.stat().st_size,
                    },
                },
            }
            _atomic_json(self.receipt_path, receipt)
            return ExperimentResult(
                output_dir=str(self.output_dir),
                events_path=str(self.events_path),
                summary_path=str(self.summary_path),
                receipt_path=str(self.receipt_path),
                summary=summary,
            )
        finally:
            if self._event_log is not None:
                self._event_log.close()


def run_experiment(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
) -> ExperimentResult:
    return BaseNegotiationProtocol(
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
    "MethodBehavior",
    "BaseNegotiationProtocol",
    "method_behavior",
    "normalize_method",
    "run_experiment",
]
