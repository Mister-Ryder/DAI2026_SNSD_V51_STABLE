"""Structured boundary-state lifting and coalitional refill for SNSD v5.0.

This module deliberately operates on *real* local records and graph nodes.
It does not pool a subgraph into one scalar message and it does not call the
paid local solver.  Each coordination cycle:

1. retains several realized local elites per agent in a BoundaryPacket;
2. exposes cut-conflict, local-substitution and refill-response structure;
3. updates cut-edge prices for several primal--dual microcycles;
4. repairs the incompatible private-elite union; and
5. creates a novel local response by refilling the subgraphs exposed by the
   boundary deletions.

The caller still performs the authoritative full-graph verification and exact
objective comparison against J/R0/Rm.  Consequently this module can add an
opportunity but cannot force a worse committed solution.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class BoundaryCandidate:
    agent_id: int
    node_id: int
    selected_frequency: float
    raw_value: float
    local_marginal: float
    delete_cost: float
    refill_gain: float
    uncertainty: float
    role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": int(self.agent_id),
            "node_id": int(self.node_id),
            "selected_frequency": float(self.selected_frequency),
            "raw_value": float(self.raw_value),
            "local_marginal": float(self.local_marginal),
            "delete_cost": float(self.delete_cost),
            "refill_gain": float(self.refill_gain),
            "uncertainty": float(self.uncertainty),
            "role": str(self.role),
        }


@dataclass(frozen=True)
class BoundaryPacket:
    agent_id: int
    base_anchor_hash: str
    elite_candidate_ids: Tuple[str, ...]
    selected_boundary_ids: Tuple[int, ...]
    candidate_ids: Tuple[int, ...]
    candidates: Tuple[BoundaryCandidate, ...]
    cut_conflicts: Tuple[Tuple[int, int], ...]
    relation_counts: Mapping[str, int]
    payload_scalars: int
    payload_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": int(self.agent_id),
            "base_anchor_hash": str(self.base_anchor_hash),
            "elite_candidate_ids": list(self.elite_candidate_ids),
            "selected_boundary_ids": list(self.selected_boundary_ids),
            "candidate_ids": list(self.candidate_ids),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "cut_conflicts": [list(edge) for edge in self.cut_conflicts],
            "relation_counts": dict(self.relation_counts),
            "payload_scalars": int(self.payload_scalars),
            "payload_bytes": int(self.payload_bytes),
        }


@dataclass(frozen=True)
class CoalitionProposal:
    members: Tuple[int, ...]
    removed_ids: Tuple[int, ...]
    added_ids: Tuple[int, ...]
    refill_ids: Tuple[int, ...]
    selected_ids_by_agent: Mapping[int, Tuple[int, ...]]
    feasible: bool
    objective: float
    primal_residual: float
    dual_residual: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "members": list(self.members),
            "removed_ids": list(self.removed_ids),
            "added_ids": list(self.added_ids),
            "refill_ids": list(self.refill_ids),
            "selected_ids_by_agent": {
                str(agent): list(ids)
                for agent, ids in self.selected_ids_by_agent.items()
            },
            "feasible": bool(self.feasible),
            "objective": float(self.objective),
            "primal_residual": float(self.primal_residual),
            "dual_residual": float(self.dual_residual),
        }


@dataclass(frozen=True)
class StructuredBoundaryCycleResult:
    candidate_selected_ids_by_agent: Mapping[int, Tuple[int, ...]]
    feasible: bool
    packets: Tuple[Mapping[str, Any], ...]
    micro_trace: Tuple[Mapping[str, Any], ...]
    payload_scalars: int
    payload_bytes: int
    primal_residual: float
    dual_residual: float
    proposal_count: int
    members: Tuple[int, ...]
    removed_ids: Tuple[int, ...]
    added_ids: Tuple[int, ...]
    refill_ids: Tuple[int, ...]
    relation_counts: Mapping[str, int]


def _record_id(record: Any) -> str:
    value = getattr(record, "candidate_id", None)
    if value is not None:
        return str(value)
    result = getattr(record, "result", record)
    signature = getattr(result, "signature_hash", "")
    selected = tuple(int(node) for node in getattr(result, "selected_ids", ()))
    digest = hashlib.sha256()
    digest.update(str(signature).encode("utf-8"))
    digest.update(repr(selected).encode("ascii"))
    return digest.hexdigest()


def _selected(record: Any) -> Tuple[int, ...]:
    result = getattr(record, "result", record)
    return tuple(int(node) for node in getattr(result, "selected_ids", ()))


def _objective_record(record: Any) -> float:
    result = getattr(record, "result", record)
    return float(getattr(result, "objective", 0.0))


def _feasible_record(record: Any) -> bool:
    result = getattr(record, "result", record)
    return str(getattr(result, "status", "feasible")) == "feasible"


def _neighbors(graph: Any, node: int) -> Tuple[int, ...]:
    begin = int(graph.indptr[int(node)])
    end = int(graph.indptr[int(node) + 1])
    return tuple(int(value) for value in graph.indices[begin:end])


def _node_weight(arcs: Sequence[Any], node: int) -> float:
    return float(getattr(arcs[int(node)], "weight"))


def _objective(arcs: Sequence[Any], selected: Iterable[int]) -> float:
    return float(sum(_node_weight(arcs, node) for node in selected))


def _conflict_count(graph: Any, selected: Set[int]) -> int:
    count = 0
    for node in selected:
        for other in _neighbors(graph, node):
            if other in selected and int(node) < int(other):
                count += 1
    return int(count)


def _globally_feasible(graph: Any, selected: Set[int]) -> bool:
    return _conflict_count(graph, selected) == 0


def _can_add(graph: Any, node: int, selected: Set[int]) -> bool:
    if int(node) in selected:
        return False
    return not any(int(other) in selected for other in _neighbors(graph, int(node)))


def _top_records(
    private_elite: Any,
    records: Sequence[Any],
    top_k: int,
) -> Tuple[Any, ...]:
    unique = {}  # type: Dict[str, Any]
    for record in (private_elite,) + tuple(records):
        if not _feasible_record(record):
            continue
        key = _record_id(record)
        previous = unique.get(key)
        if previous is None or _objective_record(record) > _objective_record(previous):
            unique[key] = record
    ordered = sorted(
        unique.values(),
        key=lambda record: (-_objective_record(record), _record_id(record)),
    )
    if private_elite not in ordered and _feasible_record(private_elite):
        ordered.insert(0, private_elite)
    return tuple(ordered[: max(1, int(top_k))])


def _normalise(values: Mapping[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    scale = max([abs(float(value)) for value in values.values()] + [1.0e-12])
    return {int(node): float(value) / float(scale) for node, value in values.items()}


def _channel_map(
    candidate_nodes: Mapping[int, Sequence[int]],
    channels: Optional[Mapping[int, Mapping[str, Mapping[int, float]]]],
    semantic_mode: str,
    num_agents: int,
) -> Dict[int, Dict[int, float]]:
    """Return one bounded semantic keep-credit per real packet node.

    Shuffling preserves the number of transmitted values: the sorted payload
    from the next agent is assigned by rank to the target packet nodes.  It
    therefore cannot collapse to all zeros merely because node IDs differ.
    """

    mode = str(semantic_mode)
    result = {agent: {} for agent in range(num_agents)}
    if channels is None or mode == "zero":
        return result

    def raw_for(source: int, node: int) -> float:
        item = channels.get(int(source), {})
        pressure = float(item.get("pressure", {}).get(int(node), 0.0))
        conflict = float(item.get("conflict", {}).get(int(node), 0.0))
        opportunity = float(item.get("opportunity", {}).get(int(node), 0.0))
        if mode == "conflict_only":
            return pressure + conflict
        if mode == "benefit_only":
            return pressure - opportunity
        return pressure + conflict - 0.5 * abs(opportunity)

    if mode != "shuffled":
        for agent, nodes in candidate_nodes.items():
            raw = {int(node): raw_for(int(agent), int(node)) for node in nodes}
            result[int(agent)] = _normalise(raw)
        return result

    for agent, nodes in candidate_nodes.items():
        target = tuple(sorted(int(node) for node in nodes))
        source = (int(agent) + 1) % int(num_agents)
        source_item = channels.get(source, {})
        source_nodes = sorted(
            set(int(node) for channel in source_item.values() for node in channel)
        )
        source_values = [raw_for(source, node) for node in source_nodes]
        if not source_values:
            source_values = [0.0]
        assigned = {
            node: float(source_values[index % len(source_values)])
            for index, node in enumerate(target)
        }
        result[int(agent)] = _normalise(assigned)
    return result


def _candidate_pool(
    agent_id: int,
    removed: Sequence[int],
    records: Sequence[Any],
    selected: Set[int],
    graph: Any,
    partitions: Any,
    limit: int = 192,
) -> Set[int]:
    pool = set()  # type: Set[int]
    owner = partitions.partition_id
    for raw in removed:
        node = int(raw)
        if int(owner[node]) != int(agent_id):
            continue
        first_hop = _neighbors(graph, node)
        for candidate in first_hop:
            if int(owner[candidate]) == int(agent_id):
                pool.add(int(candidate))
                for second in _neighbors(graph, candidate):
                    if int(owner[second]) == int(agent_id):
                        pool.add(int(second))
    for record in records:
        for node in _selected(record):
            if int(owner[node]) == int(agent_id) and int(node) not in selected:
                pool.add(int(node))
    pool.difference_update(int(node) for node in removed)
    if len(pool) > int(limit):
        degree = graph.degree
        ordered = sorted(
            pool,
            key=lambda node: (
                -1.0 / (1.0 + float(degree[node])),
                int(node),
            ),
        )
        pool = set(ordered[: int(limit)])
    return pool


def _estimate_refill(
    agent_id: int,
    node: int,
    records: Sequence[Any],
    private_selected: Set[int],
    graph: Any,
    partitions: Any,
    arcs: Sequence[Any],
) -> float:
    pool = _candidate_pool(
        agent_id,
        [node],
        records,
        private_selected.difference({int(node)}),
        graph,
        partitions,
        limit=48,
    )
    available = []
    remaining = private_selected.difference({int(node)})
    for candidate in pool:
        if _node_weight(arcs, candidate) <= 0.0:
            continue
        if any(other in remaining for other in _neighbors(graph, candidate)):
            continue
        available.append(candidate)
    available.sort(
        key=lambda candidate: (
            -_node_weight(arcs, candidate)
            / (1.0 + float(len(_neighbors(graph, candidate)))),
            -_node_weight(arcs, candidate),
            candidate,
        )
    )
    chosen = set()  # type: Set[int]
    value = 0.0
    for candidate in available:
        if any(other in chosen for other in _neighbors(graph, candidate)):
            continue
        chosen.add(candidate)
        value += _node_weight(arcs, candidate)
        if len(chosen) >= 4:
            break
    return float(value)


def _packet_candidates(
    agent_id: int,
    elites: Sequence[Any],
    private_elite: Any,
    boundary: Set[int],
    graph: Any,
    partitions: Any,
    arcs: Sequence[Any],
    capacity: int,
) -> Tuple[BoundaryCandidate, ...]:
    private_ids = set(_selected(private_elite))
    counts = {}  # type: Dict[int, int]
    objectives_on = {}  # type: Dict[int, List[float]]
    objectives_off = {}  # type: Dict[int, List[float]]
    candidate_nodes = set()  # type: Set[int]
    for record in elites:
        ids = set(_selected(record))
        candidate_nodes.update(node for node in ids if node in boundary)
        for node in boundary:
            target = objectives_on if node in ids else objectives_off
            target.setdefault(int(node), []).append(_objective_record(record))
            if node in ids:
                counts[int(node)] = counts.get(int(node), 0) + 1
    # Boundary alternatives exposed by a private selected boundary vertex are
    # also real subgraph nodes, even when none of the retained elites selects
    # them.  They are useful add/refill responses.
    for node in tuple(candidate_nodes):
        for other in _neighbors(graph, node):
            if int(partitions.partition_id[other]) == int(agent_id):
                candidate_nodes.add(int(other))

    # The graph boundary can be large on C3.  Preserve a wider structural
    # shortlist before computing the more expensive two-hop refill response;
    # the final packet is still capped below after exact fields are attached.
    candidate_nodes = set(
        sorted(
            candidate_nodes,
            key=lambda node: (
                -int(node in private_ids),
                -counts.get(int(node), 0),
                -_node_weight(arcs, int(node)),
                int(node),
            ),
        )[: max(12, int(capacity) * 3)]
    )

    rows = []  # type: List[BoundaryCandidate]
    for node in candidate_nodes:
        selected_frequency = float(counts.get(node, 0)) / float(max(1, len(elites)))
        on = objectives_on.get(node, [])
        off = objectives_off.get(node, [])
        local_marginal = (
            float(sum(on) / len(on) - sum(off) / len(off))
            if on and off
            else _node_weight(arcs, node)
        )
        refill_gain = _estimate_refill(
            agent_id,
            node,
            elites,
            private_ids,
            graph,
            partitions,
            arcs,
        )
        raw = _node_weight(arcs, node)
        rows.append(
            BoundaryCandidate(
                agent_id=int(agent_id),
                node_id=int(node),
                selected_frequency=selected_frequency,
                raw_value=raw,
                local_marginal=local_marginal,
                delete_cost=max(0.0, raw - refill_gain),
                refill_gain=refill_gain,
                uncertainty=selected_frequency * (1.0 - selected_frequency),
                role="keep" if node in private_ids else "add_or_refill",
            )
        )
    rows.sort(
        key=lambda item: (
            -int(item.node_id in private_ids),
            -item.selected_frequency,
            -abs(item.local_marginal),
            -item.raw_value,
            item.node_id,
        )
    )
    return tuple(rows[: max(4, int(capacity))])


def _build_packets(
    graph: Any,
    partitions: Any,
    arcs: Sequence[Any],
    generators: Sequence[Any],
    private_elites: Mapping[int, Any],
    current_records: Sequence[Sequence[Any]],
    top_k: int,
) -> Tuple[Tuple[BoundaryPacket, ...], Dict[int, Tuple[Any, ...]]]:
    packets = []  # type: List[BoundaryPacket]
    elites_by_agent = {}  # type: Dict[int, Tuple[Any, ...]]
    cut_edges = tuple((int(edge[0]), int(edge[1])) for edge in partitions.cut_edges)
    for agent_id in range(len(generators)):
        elites = _top_records(
            private_elites[agent_id], current_records[agent_id], int(top_k)
        )
        elites_by_agent[agent_id] = elites
        boundary = set(int(node) for node in generators[agent_id].boundary)
        candidates = _packet_candidates(
            agent_id,
            elites,
            private_elites[agent_id],
            boundary,
            graph,
            partitions,
            arcs,
            capacity=int(top_k) * 4,
        )
        candidate_ids = tuple(candidate.node_id for candidate in candidates)
        candidate_set = set(candidate_ids)
        packet_conflicts = tuple(
            edge for edge in cut_edges if edge[0] in candidate_set or edge[1] in candidate_set
        )
        substitutions = 0
        complements = 0
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if graph.has_edge(left.node_id, right.node_id):
                    substitutions += 1
                elif left.role != right.role:
                    complements += 1
        relations = {
            "conflict": len(packet_conflicts),
            "substitution": int(substitutions),
            "complement": int(complements),
            "coalition": len(
                {
                    int(partitions.partition_id[v if int(partitions.partition_id[u]) == agent_id else u])
                    for u, v in packet_conflicts
                    if int(partitions.partition_id[u]) == agent_id
                    or int(partitions.partition_id[v]) == agent_id
                }
            ),
        }
        # Eight numeric fields per candidate, two endpoints plus a type per
        # relation, and a small fixed header.  IDs/hashes are charged as eight
        # byte slots to keep the estimate deterministic across Python builds.
        payload_scalars = (
            8 * len(candidates)
            + 3 * sum(relations.values())
            + len(elites)
            + len(boundary.intersection(set(_selected(private_elites[agent_id]))))
            + 6
        )
        packets.append(
            BoundaryPacket(
                agent_id=agent_id,
                base_anchor_hash=_record_id(private_elites[agent_id]),
                elite_candidate_ids=tuple(_record_id(record) for record in elites),
                selected_boundary_ids=tuple(
                    sorted(boundary.intersection(set(_selected(private_elites[agent_id]))))
                ),
                candidate_ids=tuple(sorted(candidate_ids)),
                candidates=candidates,
                cut_conflicts=packet_conflicts,
                relation_counts=relations,
                payload_scalars=int(payload_scalars),
                payload_bytes=int(payload_scalars) * 8,
            )
        )
    return tuple(packets), elites_by_agent


def _refill_gain_map(packets: Sequence[BoundaryPacket]) -> Dict[int, float]:
    result = {}  # type: Dict[int, float]
    for packet in packets:
        for candidate in packet.candidates:
            result[candidate.node_id] = max(
                result.get(candidate.node_id, 0.0), float(candidate.refill_gain)
            )
    return result


def _semantic_candidate_nodes(
    packets: Sequence[BoundaryPacket],
) -> Dict[int, Tuple[int, ...]]:
    return {
        int(packet.agent_id): tuple(candidate.node_id for candidate in packet.candidates)
        for packet in packets
    }


def _coalition_components(
    conflicts: Sequence[Tuple[int, int]], partitions: Any, max_size: int = 3
) -> Tuple[Tuple[int, ...], ...]:
    adjacency = {}  # type: Dict[int, Set[int]]
    for u, v in conflicts:
        left = int(partitions.partition_id[u])
        right = int(partitions.partition_id[v])
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components = []  # type: List[Tuple[int, ...]]
    visited = set()  # type: Set[int]
    for start in sorted(adjacency):
        if start in visited:
            continue
        queue = [start]
        members = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            members.append(current)
            queue.extend(sorted(adjacency.get(current, ())))
        if len(members) <= max_size:
            components.append(tuple(sorted(members)))
        else:
            for begin in range(0, len(members) - 1, max_size - 1):
                part = tuple(sorted(members[begin : begin + max_size]))
                if len(part) >= 2:
                    components.append(part)
    return tuple(components)


def _repair_and_refill(
    graph: Any,
    partitions: Any,
    arcs: Sequence[Any],
    private_elites: Mapping[int, Any],
    elites_by_agent: Mapping[int, Sequence[Any]],
    packets: Sequence[BoundaryPacket],
    semantic_credit: Mapping[int, Mapping[int, float]],
    prices: Mapping[Tuple[int, int], float],
    max_deletions: int,
) -> Tuple[CoalitionProposal, int]:
    private_union = set()  # type: Set[int]
    for record in private_elites.values():
        private_union.update(_selected(record))
    selected = set(private_union)
    active_cut = [
        (int(edge[0]), int(edge[1]))
        for edge in partitions.cut_edges
        if int(edge[0]) in selected and int(edge[1]) in selected
    ]
    primal_before = len(active_cut)
    refill_estimate = _refill_gain_map(packets)
    node_price = {}  # type: Dict[int, float]
    for edge, edge_price in prices.items():
        node_price[int(edge[0])] = node_price.get(int(edge[0]), 0.0) + float(edge_price)
        node_price[int(edge[1])] = node_price.get(int(edge[1]), 0.0) + float(edge_price)
    removed = []  # type: List[int]
    removed_by_agent = {}  # type: Dict[int, List[int]]

    # Resolve every cut conflict.  max_deletions is enforced per local
    # coalition component; a complete graph can contain several independent
    # 2/3-agent coalitions in the same microcycle.
    component_budget = {}  # type: Dict[Tuple[int, ...], int]
    components = _coalition_components(active_cut, partitions)
    edge_component = {}  # type: Dict[Tuple[int, int], Tuple[int, ...]]
    for edge in active_cut:
        owners = {int(partitions.partition_id[edge[0]]), int(partitions.partition_id[edge[1]])}
        component = next(
            (item for item in components if owners.issubset(set(item))),
            tuple(sorted(owners)),
        )
        edge_component[edge] = component
        component_budget.setdefault(component, max(1, int(max_deletions)))

    for u, v in sorted(active_cut):
        if u not in selected or v not in selected:
            continue
        au = int(partitions.partition_id[u])
        av = int(partitions.partition_id[v])
        edge = (min(u, v), max(u, v))
        credit_u = float(semantic_credit.get(au, {}).get(u, 0.0))
        credit_v = float(semantic_credit.get(av, {}).get(v, 0.0))
        scale = max(1.0, 0.5 * (_node_weight(arcs, u) + _node_weight(arcs, v)))
        keep_u = (
            _node_weight(arcs, u)
            - min(_node_weight(arcs, u), refill_estimate.get(u, 0.0))
            + 0.35 * scale * credit_u
            - float(node_price.get(u, 0.0))
        )
        keep_v = (
            _node_weight(arcs, v)
            - min(_node_weight(arcs, v), refill_estimate.get(v, 0.0))
            + 0.35 * scale * credit_v
            - float(node_price.get(v, 0.0))
        )
        loser = v if (keep_u, _node_weight(arcs, u), -u) >= (
            keep_v,
            _node_weight(arcs, v),
            -v,
        ) else u
        component = edge_component[(u, v)]
        # If the preferred component already used its advertised deletion
        # width, still resolve the conflict, but choose the cheaper raw endpoint
        # and count it as a second coalition proposal rather than returning an
        # infeasible partial repair.
        if component_budget[component] <= 0:
            loser = min(
                (u, v),
                key=lambda node: (_node_weight(arcs, node), int(node)),
            )
            component_budget[component] = max(1, int(max_deletions))
        component_budget[component] -= 1
        selected.remove(int(loser))
        removed.append(int(loser))
        owner = int(partitions.partition_id[loser])
        removed_by_agent.setdefault(owner, []).append(int(loser))

    after_delete = set(selected)
    refill_added = []  # type: List[int]
    degree = graph.degree
    for agent_id in sorted(removed_by_agent):
        pool = _candidate_pool(
            agent_id,
            removed_by_agent[agent_id],
            elites_by_agent.get(agent_id, ()),
            selected,
            graph,
            partitions,
        )
        credit = semantic_credit.get(agent_id, {})
        ordered = sorted(
            pool,
            key=lambda node: (
                -(
                    _node_weight(arcs, node) / (1.0 + math.log1p(float(degree[node])))
                    + 0.20 * _node_weight(arcs, node) * float(credit.get(node, 0.0))
                ),
                -_node_weight(arcs, node),
                int(node),
            ),
        )
        for node in ordered:
            if _node_weight(arcs, node) <= 0.0:
                continue
            if _can_add(graph, node, selected):
                selected.add(int(node))
                refill_added.append(int(node))

        # One deterministic one-for-many local exchange gives the response
        # book a genuinely new local configuration while preserving the
        # coalition boundary and exact global feasibility.
        for node in ordered[:64]:
            if node in selected or _node_weight(arcs, node) <= 0.0:
                continue
            conflicts = [
                other for other in _neighbors(graph, node) if other in selected
            ]
            if not conflicts:
                selected.add(int(node))
                refill_added.append(int(node))
                continue
            if any(int(partitions.partition_id[other]) != int(agent_id) for other in conflicts):
                continue
            removed_value = sum(_node_weight(arcs, other) for other in conflicts)
            if _node_weight(arcs, node) <= removed_value + 1.0e-9:
                continue
            for other in conflicts:
                selected.remove(int(other))
                if other in refill_added:
                    refill_added.remove(other)
                if other not in removed:
                    removed.append(int(other))
            if _can_add(graph, node, selected):
                selected.add(int(node))
                refill_added.append(int(node))
            else:
                for other in conflicts:
                    if _can_add(graph, other, selected):
                        selected.add(int(other))

    feasible = _globally_feasible(graph, selected)
    primal_after = float(_conflict_count(graph, selected))
    selected_by_agent = {
        agent_id: tuple(
            sorted(
                node
                for node in selected
                if int(partitions.partition_id[node]) == int(agent_id)
            )
        )
        for agent_id in range(int(partitions.k))
    }
    affected = set(int(partitions.partition_id[node]) for node in removed)
    for node in refill_added:
        affected.add(int(partitions.partition_id[node]))
    added = tuple(sorted(selected.difference(private_union)))
    proposal = CoalitionProposal(
        members=tuple(sorted(affected))[:3],
        removed_ids=tuple(sorted(set(removed))),
        added_ids=added,
        refill_ids=tuple(sorted(set(refill_added))),
        selected_ids_by_agent=selected_by_agent,
        feasible=bool(feasible),
        objective=_objective(arcs, selected) if feasible else float("-inf"),
        primal_residual=primal_after,
        dual_residual=0.0,
    )
    return proposal, max(1, len(components)) if active_cut else 0


def run_structured_boundary_cycle(
    graph: Any,
    partitions: Any,
    arcs: Sequence[Any],
    generators: Sequence[Any],
    private_elites: Mapping[int, Any],
    current_records: Sequence[Sequence[Any]],
    base_records: Mapping[int, Any],
    message_channels: Optional[Mapping[int, Mapping[str, Mapping[int, float]]]],
    semantic_mode: str,
    top_k: int,
    micro_rounds: int,
    max_deletions: int,
    seed: int,
    macro_round: int,
    step: int,
) -> StructuredBoundaryCycleResult:
    """Execute one structured late-stage negotiation cycle.

    ``base_records`` is intentionally retained in the API and packet audit but
    is not used as the refill starting point: current-domain Max-Sum already
    optimizes combinations of those records.  The novel response starts from
    the incompatible private-elite union and becomes valuable only by creating
    new delete/refill local records outside that existing factor domain.
    """

    del seed  # All tie breaks are canonical; the launch seed controls records.
    if int(top_k) <= 0 or int(micro_rounds) <= 0:
        raise ValueError("top_k and micro_rounds must be positive")
    if len(generators) != int(partitions.k):
        raise ValueError("generator count must equal partition count")
    if len(current_records) != int(partitions.k):
        raise ValueError("current record tables must equal partition count")
    if any(agent not in base_records for agent in range(int(partitions.k))):
        raise ValueError("base_records must contain every agent")

    packets, elites_by_agent = _build_packets(
        graph,
        partitions,
        arcs,
        generators,
        private_elites,
        current_records,
        int(top_k),
    )
    candidate_nodes = _semantic_candidate_nodes(packets)
    semantic_credit = _channel_map(
        candidate_nodes,
        message_channels,
        str(semantic_mode),
        int(partitions.k),
    )
    prices = {}  # type: Dict[Tuple[int, int], float]
    private_selected = set()  # type: Set[int]
    for record in private_elites.values():
        private_selected.update(_selected(record))
    active_cut = [
        (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
        for edge in partitions.cut_edges
        if int(edge[0]) in private_selected and int(edge[1]) in private_selected
    ]
    best = None  # type: Optional[CoalitionProposal]
    traces = []  # type: List[Mapping[str, Any]]
    total_proposals = 0
    latest_dual = 0.0
    for micro in range(int(micro_rounds)):
        previous = dict(prices)
        step_size = 0.20 / math.sqrt(float(micro + 1))
        for edge in active_cut:
            u, v = edge
            scale = max(1.0, 0.5 * (_node_weight(arcs, u) + _node_weight(arcs, v)))
            prices[edge] = max(0.0, prices.get(edge, 0.0) + step_size * scale)
        latest_dual = math.sqrt(
            sum(
                (prices.get(edge, 0.0) - previous.get(edge, 0.0)) ** 2
                for edge in set(prices).union(previous)
            )
        )
        proposal, proposal_count = _repair_and_refill(
            graph,
            partitions,
            arcs,
            private_elites,
            elites_by_agent,
            packets,
            semantic_credit,
            prices,
            int(max_deletions),
        )
        proposal = CoalitionProposal(
            members=proposal.members,
            removed_ids=proposal.removed_ids,
            added_ids=proposal.added_ids,
            refill_ids=proposal.refill_ids,
            selected_ids_by_agent=proposal.selected_ids_by_agent,
            feasible=proposal.feasible,
            objective=proposal.objective,
            primal_residual=proposal.primal_residual,
            dual_residual=float(latest_dual),
        )
        total_proposals += int(proposal_count)
        if proposal.feasible and (
            best is None
            or (proposal.objective, tuple(-node for node in proposal.removed_ids))
            > (best.objective, tuple(-node for node in best.removed_ids))
        ):
            best = proposal
        traces.append(
            {
                "macro_round": int(macro_round),
                "source_step": int(step),
                "microcycle": int(micro),
                "semantic_mode": str(semantic_mode),
                "active_cut_conflicts": len(active_cut),
                "coalition_proposals": int(proposal_count),
                "members": list(proposal.members),
                "removed_ids": list(proposal.removed_ids),
                "refill_ids": list(proposal.refill_ids),
                "candidate_objective": (
                    float(proposal.objective) if proposal.feasible else None
                ),
                "primal_residual": float(proposal.primal_residual),
                "dual_residual": float(latest_dual),
                "novel_response": bool(proposal.added_ids or proposal.refill_ids),
            }
        )

    if best is None:
        fallback = {
            agent: tuple(sorted(_selected(base_records[agent])))
            for agent in range(int(partitions.k))
        }
        fallback_union = set(node for ids in fallback.values() for node in ids)
        best = CoalitionProposal(
            members=(),
            removed_ids=(),
            added_ids=(),
            refill_ids=(),
            selected_ids_by_agent=fallback,
            feasible=_globally_feasible(graph, fallback_union),
            objective=_objective(arcs, fallback_union),
            primal_residual=float(_conflict_count(graph, fallback_union)),
            dual_residual=float(latest_dual),
        )

    relation_counts = {"conflict": 0, "complement": 0, "substitution": 0, "coalition": 0}
    for packet in packets:
        for relation, count in packet.relation_counts.items():
            relation_counts[str(relation)] = relation_counts.get(str(relation), 0) + int(count)
    payload_scalars = sum(packet.payload_scalars for packet in packets) * int(micro_rounds)
    payload_bytes = sum(packet.payload_bytes for packet in packets) * int(micro_rounds)
    return StructuredBoundaryCycleResult(
        candidate_selected_ids_by_agent=best.selected_ids_by_agent,
        feasible=bool(best.feasible),
        packets=tuple(packet.to_dict() for packet in packets),
        micro_trace=tuple(traces),
        payload_scalars=int(payload_scalars),
        payload_bytes=int(payload_bytes),
        primal_residual=float(best.primal_residual),
        dual_residual=float(best.dual_residual),
        proposal_count=int(total_proposals),
        members=best.members,
        removed_ids=best.removed_ids,
        added_ids=best.added_ids,
        refill_ids=best.refill_ids,
        relation_counts=relation_counts,
    )


__all__ = [
    "BoundaryCandidate",
    "BoundaryPacket",
    "CoalitionProposal",
    "StructuredBoundaryCycleResult",
    "run_structured_boundary_cycle",
]
