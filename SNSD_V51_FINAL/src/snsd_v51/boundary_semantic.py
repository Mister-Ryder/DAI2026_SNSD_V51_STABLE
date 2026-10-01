"""Safeguarded semantic boundary portfolio for SNSD v5.1.

This module leaves the published v5.0 mechanism untouched.  It exposes the
same cycle inputs, but retains a portfolio of exact-feasible counterfactual
responses.  The base_protocol layer can therefore evaluate a zero-semantic
branch and a treated-semantic branch from *the same* state and take their
exact union.

Microcycle zero is the v5.0 response.  Later microcycles add deterministic
no-good feedback: an endpoint deleted by an earlier response receives a keep
bonus and an earlier refill receives a refill-ranking penalty.  This makes the
microcycles counterfactual messages_base instead of repetitions of one response;
the exact selector remains responsible for accepting only improvements.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .boundary_joint import (
    BoundaryPacket,
    CoalitionProposal,
    _build_packets,
    _channel_map,
    _globally_feasible,
    _objective,
    _repair_and_refill,
    _selected,
    _semantic_candidate_nodes,
)


@dataclass(frozen=True)
class SemanticBoundaryCycleResult:
    """v5-compatible best response plus the v5.1 response portfolio."""

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
    semantic_mode: str
    best_microcycle: int
    distinct_response_count: int
    response_signatures: Tuple[str, ...]
    portfolio: Tuple[Mapping[str, Any], ...]
    counterfactual_feedback_scalars: int
    semantic_credit_nonzero: int
    semantic_credit_linf: float


def _response_signature(proposal: CoalitionProposal) -> str:
    selected = tuple(
        (int(agent), tuple(int(node) for node in proposal.selected_ids_by_agent[agent]))
        for agent in sorted(proposal.selected_ids_by_agent)
    )
    payload = (
        tuple(int(node) for node in proposal.removed_ids),
        tuple(int(node) for node in proposal.refill_ids),
        selected,
    )
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


def _counterfactual_credit(
    base: Mapping[int, Mapping[int, float]],
    partitions: Any,
    endpoint_keep_counts: Mapping[int, int],
    refill_nogood_counts: Mapping[int, int],
    endpoint_keep_bonus: float,
    refill_nogood_penalty: float,
) -> Dict[int, Dict[int, float]]:
    result = {
        int(agent): {int(node): float(value) for node, value in values.items()}
        for agent, values in base.items()
    }
    for agent in range(int(partitions.k)):
        result.setdefault(agent, {})
    for raw_node, raw_count in endpoint_keep_counts.items():
        node = int(raw_node)
        owner = int(partitions.partition_id[node])
        result[owner][node] = float(result[owner].get(node, 0.0)) + (
            float(endpoint_keep_bonus) * float(raw_count)
        )
    for raw_node, raw_count in refill_nogood_counts.items():
        node = int(raw_node)
        owner = int(partitions.partition_id[node])
        result[owner][node] = float(result[owner].get(node, 0.0)) - (
            float(refill_nogood_penalty) * float(raw_count)
        )
    return result


def _portfolio_row(
    proposal: CoalitionProposal,
    microcycle: int,
    signature: str,
) -> Mapping[str, Any]:
    row = dict(proposal.to_dict())
    row.update(
        {
            "microcycle": int(microcycle),
            "response_signature": str(signature),
        }
    )
    return row


def run_semantic_boundary_cycle(
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
) -> SemanticBoundaryCycleResult:
    """Build one zero/treated branch from an immutable shared input state.

    Callers obtain the v5-compatible branch best through
    ``candidate_selected_ids_by_agent``.  ``portfolio`` contains every
    distinct feasible microresponse, enabling a larger exact candidate union
    without another paid worker query.
    """

    del seed  # Canonical tie breaking; the run seed already controls records.
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
    base_credit = _channel_map(
        candidate_nodes,
        message_channels,
        str(semantic_mode),
        int(partitions.k),
    )
    credit_values = [
        abs(float(value))
        for values in base_credit.values()
        for value in values.values()
    ]
    semantic_credit_nonzero = sum(value > 1.0e-12 for value in credit_values)
    semantic_credit_linf = max(credit_values + [0.0])

    private_selected = set()
    for record in private_elites.values():
        private_selected.update(_selected(record))
    active_cut = [
        (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
        for edge in partitions.cut_edges
        if int(edge[0]) in private_selected and int(edge[1]) in private_selected
    ]

    # A strong, explicit no-good coefficient is safe because each generated
    # response is subsequently evaluated by exact feasibility and objective.
    # It is a diversification control, not an objective approximation.
    endpoint_keep_bonus = 5.0
    refill_nogood_penalty = 3.0
    endpoint_keep_counts = {}  # type: Dict[int, int]
    refill_nogood_counts = {}  # type: Dict[int, int]
    prices = {}  # type: Dict[Tuple[int, int], float]
    traces = []  # type: List[Mapping[str, Any]]
    portfolio = []  # type: List[Mapping[str, Any]]
    portfolio_proposals = []  # type: List[Tuple[int, str, CoalitionProposal]]
    seen = set()
    total_proposals = 0
    latest_dual = 0.0
    feedback_scalars = 0

    for micro in range(int(micro_rounds)):
        previous = dict(prices)
        step_size = 0.20 / math.sqrt(float(micro + 1))
        for edge in active_cut:
            u, v = edge
            scale = max(
                1.0,
                0.5
                * (
                    float(getattr(arcs[int(u)], "weight"))
                    + float(getattr(arcs[int(v)], "weight"))
                ),
            )
            prices[edge] = max(
                0.0, float(prices.get(edge, 0.0)) + step_size * scale
            )
        latest_dual = math.sqrt(
            sum(
                (prices.get(edge, 0.0) - previous.get(edge, 0.0)) ** 2
                for edge in set(prices).union(previous)
            )
        )

        working_credit = _counterfactual_credit(
            base_credit,
            partitions,
            endpoint_keep_counts,
            refill_nogood_counts,
            endpoint_keep_bonus,
            refill_nogood_penalty,
        )
        proposal, proposal_count = _repair_and_refill(
            graph,
            partitions,
            arcs,
            private_elites,
            elites_by_agent,
            packets,
            working_credit,
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
        signature = _response_signature(proposal)
        is_distinct = signature not in seen
        if proposal.feasible and is_distinct:
            seen.add(signature)
            portfolio.append(_portfolio_row(proposal, micro, signature))
            portfolio_proposals.append((int(micro), str(signature), proposal))

        active_endpoints = set(node for edge in active_cut for node in edge)
        boundary_removed = tuple(
            node for node in proposal.removed_ids if int(node) in active_endpoints
        )
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
                "removed_boundary_ids": list(boundary_removed),
                "refill_ids": list(proposal.refill_ids),
                "candidate_objective": (
                    float(proposal.objective) if proposal.feasible else None
                ),
                "primal_residual": float(proposal.primal_residual),
                "dual_residual": float(latest_dual),
                "novel_response": bool(proposal.added_ids or proposal.refill_ids),
                "counterfactual_distinct": bool(is_distinct),
                "response_signature": str(signature),
                "endpoint_keep_nogoods": len(endpoint_keep_counts),
                "refill_nogoods": len(refill_nogood_counts),
            }
        )

        # Feed the realized response back only to subsequent microcycles.
        # IDs and their counts/penalties are charged as two numeric slots each.
        if micro + 1 < int(micro_rounds):
            for node in boundary_removed:
                endpoint_keep_counts[int(node)] = (
                    endpoint_keep_counts.get(int(node), 0) + 1
                )
            for node in proposal.refill_ids:
                refill_nogood_counts[int(node)] = (
                    refill_nogood_counts.get(int(node), 0) + 1
                )
            feedback_scalars += 2 * (
                len(boundary_removed) + len(proposal.refill_ids)
            )

    if portfolio_proposals:
        best_microcycle, best_signature, best = max(
            portfolio_proposals,
            key=lambda item: (
                float(item[2].objective),
                -int(item[0]),
                str(item[1]),
            ),
        )
    else:
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
            primal_residual=0.0,
            dual_residual=float(latest_dual),
        )
        best_microcycle = -1
        best_signature = _response_signature(best)
        portfolio.append(_portfolio_row(best, best_microcycle, best_signature))
        seen.add(best_signature)

    relation_counts = {
        "conflict": 0,
        "complement": 0,
        "substitution": 0,
        "coalition": 0,
    }
    for packet in packets:
        for relation, count in packet.relation_counts.items():
            relation_counts[str(relation)] = (
                relation_counts.get(str(relation), 0) + int(count)
            )
    packet_scalars = sum(packet.payload_scalars for packet in packets) * int(
        micro_rounds
    )
    payload_scalars = int(packet_scalars + feedback_scalars)
    return SemanticBoundaryCycleResult(
        candidate_selected_ids_by_agent=best.selected_ids_by_agent,
        feasible=bool(best.feasible),
        packets=tuple(packet.to_dict() for packet in packets),
        micro_trace=tuple(traces),
        payload_scalars=payload_scalars,
        payload_bytes=int(payload_scalars) * 8,
        primal_residual=float(best.primal_residual),
        dual_residual=float(best.dual_residual),
        proposal_count=int(total_proposals),
        members=best.members,
        removed_ids=best.removed_ids,
        added_ids=best.added_ids,
        refill_ids=best.refill_ids,
        relation_counts=relation_counts,
        semantic_mode=str(semantic_mode),
        best_microcycle=int(best_microcycle),
        distinct_response_count=len(seen),
        response_signatures=tuple(row["response_signature"] for row in portfolio),
        portfolio=tuple(portfolio),
        counterfactual_feedback_scalars=int(feedback_scalars),
        semantic_credit_nonzero=int(semantic_credit_nonzero),
        semantic_credit_linf=float(semantic_credit_linf),
    )


__all__ = [
    "SemanticBoundaryCycleResult",
    "run_semantic_boundary_cycle",
]
