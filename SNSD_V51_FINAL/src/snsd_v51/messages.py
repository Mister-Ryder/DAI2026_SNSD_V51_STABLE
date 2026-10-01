"""Fair, auditable proposal generation for the v1.5 mechanism experiments.

This module is intentionally isolated from :mod:`messages_base`.  It keeps the
same public data graph_layers where possible, but makes all pseudo-random tie breaks
depend only on the pre-registered base_protocol slot.  Message/state hashes are
provenance only: they never choose a candidate or a worker seed.

Materialisation is also deliberately non-repairing.  The v1 generator may
project an internally incompatible request, but the projection is recorded and
is ineligible for query by default.  Likewise, a signature already observed by
the caller is diagnosed as a duplicate without mutating the caller's set.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import messages_base as _v1


H_DYNAMIC_DEFAULT = _v1.H_DYNAMIC_DEFAULT
WAVE_A_DEFAULT_COUNT = _v1.WAVE_A_DEFAULT_COUNT
WAVE_B_DEFAULT_COUNT = _v1.WAVE_B_DEFAULT_COUNT
stable_hash = _v1.stable_hash
derived_seed = _v1.derived_seed
BoundaryMessage = _v1.BoundaryMessage
MessageIntervention = _v1.MessageIntervention
DynamicHotspot = _v1.DynamicHotspot
ComplianceRecord = _v1.ComplianceRecord


def _ordered_subset(order: Sequence[int], values: Iterable[int]) -> Tuple[int, ...]:
    wanted = set(int(value) for value in values)
    return tuple(int(node_id) for node_id in order if int(node_id) in wanted)


class BoundaryState(_v1.BoundaryState):
    """The v1 state with one-shot iterable-safe construction."""

    @classmethod
    def from_maps(
        cls,
        agent_id: int,
        boundary_order: Sequence[int],
        incumbent_selected: Iterable[int],
        raw_benefit: Optional[Any] = None,
        local_marginal: Optional[Any] = None,
        conflict_score: Optional[Any] = None,
        prior_opportunity_cost: Optional[Any] = None,
        uncertainty: Optional[Any] = None,
        prior_pressure: Optional[Any] = None,
    ) -> "BoundaryState":
        # The v1 implementation iterates ``incumbent_selected`` twice.  Capture
        # it once so generators and other one-shot iterables remain valid.
        selected = tuple(int(value) for value in incumbent_selected)
        base = _v1.BoundaryState.from_maps(
            agent_id=agent_id,
            boundary_order=boundary_order,
            incumbent_selected=selected,
            raw_benefit=raw_benefit,
            local_marginal=local_marginal,
            conflict_score=conflict_score,
            prior_opportunity_cost=prior_opportunity_cost,
            uncertainty=uncertainty,
            prior_pressure=prior_pressure,
        )
        return cls(**dict(base.__dict__))


def _normalise_intervention(mode: Any) -> MessageIntervention:
    if isinstance(mode, MessageIntervention):
        return mode
    token = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "real": MessageIntervention.REAL,
        "none": MessageIntervention.REAL,
        "zero": MessageIntervention.ZERO,
        "zero_message": MessageIntervention.ZERO,
        "zeromessage": MessageIntervention.ZERO,
        "shuffle": MessageIntervention.SHUFFLED,
        "shuffled": MessageIntervention.SHUFFLED,
        "shuffled_message": MessageIntervention.SHUFFLED,
    }
    if token not in aliases:
        raise ValueError("unknown message intervention: %s" % mode)
    return aliases[token]


def apply_message_intervention(
    message: BoundaryMessage,
    mode: Any,
    experiment_seed: int,
    agent_id: int,
    macro_round: int,
) -> BoundaryMessage:
    """Apply an intervention with a common, message-independent shuffle RNG."""

    intervention = _normalise_intervention(mode)
    order = tuple(message.boundary_order)
    identity = tuple(range(len(order)))
    if intervention is MessageIntervention.REAL:
        channels = (
            message.opportunity_cost,
            message.pressure,
            message.neighbor_recovery,
            message.conflict_score,
            message.uncertainty,
        )
        permutation = identity
        original_hash = message.original_message_hash
    elif intervention is MessageIntervention.ZERO:
        zeros = tuple(0.0 for _ in order)
        channels = (zeros, zeros, zeros, zeros, zeros)
        permutation = identity
        original_hash = "masked"
    else:
        n = len(order)
        permutation_list = list(identity)
        rng = random.Random(
            derived_seed(
                "message-shuffle-rng-v2",
                int(experiment_seed),
                int(agent_id),
                int(macro_round),
                order,
            )
        )
        rng.shuffle(permutation_list)
        permutation = tuple(permutation_list)
        if n > 1 and permutation == identity:
            permutation = tuple(list(identity[1:]) + [identity[0]])

        def permute(values: Sequence[float]) -> Tuple[float, ...]:
            return tuple(float(values[source]) for source in permutation)

        channels = (
            permute(message.opportunity_cost),
            permute(message.pressure),
            permute(message.neighbor_recovery),
            permute(message.conflict_score),
            permute(message.uncertainty),
        )
        original_hash = message.original_message_hash
    digest = stable_hash(
        "intervened-message-v2",
        intervention.value,
        order,
        channels,
        permutation,
    )
    return BoundaryMessage(
        boundary_order=order,
        opportunity_cost=tuple(channels[0]),
        pressure=tuple(channels[1]),
        neighbor_recovery=tuple(channels[2]),
        conflict_score=tuple(channels[3]),
        uncertainty=tuple(channels[4]),
        intervention=intervention.value,
        original_message_hash=original_hash,
        message_hash=digest,
        permutation=permutation,
    )


def _rank_percentiles(
    nodes: Sequence[int],
    value_by_node: Mapping[int, float],
    tie_context: Tuple[Any, ...],
) -> Tuple[Tuple[int, ...], Dict[int, float]]:
    ranked = tuple(
        sorted(
            (int(node_id) for node_id in nodes),
            key=lambda node_id: (
                -float(value_by_node.get(node_id, 0.0)),
                derived_seed("rank-tie-rng-v2", tie_context, node_id),
                node_id,
            ),
        )
    )
    if len(ranked) <= 1:
        return ranked, {node_id: 1.0 for node_id in ranked}
    scale = float(len(ranked) - 1)
    return ranked, {
        node_id: 1.0 - float(rank) / scale
        for rank, node_id in enumerate(ranked)
    }


def _phase_context(
    experiment_seed: int,
    agent_id: int,
    macro_round: int,
    phase: str,
) -> Tuple[Any, ...]:
    """The complete and only random context used by proposal ranking."""

    return (
        "proposal-rng-v2",
        int(experiment_seed),
        int(agent_id),
        int(macro_round),
        str(phase),
    )


def select_dynamic_hotspot(
    state: BoundaryState,
    experiment_seed: int,
    macro_round: int,
    hotspot_size: int = H_DYNAMIC_DEFAULT,
    phase: str = "hotspot",
) -> DynamicHotspot:
    """Select a signal-balanced hotspot using a common-slot tie stream."""

    requested_size = max(1, int(hotspot_size))
    nodes = tuple(state.boundary_order)
    target = min(len(nodes), max(requested_size, len(state.incumbent_selected)))
    raw = state.value_map("raw_benefit")
    local = state.value_map("local_marginal")
    conflicts = state.value_map("conflict_score")
    opportunity = state.value_map("prior_opportunity_cost")
    uncertainty = state.value_map("uncertainty")
    pressure = state.value_map("prior_pressure")
    context = _phase_context(
        experiment_seed, state.agent_id, macro_round, "%s/rank" % phase
    )

    _, raw_pct = _rank_percentiles(nodes, raw, context + ("raw",))
    _, local_pct = _rank_percentiles(nodes, local, context + ("local",))
    _, conflict_pct = _rank_percentiles(nodes, conflicts, context + ("conflict",))
    _, opportunity_pct = _rank_percentiles(
        nodes, opportunity, context + ("opportunity",)
    )
    uncertainty_ranked, uncertainty_pct = _rank_percentiles(
        nodes, uncertainty, context + ("uncertainty",)
    )
    abs_pressure = {node: abs(pressure[node]) for node in nodes}
    pressure_ranked, pressure_pct = _rank_percentiles(
        nodes, abs_pressure, context + ("pressure",)
    )
    benefit = {
        node: 0.55 * raw_pct[node] + 0.45 * local_pct[node] for node in nodes
    }
    opportunity_signal = {
        node: 0.60 * opportunity_pct[node] + 0.40 * conflict_pct[node]
        for node in nodes
    }
    benefit_ranked, benefit_pct = _rank_percentiles(
        nodes, benefit, context + ("benefit",)
    )
    opportunity_ranked, opportunity_combined_pct = _rank_percentiles(
        nodes, opportunity_signal, context + ("opportunity-combined",)
    )
    composite = {
        node: (
            0.30 * benefit_pct[node]
            + 0.25 * opportunity_combined_pct[node]
            + 0.20 * uncertainty_pct[node]
            + 0.25 * pressure_pct[node]
        )
        for node in nodes
    }
    composite_ranked, _ = _rank_percentiles(
        nodes, composite, context + ("composite",)
    )
    selected_ranked = tuple(
        sorted(
            state.incumbent_selected,
            key=lambda node: (
                -composite[node],
                derived_seed("selected-hotspot-tie-rng-v2", context, node),
                node,
            ),
        )
    )

    reasons_by_node: Dict[int, Set[str]] = {}
    chosen: List[int] = []
    chosen_set: Set[int] = set()

    def add(node_id: int, reason: str) -> None:
        node_id = int(node_id)
        reasons_by_node.setdefault(node_id, set()).add(str(reason))
        if node_id not in chosen_set and len(chosen) < target:
            chosen.append(node_id)
            chosen_set.add(node_id)

    for node in selected_ranked:
        add(node, "incumbent_selected")
    sources = (
        ("benefit", benefit_ranked),
        ("opportunity", opportunity_ranked),
        ("uncertainty", uncertainty_ranked),
        ("pressure", pressure_ranked),
    )
    cursors = [0 for _ in sources]
    while len(chosen) < target:
        progressed = False
        for index, (name, ranking) in enumerate(sources):
            while cursors[index] < len(ranking):
                node = ranking[cursors[index]]
                cursors[index] += 1
                if node not in chosen_set:
                    add(node, name)
                    progressed = True
                    break
                reasons_by_node.setdefault(node, set()).add(name)
            if len(chosen) >= target:
                break
        if not progressed:
            break
    for node in composite_ranked:
        if len(chosen) >= target:
            break
        add(node, "composite")
    reasons = tuple(
        (node, tuple(sorted(reasons_by_node.get(node, {"composite"}))))
        for node in chosen
    )
    digest = stable_hash(
        "dynamic-hotspot-v2",
        state.agent_id,
        int(macro_round),
        str(phase),
        requested_size,
        chosen,
        reasons,
        state.state_hash,
        int(experiment_seed),
    )
    return DynamicHotspot(
        agent_id=state.agent_id,
        macro_round=int(macro_round),
        requested_size=requested_size,
        effective_size=len(chosen),
        node_ids=tuple(chosen),
        benefit_ranked=benefit_ranked,
        opportunity_ranked=opportunity_ranked,
        uncertainty_ranked=uncertainty_ranked,
        pressure_ranked=pressure_ranked,
        composite_ranked=composite_ranked,
        reasons=reasons,
        state_hash=state.state_hash,
        hotspot_hash=digest,
    )


@dataclass(frozen=True)
class ProposalIntent:
    agent_id: int
    macro_round: int
    proposal_wave: str
    intent_type: str
    slot: int
    boundary_order: Tuple[int, ...]
    anchor_selected_boundary: Tuple[int, ...]
    hotspot_ids: Tuple[int, ...]
    requested_fix_on: Tuple[int, ...]
    requested_fix_off: Tuple[int, ...]
    requested_selected_boundary: Tuple[int, ...]
    added_ids: Tuple[int, ...]
    removed_ids: Tuple[int, ...]
    source_message_hash: str
    message_intervention: str
    top_pressure_ids: Tuple[int, ...]
    top_pressure_values: Tuple[float, ...]
    boundary_order_hash: str
    state_hash: str
    hotspot_hash: str
    random_seed: int
    intent_hash: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "macro_round": self.macro_round,
            "proposal_wave": self.proposal_wave,
            "intent_type": self.intent_type,
            "slot": self.slot,
            "boundary_order": list(self.boundary_order),
            "anchor_selected_boundary": list(self.anchor_selected_boundary),
            "hotspot_ids": list(self.hotspot_ids),
            "requested_fix_on": list(self.requested_fix_on),
            "requested_fix_off": list(self.requested_fix_off),
            "requested_selected_boundary": list(self.requested_selected_boundary),
            "added_ids": list(self.added_ids),
            "removed_ids": list(self.removed_ids),
            "source_message_hash": self.source_message_hash,
            "message_intervention": self.message_intervention,
            "top_pressure_ids": list(self.top_pressure_ids),
            "top_pressure_values": list(self.top_pressure_values),
            "boundary_order_hash": self.boundary_order_hash,
            "state_hash": self.state_hash,
            "hotspot_hash": self.hotspot_hash,
            "random_seed": self.random_seed,
            "intent_hash": self.intent_hash,
        }


@dataclass(frozen=True)
class MaterializedProposal:
    intent: ProposalIntent
    signature: Any
    projection_compliance: ComplianceRecord
    projection_changed: bool
    duplicate_signature: bool
    eligible_for_query: bool
    rejection_reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        signature_dict = (
            self.signature.to_dict()
            if hasattr(self.signature, "to_dict")
            else {"signature_hash": getattr(self.signature, "signature_hash", "")}
        )
        return {
            "intent": self.intent.to_dict(),
            "signature": signature_dict,
            "projection_compliance": self.projection_compliance.to_dict(),
            "projection_changed": self.projection_changed,
            "duplicate_signature": self.duplicate_signature,
            "eligible_for_query": self.eligible_for_query,
            "rejection_reasons": list(self.rejection_reasons),
        }


def measure_compliance(
    intent: ProposalIntent,
    realized_selected_ids: Iterable[int],
    signature_hash: str = "",
    solver_status: str = "",
) -> ComplianceRecord:
    """Compare a realization against the intent's exact boundary ordering."""

    boundary = tuple(intent.boundary_order)
    boundary_set = set(boundary)
    requested_on = set(intent.requested_fix_on)
    requested_off = set(intent.requested_fix_off)
    if requested_on.union(requested_off) != boundary_set:
        raise ValueError("intent fixes do not cover its boundary")
    if requested_on.intersection(requested_off):
        raise ValueError("intent fixes overlap")
    realized_set = set(int(value) for value in realized_selected_ids).intersection(
        boundary_set
    )
    anchor = set(intent.anchor_selected_boundary)
    violated_on = requested_on.difference(realized_set)
    violated_off = requested_off.intersection(realized_set)
    fix_on_compliance = (
        float(len(requested_on.intersection(realized_set))) / float(len(requested_on))
        if requested_on else 1.0
    )
    fix_off_compliance = (
        float(len(requested_off.difference(realized_set))) / float(len(requested_off))
        if requested_off else 1.0
    )
    return ComplianceRecord(
        intent_hash=intent.intent_hash,
        signature_hash=str(signature_hash),
        solver_status=str(solver_status),
        requested_fix_on=intent.requested_fix_on,
        requested_fix_off=intent.requested_fix_off,
        requested_selected_boundary=intent.requested_selected_boundary,
        realized_selected_boundary=_ordered_subset(boundary, realized_set),
        realized_fix_on=_ordered_subset(boundary, realized_set),
        realized_fix_off=_ordered_subset(boundary, boundary_set.difference(realized_set)),
        violated_fix_on=_ordered_subset(boundary, violated_on),
        violated_fix_off=_ordered_subset(boundary, violated_off),
        requested_added_ids=intent.added_ids,
        requested_removed_ids=intent.removed_ids,
        realized_added_ids=_ordered_subset(boundary, realized_set.difference(anchor)),
        realized_removed_ids=_ordered_subset(boundary, anchor.difference(realized_set)),
        fix_on_compliance=fix_on_compliance,
        fix_off_compliance=fix_off_compliance,
        full_compliance=not violated_on and not violated_off,
    )


def materialize_intent(
    intent: ProposalIntent,
    candidate_generator: Any,
    anchor_signature: Optional[Any] = None,
    strict_compliance: bool = True,
    seen_signature_hashes: Iterable[str] = (),
) -> MaterializedProposal:
    """Project once, diagnose faithfully, and never mutate caller-owned state."""

    if int(getattr(candidate_generator, "agent_id")) != intent.agent_id:
        raise ValueError("candidate generator belongs to a different agent")
    generator_boundary = tuple(int(value) for value in candidate_generator.boundary)
    if generator_boundary != tuple(intent.boundary_order):
        raise ValueError("candidate generator boundary order differs from intent")
    if anchor_signature is not None and set(anchor_signature.selected_ids) != set(
        intent.anchor_selected_boundary
    ):
        raise ValueError("anchor signature differs from intent incumbent")
    # No retry, mutation, fallback, or repair is performed here.  Projection is
    # an observed property of this exact requested signature.
    signature = candidate_generator.make_signature(
        intent.requested_selected_boundary,
        anchor=anchor_signature,
        mutable_override=intent.hotspot_ids,
    )
    compliance = measure_compliance(
        intent,
        signature.selected_ids,
        signature_hash=signature.signature_hash,
        solver_status="projected",
    )
    projection_changed = not compliance.full_compliance
    seen = frozenset(str(value) for value in seen_signature_hashes)
    duplicate = str(signature.signature_hash) in seen
    reasons: List[str] = []
    if bool(strict_compliance) and projection_changed:
        reasons.append("projection_noncompliance")
    if duplicate:
        reasons.append("duplicate_signature")
    return MaterializedProposal(
        intent=intent,
        signature=signature,
        projection_compliance=compliance,
        projection_changed=projection_changed,
        duplicate_signature=duplicate,
        eligible_for_query=not reasons,
        rejection_reasons=tuple(reasons),
    )


def solver_compliance(
    materialized: MaterializedProposal,
    solve_result: Any,
) -> ComplianceRecord:
    return measure_compliance(
        materialized.intent,
        getattr(solve_result, "selected_ids"),
        signature_hash=str(getattr(solve_result, "signature_hash", "")),
        solver_status=str(getattr(solve_result, "status", "")),
    )


def _allocate_quotas(count: int, weights: Sequence[int]) -> Tuple[int, ...]:
    count = max(0, int(count))
    if count == 0:
        return tuple(0 for _ in weights)
    total = float(sum(int(weight) for weight in weights))
    raw = [float(count) * float(weight) / total for weight in weights]
    result = [int(math.floor(value)) for value in raw]
    remaining = count - sum(result)
    order = sorted(
        range(len(weights)), key=lambda index: (-(raw[index] - result[index]), index)
    )
    for index in order[:remaining]:
        result[index] += 1
    return tuple(result)


class ProposalEngine:
    """Two-wave generator with common random numbers across interventions."""

    def __init__(
        self,
        agent_id: int,
        experiment_seed: int,
        hotspot_size: int = H_DYNAMIC_DEFAULT,
    ) -> None:
        self.agent_id = int(agent_id)
        self.experiment_seed = int(experiment_seed)
        self.hotspot_size = max(1, int(hotspot_size))

    def _check_state(self, state: BoundaryState) -> None:
        if int(state.agent_id) != self.agent_id:
            raise ValueError("boundary state belongs to a different agent")

    def dynamic_hotspot(
        self, state: BoundaryState, macro_round: int, phase: str = "A/hotspot"
    ) -> DynamicHotspot:
        self._check_state(state)
        return select_dynamic_hotspot(
            state,
            experiment_seed=self.experiment_seed,
            macro_round=macro_round,
            hotspot_size=self.hotspot_size,
            phase=phase,
        )

    def _make_intent(
        self,
        state: BoundaryState,
        hotspot: DynamicHotspot,
        macro_round: int,
        proposal_wave: str,
        intent_type: str,
        slot: int,
        requested_selected: Iterable[int],
        source_message_hash: str,
        message_intervention: str,
        top_pressure_ids: Sequence[int],
        top_pressure_values: Sequence[float],
    ) -> ProposalIntent:
        order = tuple(state.boundary_order)
        boundary_set = set(order)
        hotspot_set = set(hotspot.node_ids)
        anchor = set(state.incumbent_selected)
        requested = set(int(value) for value in requested_selected).intersection(
            boundary_set
        )
        for node in boundary_set.difference(hotspot_set):
            if (node in requested) != (node in anchor):
                raise ValueError("a non-hotspot bit differs from the incumbent")
        fix_on = _ordered_subset(order, requested)
        fix_off = _ordered_subset(order, boundary_set.difference(requested))
        added = _ordered_subset(order, requested.difference(anchor))
        removed = _ordered_subset(order, anchor.difference(requested))
        random_seed = derived_seed(
            "proposal-intent-slot-rng-v2",
            self.experiment_seed,
            self.agent_id,
            int(macro_round),
            str(proposal_wave),
            int(slot),
        )
        payload = {
            "agent_id": self.agent_id,
            "macro_round": int(macro_round),
            "proposal_wave": str(proposal_wave),
            "intent_type": str(intent_type),
            "slot": int(slot),
            "boundary_order": order,
            "anchor": state.incumbent_selected,
            "hotspot": hotspot.node_ids,
            "fix_on": fix_on,
            "fix_off": fix_off,
            "source_message_hash": str(source_message_hash),
            "message_intervention": str(message_intervention),
            "top_pressure_ids": tuple(int(value) for value in top_pressure_ids),
            "top_pressure_values": tuple(float(value) for value in top_pressure_values),
            "boundary_order_hash": state.boundary_order_hash,
            "state_hash": state.state_hash,
            "hotspot_hash": hotspot.hotspot_hash,
            "random_seed": random_seed,
        }
        intent_hash = stable_hash("proposal-intent-v2", payload)
        return ProposalIntent(
            agent_id=self.agent_id,
            macro_round=int(macro_round),
            proposal_wave=str(proposal_wave),
            intent_type=str(intent_type),
            slot=int(slot),
            boundary_order=order,
            anchor_selected_boundary=state.incumbent_selected,
            hotspot_ids=hotspot.node_ids,
            requested_fix_on=fix_on,
            requested_fix_off=fix_off,
            requested_selected_boundary=fix_on,
            added_ids=added,
            removed_ids=removed,
            source_message_hash=str(source_message_hash),
            message_intervention=str(message_intervention),
            top_pressure_ids=tuple(int(value) for value in top_pressure_ids),
            top_pressure_values=tuple(float(value) for value in top_pressure_values),
            boundary_order_hash=state.boundary_order_hash,
            state_hash=state.state_hash,
            hotspot_hash=hotspot.hotspot_hash,
            random_seed=random_seed,
            intent_hash=intent_hash,
        )

    def _append_patterns(
        self,
        result: List[ProposalIntent],
        seen: Set[Tuple[int, ...]],
        target_for_type: int,
        state: BoundaryState,
        hotspot: DynamicHotspot,
        macro_round: int,
        wave: str,
        intent_type: str,
        patterns: Iterable[Tuple[Iterable[int], Iterable[int]]],
        source_message_hash: str,
        intervention: str,
        top_pressure_ids: Sequence[int],
        top_pressure_values: Sequence[float],
    ) -> None:
        anchor = set(state.incumbent_selected)
        start_count = len(result)
        for add_ids, remove_ids in patterns:
            requested = anchor.union(int(value) for value in add_ids)
            requested.difference_update(int(value) for value in remove_ids)
            ordered = _ordered_subset(state.boundary_order, requested)
            if ordered == state.incumbent_selected or ordered in seen:
                continue
            result.append(
                self._make_intent(
                    state, hotspot, macro_round, wave, intent_type, len(result),
                    requested, source_message_hash, intervention,
                    top_pressure_ids, top_pressure_values,
                )
            )
            seen.add(ordered)
            if len(result) - start_count >= int(target_for_type):
                break

    def _fallback_fill(
        self,
        result: List[ProposalIntent],
        seen: Set[Tuple[int, ...]],
        count: int,
        state: BoundaryState,
        hotspot: DynamicHotspot,
        macro_round: int,
        wave: str,
        intent_type: str,
        source_message_hash: str,
        intervention: str,
        top_pressure_ids: Sequence[int],
        top_pressure_values: Sequence[float],
    ) -> None:
        anchor = set(state.incumbent_selected)
        mutable = list(hotspot.node_ids)
        # One common stream for the same wave slot, irrespective of state or
        # intervention provenance.
        rng = random.Random(
            derived_seed(
                "proposal-fallback-slot-rng-v2",
                self.experiment_seed,
                self.agent_id,
                int(macro_round),
                str(wave),
            )
        )
        attempts = 0
        max_attempts = max(512, int(count) * 128)
        while len(result) < int(count) and mutable and attempts < max_attempts:
            attempts += 1
            width = 1 + (attempts - 1) % min(4, len(mutable))
            requested = set(anchor)
            for node in rng.sample(mutable, width):
                if node in requested:
                    requested.remove(node)
                else:
                    requested.add(node)
            ordered = _ordered_subset(state.boundary_order, requested)
            if ordered == state.incumbent_selected or ordered in seen:
                continue
            result.append(
                self._make_intent(
                    state, hotspot, macro_round, wave, intent_type, len(result),
                    requested, source_message_hash, intervention,
                    top_pressure_ids, top_pressure_values,
                )
            )
            seen.add(ordered)

    def _state_with_message(
        self, state: BoundaryState, message: BoundaryMessage
    ) -> BoundaryState:
        if tuple(message.boundary_order) != tuple(state.boundary_order):
            raise ValueError("message boundary order differs from boundary state")
        return BoundaryState.from_maps(
            agent_id=state.agent_id,
            boundary_order=state.boundary_order,
            incumbent_selected=state.incumbent_selected,
            raw_benefit=state.raw_benefit,
            local_marginal=state.local_marginal,
            conflict_score=tuple(
                float(state.conflict_score[index])
                + abs(float(message.conflict_score[index]))
                for index in range(len(state.boundary_order))
            ),
            prior_opportunity_cost=tuple(
                float(state.prior_opportunity_cost[index])
                + abs(float(message.opportunity_cost[index]))
                + max(0.0, float(message.neighbor_recovery[index]))
                for index in range(len(state.boundary_order))
            ),
            uncertainty=tuple(
                max(float(state.uncertainty[index]), abs(float(message.uncertainty[index])))
                for index in range(len(state.boundary_order))
            ),
            prior_pressure=tuple(
                float(state.prior_pressure[index]) + float(message.pressure[index])
                for index in range(len(state.boundary_order))
            ),
        )

    def wave_a(
        self,
        state: BoundaryState,
        macro_round: int,
        count: int = WAVE_A_DEFAULT_COUNT,
    ) -> List[ProposalIntent]:
        self._check_state(state)
        count = max(0, int(count))
        if count == 0:
            return []
        hotspot = self.dynamic_hotspot(state, macro_round, phase="A/hotspot")
        nodes = list(hotspot.node_ids)
        selected = set(state.incumbent_selected)
        context = _phase_context(
            self.experiment_seed, self.agent_id, macro_round, "A/rank"
        )
        raw = state.value_map("raw_benefit")
        local = state.value_map("local_marginal")
        conflict = state.value_map("conflict_score")
        opportunity = state.value_map("prior_opportunity_cost")
        uncertainty = state.value_map("uncertainty")
        pressure = state.value_map("prior_pressure")
        _, raw_pct = _rank_percentiles(nodes, raw, context + ("raw",))
        _, local_pct = _rank_percentiles(nodes, local, context + ("local",))
        _, conflict_pct = _rank_percentiles(nodes, conflict, context + ("conflict",))
        _, opp_pct = _rank_percentiles(nodes, opportunity, context + ("opportunity",))
        _, uncertainty_pct = _rank_percentiles(
            nodes, uncertainty, context + ("uncertainty",)
        )
        benefit = {
            node: 0.55 * raw_pct[node] + 0.45 * local_pct[node]
            - 0.15 * conflict_pct[node] - 0.10 * opp_pct[node]
            for node in nodes
        }
        release = {
            node: 0.45 * conflict_pct[node] + 0.45 * opp_pct[node]
            - 0.10 * raw_pct[node]
            for node in nodes
        }
        explore = {
            node: 0.75 * uncertainty_pct[node]
            + 0.25 / (1.0 + abs(float(pressure[node])))
            for node in nodes
        }
        add_ranked, _ = _rank_percentiles(
            [node for node in nodes if node not in selected],
            benefit, context + ("add",),
        )
        release_ranked, _ = _rank_percentiles(
            [node for node in nodes if node in selected],
            release, context + ("release",),
        )
        explore_ranked, _ = _rank_percentiles(nodes, explore, context + ("explore",))
        pressure_ranked, _ = _rank_percentiles(
            nodes, {node: abs(pressure[node]) for node in nodes},
            context + ("pressure",),
        )
        top_ids = tuple(pressure_ranked[: min(8, len(pressure_ranked))])
        top_values = tuple(float(pressure[node]) for node in top_ids)
        source_hash = stable_hash("wave-a-no-peer-message-v2", state.state_hash)
        quotas = _allocate_quotas(count, (8, 8, 4, 4))
        result: List[ProposalIntent] = []
        seen: Set[Tuple[int, ...]] = set()

        benefit_patterns = itertools.chain(
            ((combo, ()) for combo in itertools.combinations(add_ranked[:16], 1)),
            ((combo, ()) for combo in itertools.combinations(add_ranked[:16], 2)),
            (((a,), (r,)) for a in add_ranked[:12] for r in reversed(release_ranked[:8])),
        )
        self._append_patterns(result, seen, quotas[0], state, hotspot, macro_round,
                              "A", "benefit", benefit_patterns, source_hash,
                              "none", top_ids, top_values)
        release_patterns = itertools.chain(
            (((), combo) for combo in itertools.combinations(release_ranked[:16], 1)),
            (((), combo) for combo in itertools.combinations(release_ranked[:16], 2)),
            (((), combo) for combo in itertools.combinations(release_ranked[:12], 3)),
        )
        self._append_patterns(result, seen, quotas[1], state, hotspot, macro_round,
                              "A", "release", release_patterns, source_hash,
                              "none", top_ids, top_values)
        swap_patterns = (
            ((a,), (r,)) for r in release_ranked[:12] for a in add_ranked[:12]
        )
        self._append_patterns(result, seen, quotas[2], state, hotspot, macro_round,
                              "A", "swap", swap_patterns, source_hash,
                              "none", top_ids, top_values)
        explore_patterns = itertools.chain(
            (((node,), ()) if node not in selected else ((), (node,))
             for node in explore_ranked[:24]),
            ((tuple(node for node in pair if node not in selected),
              tuple(node for node in pair if node in selected))
             for pair in itertools.combinations(explore_ranked[:16], 2)),
        )
        self._append_patterns(result, seen, quotas[3], state, hotspot, macro_round,
                              "A", "probe", explore_patterns, source_hash,
                              "none", top_ids, top_values)
        self._fallback_fill(result, seen, count, state, hotspot, macro_round,
                            "A", "probe", source_hash, "none", top_ids, top_values)
        return result[:count]

    def wave_b(
        self,
        state: BoundaryState,
        real_message: BoundaryMessage,
        macro_round: int,
        intervention: Any = MessageIntervention.REAL,
        count: int = WAVE_B_DEFAULT_COUNT,
    ) -> List[ProposalIntent]:
        self._check_state(state)
        count = max(0, int(count))
        if count == 0:
            return []
        message = apply_message_intervention(
            real_message, intervention, self.experiment_seed,
            self.agent_id, macro_round,
        )
        if tuple(message.boundary_order) != tuple(state.boundary_order):
            raise ValueError("message boundary order differs from boundary state")
        conditioned = self._state_with_message(state, message)
        hotspot = select_dynamic_hotspot(
            conditioned, self.experiment_seed, macro_round,
            self.hotspot_size, phase="B/hotspot",
        )
        nodes = list(hotspot.node_ids)
        selected = set(state.incumbent_selected)
        context = _phase_context(
            self.experiment_seed, self.agent_id, macro_round, "B/rank"
        )
        raw = state.value_map("raw_benefit")
        local = state.value_map("local_marginal")
        base_conflict = state.value_map("conflict_score")
        msg_opp = message.value_map("opportunity_cost")
        msg_pressure = message.value_map("pressure")
        msg_recovery = message.value_map("neighbor_recovery")
        msg_conflict = message.value_map("conflict_score")
        msg_uncertainty = message.value_map("uncertainty")
        state_uncertainty = state.value_map("uncertainty")
        _, raw_pct = _rank_percentiles(nodes, raw, context + ("raw",))
        _, local_pct = _rank_percentiles(nodes, local, context + ("local",))
        _, opp_pct = _rank_percentiles(
            nodes, {node: abs(msg_opp[node]) for node in nodes},
            context + ("opportunity",),
        )
        _, recovery_pct = _rank_percentiles(
            nodes, {node: max(0.0, msg_recovery[node]) for node in nodes},
            context + ("recovery",),
        )
        _, conflict_pct = _rank_percentiles(
            nodes,
            {node: abs(msg_conflict[node]) + max(0.0, base_conflict[node])
             for node in nodes},
            context + ("conflict",),
        )
        _, uncertainty_pct = _rank_percentiles(
            nodes,
            {node: max(abs(msg_uncertainty[node]), state_uncertainty[node])
             for node in nodes},
            context + ("uncertainty",),
        )
        _, pressure_pct = _rank_percentiles(
            nodes, {node: abs(msg_pressure[node]) for node in nodes},
            context + ("pressure",),
        )
        benefit = {
            node: 0.55 * raw_pct[node] + 0.45 * local_pct[node] for node in nodes
        }
        yield_value = {
            node: 0.35 * opp_pct[node] + 0.30 * recovery_pct[node]
            + 0.20 * conflict_pct[node] + 0.10 * float(msg_pressure[node] < 0.0)
            - 0.15 * benefit[node]
            for node in nodes
        }
        add_value = {
            node: 0.45 * benefit[node]
            + 0.35 * float(msg_pressure[node] > 0.0) * pressure_pct[node]
            + 0.20 * recovery_pct[node] - 0.20 * opp_pct[node]
            - 0.15 * conflict_pct[node]
            for node in nodes
        }
        explore_value = {
            node: 0.65 * uncertainty_pct[node]
            + 0.35 / (1.0 + abs(float(msg_pressure[node])))
            for node in nodes
        }
        yield_ranked, _ = _rank_percentiles(
            [node for node in nodes if node in selected],
            yield_value, context + ("yield",),
        )
        add_ranked, _ = _rank_percentiles(
            [node for node in nodes if node not in selected],
            add_value, context + ("add",),
        )
        explore_ranked, _ = _rank_percentiles(
            nodes, explore_value, context + ("explore",),
        )
        pressure_strength = {
            node: abs(msg_pressure[node]) + abs(msg_opp[node])
            + max(0.0, msg_recovery[node]) for node in nodes
        }
        top_ranked, _ = _rank_percentiles(
            nodes, pressure_strength, context + ("top-pressure",),
        )
        top_ids = tuple(top_ranked[: min(8, len(top_ranked))])
        top_values = tuple(float(msg_pressure[node]) for node in top_ids)
        quotas = _allocate_quotas(count, (1, 1, 1, 1))
        result: List[ProposalIntent] = []
        seen: Set[Tuple[int, ...]] = set()

        yield_patterns = itertools.chain(
            (((), combo) for combo in itertools.combinations(yield_ranked[:16], 1)),
            (((), combo) for combo in itertools.combinations(yield_ranked[:16], 2)),
        )
        self._append_patterns(result, seen, quotas[0], state, hotspot, macro_round,
                              "B", "yield", yield_patterns, message.message_hash,
                              message.intervention, top_ids, top_values)
        retreat_order = tuple(yield_ranked[1:]) + tuple(yield_ranked[:1])
        retreat_patterns = itertools.chain(
            (((), combo) for combo in itertools.combinations(retreat_order[:16], 2)),
            (((), combo) for combo in itertools.combinations(retreat_order[:12], 3)),
            (((), combo) for combo in itertools.combinations(retreat_order[:16], 1)),
        )
        self._append_patterns(result, seen, quotas[1], state, hotspot, macro_round,
                              "B", "retreat", retreat_patterns, message.message_hash,
                              message.intervention, top_ids, top_values)
        replace_patterns = itertools.chain(
            (((a,), (r,)) for r in yield_ranked[:16] for a in add_ranked[:16]),
            ((tuple(pair), (r,)) for r in yield_ranked[:8]
             for pair in itertools.combinations(add_ranked[:12], 2)),
        )
        self._append_patterns(result, seen, quotas[2], state, hotspot, macro_round,
                              "B", "replace", replace_patterns,
                              message.message_hash, message.intervention,
                              top_ids, top_values)
        explore_patterns = itertools.chain(
            (((node,), ()) if node not in selected else ((), (node,))
             for node in explore_ranked[:24]),
            ((tuple(node for node in pair if node not in selected),
              tuple(node for node in pair if node in selected))
             for pair in itertools.combinations(explore_ranked[:16], 2)),
        )
        self._append_patterns(result, seen, quotas[3], state, hotspot, macro_round,
                              "B", "explore", explore_patterns,
                              message.message_hash, message.intervention,
                              top_ids, top_values)
        self._fallback_fill(result, seen, count, state, hotspot, macro_round,
                            "B", "explore", message.message_hash,
                            message.intervention, top_ids, top_values)
        return result[:count]


def generate_wave_a_intents(
    state: BoundaryState,
    experiment_seed: int,
    macro_round: int,
    count: int = WAVE_A_DEFAULT_COUNT,
    hotspot_size: int = H_DYNAMIC_DEFAULT,
) -> List[ProposalIntent]:
    return ProposalEngine(state.agent_id, experiment_seed, hotspot_size).wave_a(
        state, macro_round, count
    )


def generate_wave_b_intents(
    state: BoundaryState,
    real_message: BoundaryMessage,
    experiment_seed: int,
    macro_round: int,
    intervention: Any = MessageIntervention.REAL,
    count: int = WAVE_B_DEFAULT_COUNT,
    hotspot_size: int = H_DYNAMIC_DEFAULT,
) -> List[ProposalIntent]:
    return ProposalEngine(state.agent_id, experiment_seed, hotspot_size).wave_b(
        state, real_message, macro_round, intervention, count
    )


__all__ = [
    "H_DYNAMIC_DEFAULT",
    "WAVE_A_DEFAULT_COUNT",
    "WAVE_B_DEFAULT_COUNT",
    "BoundaryState",
    "BoundaryMessage",
    "MessageIntervention",
    "DynamicHotspot",
    "ProposalIntent",
    "ComplianceRecord",
    "MaterializedProposal",
    "ProposalEngine",
    "stable_hash",
    "derived_seed",
    "select_dynamic_hotspot",
    "apply_message_intervention",
    "generate_wave_a_intents",
    "generate_wave_b_intents",
    "measure_compliance",
    "materialize_intent",
    "solver_compliance",
]
