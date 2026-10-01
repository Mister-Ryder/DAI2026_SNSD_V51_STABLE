"""Deterministic boundary intents for TriGraph-CBCR.

This module deliberately stops before the local worker.  It turns a verified
boundary state (and, for wave B, an intervened peer message) into complete
boundary requests.  A request can then be projected by the v1
``CandidateGenerator`` and passed unchanged to the v1
``ConditionedLocalSolver``.

Two invariants are central:

* every intent specifies *all* boundary bits through ``requested_fix_on`` and
  ``requested_fix_off``;
* a non-hotspot bit always inherits the verified incumbent value.

The stable hashes in this file never use Python's process-randomised ``hash``.
Consequently proposal streams are reproducible across processes for a fixed
state, base_protocol seed and intervention.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


H_DYNAMIC_DEFAULT = 64
WAVE_A_DEFAULT_COUNT = 24
WAVE_B_DEFAULT_COUNT = 24


def _as_builtin(value: Any) -> Any:
    """Return a JSON-stable representation, including NumPy scalar inputs."""

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("stable hashes do not accept non-finite floats")
        if value == 0.0:
            value = 0.0
        # Encoding the shortest platform-independent decimal as a tagged value
        # avoids JSON implementation differences while retaining float identity.
        return {"__float__": format(value, ".17g")}
    if isinstance(value, Mapping):
        return {
            str(key): _as_builtin(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        converted = [_as_builtin(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (tuple, list)):
        return [_as_builtin(item) for item in value]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return _as_builtin(value.to_dict())
    raise TypeError("cannot build a stable hash from %r" % (type(value),))


def stable_hash(*parts: Any) -> str:
    """SHA256 over a canonical JSON encoding."""

    raw = json.dumps(
        _as_builtin(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def derived_seed(*parts: Any) -> int:
    """Derive a stable 32-bit seed suitable for ``random`` or NumPy 1.21."""

    return int(stable_hash(*parts)[:8], 16)


def _ordered_subset(order: Sequence[int], values: Iterable[int]) -> Tuple[int, ...]:
    wanted = set(int(value) for value in values)
    return tuple(int(node_id) for node_id in order if int(node_id) in wanted)


def _aligned_values(
    boundary_order: Sequence[int],
    values: Optional[Any],
    field_name: str,
) -> Tuple[float, ...]:
    if values is None:
        result = [0.0 for _ in boundary_order]
    elif isinstance(values, Mapping):
        result = []
        for node_id in boundary_order:
            if node_id in values:
                raw = values[node_id]
            elif str(node_id) in values:
                raw = values[str(node_id)]
            else:
                raw = 0.0
            result.append(float(raw))
    else:
        result = [float(value) for value in values]
        if len(result) != len(boundary_order):
            raise ValueError(
                "%s has length %d, expected %d"
                % (field_name, len(result), len(boundary_order))
            )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("%s contains a non-finite value" % field_name)
    return tuple(result)


@dataclass(frozen=True)
class BoundaryState:
    """Complete, aligned state for one agent's boundary."""

    agent_id: int
    boundary_order: Tuple[int, ...]
    incumbent_selected: Tuple[int, ...]
    raw_benefit: Tuple[float, ...]
    local_marginal: Tuple[float, ...]
    conflict_score: Tuple[float, ...]
    prior_opportunity_cost: Tuple[float, ...]
    uncertainty: Tuple[float, ...]
    prior_pressure: Tuple[float, ...]
    boundary_order_hash: str
    state_hash: str

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
        order = tuple(int(node_id) for node_id in boundary_order)
        if len(order) != len(set(order)):
            raise ValueError("boundary_order contains duplicate node ids")
        selected = _ordered_subset(order, incumbent_selected)
        if set(int(value) for value in incumbent_selected).difference(order):
            raise ValueError("incumbent_selected contains a non-boundary node")
        arrays = {
            "raw_benefit": _aligned_values(order, raw_benefit, "raw_benefit"),
            "local_marginal": _aligned_values(
                order, local_marginal, "local_marginal"
            ),
            "conflict_score": _aligned_values(
                order, conflict_score, "conflict_score"
            ),
            "prior_opportunity_cost": _aligned_values(
                order, prior_opportunity_cost, "prior_opportunity_cost"
            ),
            "uncertainty": _aligned_values(order, uncertainty, "uncertainty"),
            "prior_pressure": _aligned_values(
                order, prior_pressure, "prior_pressure"
            ),
        }
        order_hash = stable_hash("boundary-order-v1", int(agent_id), order)
        state_hash = stable_hash(
            "boundary-state-v1", int(agent_id), order_hash, selected, arrays
        )
        return cls(
            agent_id=int(agent_id),
            boundary_order=order,
            incumbent_selected=selected,
            raw_benefit=arrays["raw_benefit"],
            local_marginal=arrays["local_marginal"],
            conflict_score=arrays["conflict_score"],
            prior_opportunity_cost=arrays["prior_opportunity_cost"],
            uncertainty=arrays["uncertainty"],
            prior_pressure=arrays["prior_pressure"],
            boundary_order_hash=order_hash,
            state_hash=state_hash,
        )

    def value_map(self, field_name: str) -> Dict[int, float]:
        values = getattr(self, field_name)
        return {
            node_id: float(values[index])
            for index, node_id in enumerate(self.boundary_order)
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "boundary_order": list(self.boundary_order),
            "incumbent_selected": list(self.incumbent_selected),
            "raw_benefit": list(self.raw_benefit),
            "local_marginal": list(self.local_marginal),
            "conflict_score": list(self.conflict_score),
            "prior_opportunity_cost": list(self.prior_opportunity_cost),
            "uncertainty": list(self.uncertainty),
            "prior_pressure": list(self.prior_pressure),
            "boundary_order_hash": self.boundary_order_hash,
            "state_hash": self.state_hash,
        }


class MessageIntervention(str, Enum):
    REAL = "real"
    ZERO = "zero"
    SHUFFLED = "shuffled"


@dataclass(frozen=True)
class BoundaryMessage:
    """Node-aligned peer information consumed by wave-B proposal generation."""

    boundary_order: Tuple[int, ...]
    opportunity_cost: Tuple[float, ...]
    pressure: Tuple[float, ...]
    neighbor_recovery: Tuple[float, ...]
    conflict_score: Tuple[float, ...]
    uncertainty: Tuple[float, ...]
    intervention: str
    original_message_hash: str
    message_hash: str
    permutation: Tuple[int, ...]

    @classmethod
    def from_maps(
        cls,
        boundary_order: Sequence[int],
        opportunity_cost: Optional[Any] = None,
        pressure: Optional[Any] = None,
        neighbor_recovery: Optional[Any] = None,
        conflict_score: Optional[Any] = None,
        uncertainty: Optional[Any] = None,
    ) -> "BoundaryMessage":
        order = tuple(int(node_id) for node_id in boundary_order)
        if len(order) != len(set(order)):
            raise ValueError("message boundary_order contains duplicate node ids")
        channels = {
            "opportunity_cost": _aligned_values(
                order, opportunity_cost, "opportunity_cost"
            ),
            "pressure": _aligned_values(order, pressure, "pressure"),
            "neighbor_recovery": _aligned_values(
                order, neighbor_recovery, "neighbor_recovery"
            ),
            "conflict_score": _aligned_values(
                order, conflict_score, "message conflict_score"
            ),
            "uncertainty": _aligned_values(
                order, uncertainty, "message uncertainty"
            ),
        }
        digest = stable_hash("boundary-message-v1", order, channels)
        return cls(
            boundary_order=order,
            opportunity_cost=channels["opportunity_cost"],
            pressure=channels["pressure"],
            neighbor_recovery=channels["neighbor_recovery"],
            conflict_score=channels["conflict_score"],
            uncertainty=channels["uncertainty"],
            intervention=MessageIntervention.REAL.value,
            original_message_hash=digest,
            message_hash=digest,
            permutation=tuple(range(len(order))),
        )

    def value_map(self, field_name: str) -> Dict[int, float]:
        values = getattr(self, field_name)
        return {
            node_id: float(values[index])
            for index, node_id in enumerate(self.boundary_order)
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "boundary_order": list(self.boundary_order),
            "opportunity_cost": list(self.opportunity_cost),
            "pressure": list(self.pressure),
            "neighbor_recovery": list(self.neighbor_recovery),
            "conflict_score": list(self.conflict_score),
            "uncertainty": list(self.uncertainty),
            "intervention": self.intervention,
            "original_message_hash": self.original_message_hash,
            "message_hash": self.message_hash,
            "permutation": list(self.permutation),
        }


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
    """Return Real, Zero or deterministically Shuffled peer information.

    Zero intentionally does not include ``message.original_message_hash`` in its
    output hash.  Thus two different hidden real messages produce the identical
    zero-message proposal stream, which is required for a clean intervention.
    Shuffling reassigns complete channel vectors to other boundary nodes and
    therefore preserves every channel's multiset and the message byte shape.
    """

    intervention = _normalise_intervention(mode)
    order = message.boundary_order
    identity = tuple(range(len(order)))
    if intervention is MessageIntervention.REAL:
        digest = stable_hash(
            "intervened-message-v1",
            intervention.value,
            order,
            message.opportunity_cost,
            message.pressure,
            message.neighbor_recovery,
            message.conflict_score,
            message.uncertainty,
        )
        return BoundaryMessage(
            boundary_order=order,
            opportunity_cost=message.opportunity_cost,
            pressure=message.pressure,
            neighbor_recovery=message.neighbor_recovery,
            conflict_score=message.conflict_score,
            uncertainty=message.uncertainty,
            intervention=intervention.value,
            original_message_hash=message.original_message_hash,
            message_hash=digest,
            permutation=identity,
        )
    if intervention is MessageIntervention.ZERO:
        zeros = tuple(0.0 for _ in order)
        digest = stable_hash("intervened-message-v1", intervention.value, order, zeros)
        return BoundaryMessage(
            boundary_order=order,
            opportunity_cost=zeros,
            pressure=zeros,
            neighbor_recovery=zeros,
            conflict_score=zeros,
            uncertainty=zeros,
            intervention=intervention.value,
            original_message_hash="masked",
            message_hash=digest,
            permutation=identity,
        )

    n = len(order)
    if n <= 1:
        permutation = identity
    else:
        shift_seed = derived_seed(
            "message-shuffle-v1",
            int(experiment_seed),
            int(agent_id),
            int(macro_round),
            message.original_message_hash,
        )
        shift = 1 + (shift_seed % (n - 1))
        permutation = tuple((index + shift) % n for index in range(n))

    def permute(values: Sequence[float]) -> Tuple[float, ...]:
        return tuple(float(values[source]) for source in permutation)

    opportunity_cost = permute(message.opportunity_cost)
    pressure = permute(message.pressure)
    neighbor_recovery = permute(message.neighbor_recovery)
    conflict_score = permute(message.conflict_score)
    uncertainty = permute(message.uncertainty)
    digest = stable_hash(
        "intervened-message-v1",
        intervention.value,
        order,
        opportunity_cost,
        pressure,
        neighbor_recovery,
        conflict_score,
        uncertainty,
        permutation,
    )
    return BoundaryMessage(
        boundary_order=order,
        opportunity_cost=opportunity_cost,
        pressure=pressure,
        neighbor_recovery=neighbor_recovery,
        conflict_score=conflict_score,
        uncertainty=uncertainty,
        intervention=intervention.value,
        original_message_hash=message.original_message_hash,
        message_hash=digest,
        permutation=permutation,
    )


def _rank_percentiles(
    nodes: Sequence[int],
    value_by_node: Mapping[int, float],
    tie_context: Any,
) -> Tuple[Tuple[int, ...], Dict[int, float]]:
    ranked = tuple(
        sorted(
            (int(node_id) for node_id in nodes),
            key=lambda node_id: (
                -float(value_by_node.get(node_id, 0.0)),
                derived_seed("rank-tie-v1", tie_context, node_id),
                node_id,
            ),
        )
    )
    if len(ranked) <= 1:
        return ranked, {node_id: 1.0 for node_id in ranked}
    scale = float(len(ranked) - 1)
    percentiles = {
        node_id: 1.0 - float(rank) / scale for rank, node_id in enumerate(ranked)
    }
    return ranked, percentiles


@dataclass(frozen=True)
class DynamicHotspot:
    agent_id: int
    macro_round: int
    requested_size: int
    effective_size: int
    node_ids: Tuple[int, ...]
    benefit_ranked: Tuple[int, ...]
    opportunity_ranked: Tuple[int, ...]
    uncertainty_ranked: Tuple[int, ...]
    pressure_ranked: Tuple[int, ...]
    composite_ranked: Tuple[int, ...]
    reasons: Tuple[Tuple[int, Tuple[str, ...]], ...]
    state_hash: str
    hotspot_hash: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "macro_round": self.macro_round,
            "requested_size": self.requested_size,
            "effective_size": self.effective_size,
            "node_ids": list(self.node_ids),
            "benefit_ranked": list(self.benefit_ranked),
            "opportunity_ranked": list(self.opportunity_ranked),
            "uncertainty_ranked": list(self.uncertainty_ranked),
            "pressure_ranked": list(self.pressure_ranked),
            "composite_ranked": list(self.composite_ranked),
            "reasons": [
                {"node_id": node_id, "signals": list(signals)}
                for node_id, signals in self.reasons
            ],
            "state_hash": self.state_hash,
            "hotspot_hash": self.hotspot_hash,
        }


def select_dynamic_hotspot(
    state: BoundaryState,
    experiment_seed: int,
    macro_round: int,
    hotspot_size: int = H_DYNAMIC_DEFAULT,
) -> DynamicHotspot:
    """Select the deterministic union of incumbent and four signal rankings.

    ``hotspot_size`` is the normal target (64).  If a verified incumbent has
    more selected boundary bits than the target, the effective hotspot expands
    rather than silently excluding an incumbent one-bit.
    """

    requested_size = max(1, int(hotspot_size))
    nodes = state.boundary_order
    target = min(len(nodes), max(requested_size, len(state.incumbent_selected)))
    raw = state.value_map("raw_benefit")
    local = state.value_map("local_marginal")
    conflicts = state.value_map("conflict_score")
    opportunity = state.value_map("prior_opportunity_cost")
    uncertainty = state.value_map("uncertainty")
    pressure = state.value_map("prior_pressure")
    context = (
        "dynamic-hotspot-v1",
        int(experiment_seed),
        state.agent_id,
        int(macro_round),
        state.state_hash,
    )

    _, raw_pct = _rank_percentiles(nodes, raw, (context, "raw"))
    _, local_pct = _rank_percentiles(nodes, local, (context, "local"))
    _, conflict_pct = _rank_percentiles(nodes, conflicts, (context, "conflict"))
    _, opportunity_pct = _rank_percentiles(
        nodes, opportunity, (context, "opportunity")
    )
    _, uncertainty_pct = _rank_percentiles(
        nodes, uncertainty, (context, "uncertainty")
    )
    abs_pressure = {node_id: abs(pressure[node_id]) for node_id in nodes}
    _, pressure_pct = _rank_percentiles(
        nodes, abs_pressure, (context, "pressure")
    )

    benefit_signal = {
        node_id: 0.55 * raw_pct[node_id] + 0.45 * local_pct[node_id]
        for node_id in nodes
    }
    opportunity_signal = {
        node_id: 0.6 * opportunity_pct[node_id] + 0.4 * conflict_pct[node_id]
        for node_id in nodes
    }
    _, benefit_pct = _rank_percentiles(
        nodes, benefit_signal, (context, "benefit-combined")
    )
    _, opportunity_combined_pct = _rank_percentiles(
        nodes, opportunity_signal, (context, "opportunity-combined")
    )
    composite = {
        node_id: (
            0.30 * benefit_pct[node_id]
            + 0.25 * opportunity_combined_pct[node_id]
            + 0.20 * uncertainty_pct[node_id]
            + 0.25 * pressure_pct[node_id]
        )
        for node_id in nodes
    }
    benefit_ranked, _ = _rank_percentiles(
        nodes, benefit_signal, (context, "benefit-order")
    )
    opportunity_ranked, _ = _rank_percentiles(
        nodes, opportunity_signal, (context, "opportunity-order")
    )
    uncertainty_ranked, _ = _rank_percentiles(
        nodes, uncertainty, (context, "uncertainty-order")
    )
    pressure_ranked, _ = _rank_percentiles(
        nodes, abs_pressure, (context, "pressure-order")
    )
    composite_ranked, _ = _rank_percentiles(
        nodes, composite, (context, "composite-order")
    )

    reasons_by_node: Dict[int, Set[str]] = {}
    chosen: List[int] = []
    chosen_set: Set[int] = set()

    def add(node_id: int, reason: str) -> None:
        node_id = int(node_id)
        reasons_by_node.setdefault(node_id, set()).add(reason)
        if node_id not in chosen_set and len(chosen) < target:
            chosen.append(node_id)
            chosen_set.add(node_id)

    selected_ranked = sorted(
        state.incumbent_selected,
        key=lambda node_id: (
            -composite[node_id],
            derived_seed("selected-hotspot-tie-v1", context, node_id),
            node_id,
        ),
    )
    for node_id in selected_ranked:
        add(node_id, "incumbent_selected")

    ranked_sources = (
        ("benefit", benefit_ranked),
        ("opportunity", opportunity_ranked),
        ("uncertainty", uncertainty_ranked),
        ("pressure", pressure_ranked),
    )
    cursors = [0, 0, 0, 0]
    while len(chosen) < target:
        progressed = False
        for source_index, (name, ranked) in enumerate(ranked_sources):
            while cursors[source_index] < len(ranked):
                node_id = ranked[cursors[source_index]]
                cursors[source_index] += 1
                if node_id not in chosen_set:
                    add(node_id, name)
                    progressed = True
                    break
                reasons_by_node.setdefault(node_id, set()).add(name)
            if len(chosen) >= target:
                break
        if not progressed:
            break
    for node_id in composite_ranked:
        if len(chosen) >= target:
            break
        add(node_id, "composite")

    reasons = tuple(
        (node_id, tuple(sorted(reasons_by_node.get(node_id, {"composite"}))))
        for node_id in chosen
    )
    digest = stable_hash(
        "dynamic-hotspot-v1",
        state.agent_id,
        int(macro_round),
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
            "anchor_selected_boundary": list(self.anchor_selected_boundary),
            "hotspot_ids": list(self.hotspot_ids),
            "requested_fix_on": list(self.requested_fix_on),
            "requested_fix_off": list(self.requested_fix_off),
            "requested_selected_boundary": list(
                self.requested_selected_boundary
            ),
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
class ComplianceRecord:
    intent_hash: str
    signature_hash: str
    solver_status: str
    requested_fix_on: Tuple[int, ...]
    requested_fix_off: Tuple[int, ...]
    requested_selected_boundary: Tuple[int, ...]
    realized_selected_boundary: Tuple[int, ...]
    realized_fix_on: Tuple[int, ...]
    realized_fix_off: Tuple[int, ...]
    violated_fix_on: Tuple[int, ...]
    violated_fix_off: Tuple[int, ...]
    requested_added_ids: Tuple[int, ...]
    requested_removed_ids: Tuple[int, ...]
    realized_added_ids: Tuple[int, ...]
    realized_removed_ids: Tuple[int, ...]
    fix_on_compliance: float
    fix_off_compliance: float
    full_compliance: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "intent_hash": self.intent_hash,
            "signature_hash": self.signature_hash,
            "solver_status": self.solver_status,
            "requested_fix_on": list(self.requested_fix_on),
            "requested_fix_off": list(self.requested_fix_off),
            "requested_selected_boundary": list(
                self.requested_selected_boundary
            ),
            "realized_selected_boundary": list(self.realized_selected_boundary),
            "realized_fix_on": list(self.realized_fix_on),
            "realized_fix_off": list(self.realized_fix_off),
            "violated_fix_on": list(self.violated_fix_on),
            "violated_fix_off": list(self.violated_fix_off),
            "requested_added_ids": list(self.requested_added_ids),
            "requested_removed_ids": list(self.requested_removed_ids),
            "realized_added_ids": list(self.realized_added_ids),
            "realized_removed_ids": list(self.realized_removed_ids),
            "fix_on_compliance": self.fix_on_compliance,
            "fix_off_compliance": self.fix_off_compliance,
            "full_compliance": self.full_compliance,
        }


@dataclass(frozen=True)
class MaterializedProposal:
    """A v1 ``BoundarySignature`` plus projection-compliance evidence."""

    intent: ProposalIntent
    signature: Any
    projection_compliance: ComplianceRecord

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
        }


def measure_compliance(
    intent: ProposalIntent,
    realized_selected_ids: Iterable[int],
    signature_hash: str = "",
    solver_status: str = "",
) -> ComplianceRecord:
    """Compare requested full boundary bits with projection or worker output."""

    boundary = tuple(intent.requested_fix_on + intent.requested_fix_off)
    boundary_set = set(boundary)
    realized_set = set(int(value) for value in realized_selected_ids).intersection(
        boundary_set
    )
    requested_on = set(intent.requested_fix_on)
    requested_off = set(intent.requested_fix_off)
    anchor = set(intent.anchor_selected_boundary)
    realized_on = _ordered_subset(boundary, realized_set)
    realized_off = _ordered_subset(boundary, boundary_set.difference(realized_set))
    violated_on = _ordered_subset(boundary, requested_on.difference(realized_set))
    violated_off = _ordered_subset(boundary, requested_off.intersection(realized_set))
    fix_on_compliance = (
        float(len(requested_on.intersection(realized_set))) / float(len(requested_on))
        if requested_on
        else 1.0
    )
    fix_off_compliance = (
        float(len(requested_off.difference(realized_set))) / float(len(requested_off))
        if requested_off
        else 1.0
    )
    return ComplianceRecord(
        intent_hash=intent.intent_hash,
        signature_hash=str(signature_hash),
        solver_status=str(solver_status),
        requested_fix_on=intent.requested_fix_on,
        requested_fix_off=intent.requested_fix_off,
        requested_selected_boundary=intent.requested_selected_boundary,
        realized_selected_boundary=_ordered_subset(boundary, realized_set),
        realized_fix_on=realized_on,
        realized_fix_off=realized_off,
        violated_fix_on=violated_on,
        violated_fix_off=violated_off,
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
) -> MaterializedProposal:
    """Project an intent with the v1 ``CandidateGenerator.make_signature`` API."""

    if int(getattr(candidate_generator, "agent_id")) != intent.agent_id:
        raise ValueError("candidate generator belongs to a different agent")
    generator_boundary = tuple(int(value) for value in candidate_generator.boundary)
    if set(generator_boundary) != set(
        intent.requested_fix_on + intent.requested_fix_off
    ):
        raise ValueError("candidate generator boundary differs from intent boundary")
    if anchor_signature is not None and set(anchor_signature.selected_ids) != set(
        intent.anchor_selected_boundary
    ):
        raise ValueError("anchor signature differs from intent incumbent")
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
    return MaterializedProposal(
        intent=intent, signature=signature, projection_compliance=compliance
    )


def solver_compliance(
    materialized: MaterializedProposal,
    solve_result: Any,
) -> ComplianceRecord:
    """Create the final requested-realized audit row after a v1 worker call."""

    return measure_compliance(
        materialized.intent,
        getattr(solve_result, "selected_ids"),
        signature_hash=str(getattr(solve_result, "signature_hash", "")),
        solver_status=str(getattr(solve_result, "status", "")),
    )


def _allocate_quotas(count: int, weights: Sequence[int]) -> Tuple[int, ...]:
    count = max(0, int(count))
    total = float(sum(int(weight) for weight in weights))
    if count == 0:
        return tuple(0 for _ in weights)
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
    """Generate deterministic wave-A and message-conditioned wave-B intents."""

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
        if state.agent_id != self.agent_id:
            raise ValueError("boundary state belongs to a different agent")

    def dynamic_hotspot(
        self, state: BoundaryState, macro_round: int
    ) -> DynamicHotspot:
        self._check_state(state)
        return select_dynamic_hotspot(
            state,
            experiment_seed=self.experiment_seed,
            macro_round=macro_round,
            hotspot_size=self.hotspot_size,
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
        order = state.boundary_order
        boundary_set = set(order)
        hotspot_set = set(hotspot.node_ids)
        anchor = set(state.incumbent_selected)
        requested = set(int(value) for value in requested_selected).intersection(
            boundary_set
        )
        for node_id in boundary_set.difference(hotspot_set):
            if (node_id in requested) != (node_id in anchor):
                raise ValueError("a non-hotspot bit differs from the incumbent")
        fix_on = _ordered_subset(order, requested)
        fix_off = _ordered_subset(order, boundary_set.difference(requested))
        added = _ordered_subset(order, requested.difference(anchor))
        removed = _ordered_subset(order, anchor.difference(requested))
        random_seed = derived_seed(
            "proposal-intent-rng-v1",
            self.experiment_seed,
            self.agent_id,
            int(macro_round),
            proposal_wave,
            intent_type,
            int(slot),
            state.state_hash,
            source_message_hash,
        )
        payload = {
            "agent_id": self.agent_id,
            "macro_round": int(macro_round),
            "proposal_wave": proposal_wave,
            "intent_type": intent_type,
            "slot": int(slot),
            "anchor": state.incumbent_selected,
            "hotspot": hotspot.node_ids,
            "fix_on": fix_on,
            "fix_off": fix_off,
            "source_message_hash": source_message_hash,
            "message_intervention": message_intervention,
            "top_pressure_ids": tuple(int(value) for value in top_pressure_ids),
            "top_pressure_values": tuple(float(value) for value in top_pressure_values),
            "boundary_order_hash": state.boundary_order_hash,
            "state_hash": state.state_hash,
            "hotspot_hash": hotspot.hotspot_hash,
            "random_seed": random_seed,
        }
        intent_hash = stable_hash("proposal-intent-v1", payload)
        return ProposalIntent(
            agent_id=self.agent_id,
            macro_round=int(macro_round),
            proposal_wave=str(proposal_wave),
            intent_type=str(intent_type),
            slot=int(slot),
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
            slot = len(result)
            intent = self._make_intent(
                state=state,
                hotspot=hotspot,
                macro_round=macro_round,
                proposal_wave=wave,
                intent_type=intent_type,
                slot=slot,
                requested_selected=requested,
                source_message_hash=source_message_hash,
                message_intervention=intervention,
                top_pressure_ids=top_pressure_ids,
                top_pressure_values=top_pressure_values,
            )
            result.append(intent)
            seen.add(ordered)
            if len(result) - start_count >= target_for_type:
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
        rng = random.Random(
            derived_seed(
                "proposal-fallback-v1",
                self.experiment_seed,
                self.agent_id,
                macro_round,
                wave,
                intent_type,
                state.state_hash,
                source_message_hash,
            )
        )
        attempts = 0
        max_attempts = max(512, int(count) * 128)
        while len(result) < count and mutable and attempts < max_attempts:
            attempts += 1
            width = 1 + (attempts - 1) % min(4, len(mutable))
            toggled = rng.sample(mutable, width)
            requested = set(anchor)
            for node_id in toggled:
                if node_id in requested:
                    requested.remove(node_id)
                else:
                    requested.add(node_id)
            ordered = _ordered_subset(state.boundary_order, requested)
            if ordered == state.incumbent_selected or ordered in seen:
                continue
            intent = self._make_intent(
                state,
                hotspot,
                macro_round,
                wave,
                intent_type,
                len(result),
                requested,
                source_message_hash,
                intervention,
                top_pressure_ids,
                top_pressure_values,
            )
            result.append(intent)
            seen.add(ordered)

    def wave_a(
        self,
        state: BoundaryState,
        macro_round: int,
        count: int = WAVE_A_DEFAULT_COUNT,
    ) -> List[ProposalIntent]:
        """Generate benefit, release, swap and coverage-probe intents."""

        self._check_state(state)
        count = max(0, int(count))
        if count == 0:
            return []
        hotspot = self.dynamic_hotspot(state, macro_round)
        hotspot_set = set(hotspot.node_ids)
        selected = set(state.incumbent_selected)
        raw = state.value_map("raw_benefit")
        local = state.value_map("local_marginal")
        conflict = state.value_map("conflict_score")
        opportunity = state.value_map("prior_opportunity_cost")
        uncertainty = state.value_map("uncertainty")
        pressure = state.value_map("prior_pressure")
        context = (
            "wave-a-v1",
            self.experiment_seed,
            self.agent_id,
            int(macro_round),
            state.state_hash,
        )

        nodes = list(hotspot.node_ids)
        _, raw_pct = _rank_percentiles(nodes, raw, (context, "raw"))
        _, local_pct = _rank_percentiles(nodes, local, (context, "local"))
        _, conflict_pct = _rank_percentiles(nodes, conflict, (context, "conflict"))
        _, opp_pct = _rank_percentiles(nodes, opportunity, (context, "opportunity"))
        _, uncertainty_pct = _rank_percentiles(
            nodes, uncertainty, (context, "uncertainty")
        )
        benefit_value = {
            node_id: 0.55 * raw_pct[node_id] + 0.45 * local_pct[node_id]
            - 0.15 * conflict_pct[node_id]
            - 0.10 * opp_pct[node_id]
            for node_id in nodes
        }
        release_value = {
            node_id: 0.45 * conflict_pct[node_id] + 0.45 * opp_pct[node_id]
            - 0.10 * raw_pct[node_id]
            for node_id in nodes
        }
        near_zero = {
            node_id: 1.0 / (1.0 + abs(float(pressure[node_id])))
            for node_id in nodes
        }
        explore_value = {
            node_id: 0.75 * uncertainty_pct[node_id] + 0.25 * near_zero[node_id]
            for node_id in nodes
        }
        add_ranked, _ = _rank_percentiles(
            [node for node in nodes if node not in selected],
            benefit_value,
            (context, "add"),
        )
        release_ranked, _ = _rank_percentiles(
            [node for node in nodes if node in selected],
            release_value,
            (context, "release"),
        )
        explore_ranked, _ = _rank_percentiles(
            nodes, explore_value, (context, "explore")
        )
        pressure_ranked = sorted(
            nodes,
            key=lambda node_id: (
                -abs(pressure[node_id]),
                derived_seed("wave-a-pressure-tie-v1", context, node_id),
                node_id,
            ),
        )
        top_pressure_ids = tuple(pressure_ranked[: min(8, len(pressure_ranked))])
        top_pressure_values = tuple(pressure[node_id] for node_id in top_pressure_ids)
        source_hash = stable_hash("wave-a-no-peer-message-v1", state.state_hash)
        quotas = _allocate_quotas(count, (8, 8, 4, 4))
        result: List[ProposalIntent] = []
        seen: Set[Tuple[int, ...]] = set()

        benefit_patterns = itertools.chain(
            ((combo, ()) for combo in itertools.combinations(add_ranked[:16], 1)),
            ((combo, ()) for combo in itertools.combinations(add_ranked[:16], 2)),
            (
                ((add_id,), (remove_id,))
                for add_id in add_ranked[:12]
                for remove_id in reversed(release_ranked[:8])
            ),
        )
        self._append_patterns(
            result, seen, quotas[0], state, hotspot, macro_round, "A", "benefit",
            benefit_patterns, source_hash, "none", top_pressure_ids,
            top_pressure_values,
        )

        release_patterns = itertools.chain(
            (((), combo) for combo in itertools.combinations(release_ranked[:16], 1)),
            (((), combo) for combo in itertools.combinations(release_ranked[:16], 2)),
            (((), combo) for combo in itertools.combinations(release_ranked[:12], 3)),
        )
        self._append_patterns(
            result, seen, quotas[1], state, hotspot, macro_round, "A", "release",
            release_patterns, source_hash, "none", top_pressure_ids,
            top_pressure_values,
        )

        swap_patterns = (
            ((add_id,), (remove_id,))
            for remove_id in release_ranked[:12]
            for add_id in add_ranked[:12]
        )
        self._append_patterns(
            result, seen, quotas[2], state, hotspot, macro_round, "A", "swap",
            swap_patterns, source_hash, "none", top_pressure_ids,
            top_pressure_values,
        )

        explore_patterns = itertools.chain(
            (
                ((node_id,), ()) if node_id not in selected else ((), (node_id,))
                for node_id in explore_ranked[:24]
            ),
            (
                (
                    tuple(node for node in pair if node not in selected),
                    tuple(node for node in pair if node in selected),
                )
                for pair in itertools.combinations(explore_ranked[:16], 2)
            ),
        )
        self._append_patterns(
            result, seen, quotas[3], state, hotspot, macro_round, "A", "probe",
            explore_patterns, source_hash, "none", top_pressure_ids,
            top_pressure_values,
        )

        self._fallback_fill(
            result, seen, count, state, hotspot, macro_round, "A", "probe",
            source_hash, "none", top_pressure_ids, top_pressure_values,
        )
        return result[:count]

    def _state_with_message(
        self, state: BoundaryState, message: BoundaryMessage
    ) -> BoundaryState:
        if tuple(message.boundary_order) != tuple(state.boundary_order):
            raise ValueError("message boundary order differs from boundary state")
        opportunity = tuple(
            float(state.prior_opportunity_cost[index])
            + abs(float(message.opportunity_cost[index]))
            + max(0.0, float(message.neighbor_recovery[index]))
            for index in range(len(state.boundary_order))
        )
        pressure = tuple(
            float(state.prior_pressure[index]) + float(message.pressure[index])
            for index in range(len(state.boundary_order))
        )
        conflict = tuple(
            float(state.conflict_score[index])
            + abs(float(message.conflict_score[index]))
            for index in range(len(state.boundary_order))
        )
        uncertainty = tuple(
            max(float(state.uncertainty[index]), abs(float(message.uncertainty[index])))
            for index in range(len(state.boundary_order))
        )
        return BoundaryState.from_maps(
            agent_id=state.agent_id,
            boundary_order=state.boundary_order,
            incumbent_selected=state.incumbent_selected,
            raw_benefit=state.raw_benefit,
            local_marginal=state.local_marginal,
            conflict_score=conflict,
            prior_opportunity_cost=opportunity,
            uncertainty=uncertainty,
            prior_pressure=pressure,
        )

    def wave_b(
        self,
        state: BoundaryState,
        real_message: BoundaryMessage,
        macro_round: int,
        intervention: Any = MessageIntervention.REAL,
        count: int = WAVE_B_DEFAULT_COUNT,
    ) -> List[ProposalIntent]:
        """Generate yield/retreat/replace/explore counterproposal intents."""

        self._check_state(state)
        count = max(0, int(count))
        if count == 0:
            return []
        message = apply_message_intervention(
            real_message,
            intervention,
            experiment_seed=self.experiment_seed,
            agent_id=self.agent_id,
            macro_round=macro_round,
        )
        if tuple(message.boundary_order) != tuple(state.boundary_order):
            raise ValueError("message boundary order differs from boundary state")
        conditioned_state = self._state_with_message(state, message)
        hotspot = select_dynamic_hotspot(
            conditioned_state,
            experiment_seed=self.experiment_seed,
            macro_round=macro_round,
            hotspot_size=self.hotspot_size,
        )
        nodes = list(hotspot.node_ids)
        selected = set(state.incumbent_selected)
        raw = state.value_map("raw_benefit")
        local = state.value_map("local_marginal")
        base_conflict = state.value_map("conflict_score")
        msg_opp = message.value_map("opportunity_cost")
        msg_pressure = message.value_map("pressure")
        msg_recovery = message.value_map("neighbor_recovery")
        msg_conflict = message.value_map("conflict_score")
        msg_uncertainty = message.value_map("uncertainty")
        context = (
            "wave-b-v1",
            self.experiment_seed,
            self.agent_id,
            int(macro_round),
            state.state_hash,
            message.message_hash,
        )

        _, raw_pct = _rank_percentiles(nodes, raw, (context, "raw"))
        _, local_pct = _rank_percentiles(nodes, local, (context, "local"))
        _, opp_pct = _rank_percentiles(
            nodes,
            {node: abs(msg_opp[node]) for node in nodes},
            (context, "opportunity"),
        )
        _, recovery_pct = _rank_percentiles(
            nodes,
            {node: max(0.0, msg_recovery[node]) for node in nodes},
            (context, "recovery"),
        )
        _, conflict_pct = _rank_percentiles(
            nodes,
            {
                node: abs(msg_conflict[node]) + max(0.0, base_conflict[node])
                for node in nodes
            },
            (context, "conflict"),
        )
        _, uncertainty_pct = _rank_percentiles(
            nodes,
            {
                node: max(abs(msg_uncertainty[node]), state.uncertainty[
                    state.boundary_order.index(node)
                ])
                for node in nodes
            },
            (context, "uncertainty"),
        )
        _, pressure_abs_pct = _rank_percentiles(
            nodes,
            {node: abs(msg_pressure[node]) for node in nodes},
            (context, "pressure-abs"),
        )
        benefit_pct = {
            node: 0.55 * raw_pct[node] + 0.45 * local_pct[node] for node in nodes
        }
        yield_value = {
            node: (
                0.35 * opp_pct[node]
                + 0.30 * recovery_pct[node]
                + 0.20 * conflict_pct[node]
                + 0.10 * float(msg_pressure[node] < 0.0)
                - 0.15 * benefit_pct[node]
            )
            for node in nodes
        }
        add_value = {
            node: (
                0.45 * benefit_pct[node]
                + 0.35 * float(msg_pressure[node] > 0.0) * pressure_abs_pct[node]
                + 0.20 * recovery_pct[node]
                - 0.20 * opp_pct[node]
                - 0.15 * conflict_pct[node]
            )
            for node in nodes
        }
        explore_value = {
            node: (
                0.65 * uncertainty_pct[node]
                + 0.35 / (1.0 + abs(float(msg_pressure[node])))
            )
            for node in nodes
        }
        yield_ranked, _ = _rank_percentiles(
            [node for node in nodes if node in selected],
            yield_value,
            (context, "yield"),
        )
        add_ranked, _ = _rank_percentiles(
            [node for node in nodes if node not in selected],
            add_value,
            (context, "replace-add"),
        )
        explore_ranked, _ = _rank_percentiles(
            nodes, explore_value, (context, "explore")
        )
        top_pressure_ranked = sorted(
            nodes,
            key=lambda node: (
                -(
                    abs(msg_pressure[node])
                    + abs(msg_opp[node])
                    + max(0.0, msg_recovery[node])
                ),
                derived_seed("wave-b-pressure-tie-v1", context, node),
                node,
            ),
        )
        top_pressure_ids = tuple(
            top_pressure_ranked[: min(8, len(top_pressure_ranked))]
        )
        top_pressure_values = tuple(
            float(msg_pressure[node]) for node in top_pressure_ids
        )
        quotas = _allocate_quotas(count, (1, 1, 1, 1))
        result: List[ProposalIntent] = []
        seen: Set[Tuple[int, ...]] = set()

        yield_patterns = itertools.chain(
            (((), combo) for combo in itertools.combinations(yield_ranked[:16], 1)),
            (((), combo) for combo in itertools.combinations(yield_ranked[:16], 2)),
        )
        self._append_patterns(
            result, seen, quotas[0], state, hotspot, macro_round, "B", "yield",
            yield_patterns, message.message_hash, message.intervention,
            top_pressure_ids, top_pressure_values,
        )

        retreat_order = tuple(yield_ranked[1:]) + tuple(yield_ranked[:1])
        retreat_patterns = itertools.chain(
            (((), combo) for combo in itertools.combinations(retreat_order[:16], 2)),
            (((), combo) for combo in itertools.combinations(retreat_order[:12], 3)),
            (((), combo) for combo in itertools.combinations(retreat_order[:16], 1)),
        )
        self._append_patterns(
            result, seen, quotas[1], state, hotspot, macro_round, "B", "retreat",
            retreat_patterns, message.message_hash, message.intervention,
            top_pressure_ids, top_pressure_values,
        )

        replace_patterns = itertools.chain(
            (
                ((add_id,), (remove_id,))
                for remove_id in yield_ranked[:16]
                for add_id in add_ranked[:16]
            ),
            (
                (tuple(add_pair), (remove_id,))
                for remove_id in yield_ranked[:8]
                for add_pair in itertools.combinations(add_ranked[:12], 2)
            ),
        )
        self._append_patterns(
            result, seen, quotas[2], state, hotspot, macro_round, "B", "replace",
            replace_patterns, message.message_hash, message.intervention,
            top_pressure_ids, top_pressure_values,
        )

        explore_patterns = itertools.chain(
            (
                ((node,), ()) if node not in selected else ((), (node,))
                for node in explore_ranked[:24]
            ),
            (
                (
                    tuple(node for node in pair if node not in selected),
                    tuple(node for node in pair if node in selected),
                )
                for pair in itertools.combinations(explore_ranked[:16], 2)
            ),
        )
        self._append_patterns(
            result, seen, quotas[3], state, hotspot, macro_round, "B", "explore",
            explore_patterns, message.message_hash, message.intervention,
            top_pressure_ids, top_pressure_values,
        )

        self._fallback_fill(
            result, seen, count, state, hotspot, macro_round, "B", "explore",
            message.message_hash, message.intervention, top_pressure_ids,
            top_pressure_values,
        )
        return result[:count]


def generate_wave_a_intents(
    state: BoundaryState,
    experiment_seed: int,
    macro_round: int,
    count: int = WAVE_A_DEFAULT_COUNT,
    hotspot_size: int = H_DYNAMIC_DEFAULT,
) -> List[ProposalIntent]:
    return ProposalEngine(
        state.agent_id, experiment_seed, hotspot_size=hotspot_size
    ).wave_a(state, macro_round=macro_round, count=count)


def generate_wave_b_intents(
    state: BoundaryState,
    real_message: BoundaryMessage,
    experiment_seed: int,
    macro_round: int,
    intervention: Any = MessageIntervention.REAL,
    count: int = WAVE_B_DEFAULT_COUNT,
    hotspot_size: int = H_DYNAMIC_DEFAULT,
) -> List[ProposalIntent]:
    return ProposalEngine(
        state.agent_id, experiment_seed, hotspot_size=hotspot_size
    ).wave_b(
        state,
        real_message=real_message,
        macro_round=macro_round,
        intervention=intervention,
        count=count,
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
