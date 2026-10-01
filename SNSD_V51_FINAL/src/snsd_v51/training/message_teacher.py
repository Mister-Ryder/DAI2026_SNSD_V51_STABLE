"""Cross-fitted exact-factor messages for leakage-safe Wave-B warm-starting.

Each warm-up candidate is assigned to one deterministic held-out fold.  Exact
messages and arc pressures are computed only from the other fold plus the
public incumbent/safe anchors.  The held-out candidates are then inserted into
one agent table at a time to obtain forced best-completion labels against that
fixed source domain.  Consequently, a candidate's own worker objective never
contributes to the message used to score it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from snsd_v51.factor_graph import CandidateAction, solve_chain_max_sum
from snsd_v51.messages import BoundaryMessage

from .offline_dataset import (
    ArcPressureLabel,
    CandidateTargets,
    OfflineDataError,
    OfflineDataset,
    OfflineExample,
    _pressure_labels,
    cut_edges_by_link,
)


def _stable_fold(signature_hash: str, folds: int) -> int:
    digest = hashlib.sha256(("message-crossfit-v3|" + signature_hash).encode("ascii")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False) % int(folds)


def _action(item: OfflineExample, incumbent_hash: str) -> CandidateAction:
    return CandidateAction(
        agent_id=item.prequery.agent_id,
        candidate_id=item.prequery.signature_hash,
        local_objective=float(item.targets.local_objective),
        selected_boundary=frozenset(item.prequery.selected_boundary),
        is_incumbent=item.prequery.signature_hash == incumbent_hash,
        is_safe=bool(item.prequery.is_safe_anchor),
        is_feasible=item.targets.solver_status == "feasible",
    )


@dataclass(frozen=True)
class MessageTeacherFold:
    seed: int
    heldout_fold: int
    target_examples: Tuple[OfflineExample, ...]
    messages: Mapping[int, BoundaryMessage]
    arc_pressures: Tuple[ArcPressureLabel, ...]
    receipt: Mapping[str, Any]

    @property
    def key(self) -> Tuple[int, int]:
        return (self.seed, self.heldout_fold)


def _message_from_source(
    agent_id: int,
    source_examples: Sequence[OfflineExample],
    pressures: Sequence[ArcPressureLabel],
) -> BoundaryMessage:
    if not source_examples:
        raise OfflineDataError("message source table is empty")
    order = tuple(source_examples[0].prequery.boundary_order)
    incumbent = set(source_examples[0].prequery.anchor_selected_boundary)
    pressure_by_node = {item.node_id: item for item in pressures}
    selected_by_candidate = [set(item.prequery.selected_boundary) for item in source_examples]
    local_values = [float(item.targets.local_objective) for item in source_examples]
    pressure_map = {}
    opportunity_map = {}
    recovery_map = {}
    conflict_map = {}
    uncertainty_map = {}
    identifiable_count = 0
    for node_id in order:
        full = pressure_by_node[int(node_id)]
        local_on = [
            value
            for selected, value in zip(selected_by_candidate, local_values)
            if node_id in selected and math.isfinite(value)
        ]
        local_off = [
            value
            for selected, value in zip(selected_by_candidate, local_values)
            if node_id not in selected and math.isfinite(value)
        ]
        identifiable = (
            full.pressure_identifiable
            and full.pressure is not None
            and bool(local_on)
            and bool(local_off)
        )
        if identifiable:
            identifiable_count += 1
            full_pressure = float(full.pressure)
            local_pressure = max(local_on) - max(local_off)
            externality_pressure = full_pressure - local_pressure
            pressure_map[node_id] = full_pressure
            opportunity_map[node_id] = abs(externality_pressure)
            if node_id in incumbent:
                recovery_map[node_id] = max(0.0, -externality_pressure)
            else:
                recovery_map[node_id] = max(0.0, externality_pressure)
            conflict_map[node_id] = abs(externality_pressure)
            uncertainty_map[node_id] = 0.0
        else:
            # Missing on/off support is unknown, not observed zero.  The numeric
            # transport value is zero only because BoundaryMessage requires a
            # finite aligned vector; uncertainty=1 carries the missing mask.
            pressure_map[node_id] = 0.0
            opportunity_map[node_id] = 0.0
            recovery_map[node_id] = 0.0
            conflict_map[node_id] = 0.0
            uncertainty_map[node_id] = 1.0
    if identifiable_count != sum(item.pressure_identifiable for item in pressures):
        raise OfflineDataError("full/local pressure identifiability masks diverged")
    return BoundaryMessage.from_maps(
        boundary_order=order,
        opportunity_cost=opportunity_map,
        pressure=pressure_map,
        neighbor_recovery=recovery_map,
        conflict_score=conflict_map,
        uncertainty=uncertainty_map,
    )


def build_crossfit_message_folds(
    offline: OfflineDataset, folds: int = 2
) -> Tuple[MessageTeacherFold, ...]:
    """Build paired-message source states and held-out exact candidate labels."""

    folds = int(folds)
    if folds < 2:
        raise OfflineDataError("message cross-fitting requires at least two folds")
    links = cut_edges_by_link(offline.partitions)
    output = []
    for seed in offline.seeds:
        by_agent = {
            agent_id: tuple(
                sorted(
                    (
                        item
                        for item in offline.examples
                        if item.prequery.seed == seed
                        and item.prequery.agent_id == agent_id
                    ),
                    key=lambda item: item.prequery.signature_hash,
                )
            )
            for agent_id in range(int(offline.partitions.k))
        }
        for agent_id, rows in by_agent.items():
            if not rows:
                raise OfflineDataError("seed %d agent %d has an empty table" % (seed, agent_id))
            anchors = [
                item
                for item in rows
                if item.prequery.signature_hash == item.prequery.anchor_signature_hash
                or item.prequery.is_safe_anchor
            ]
            if not any(
                item.prequery.signature_hash == item.prequery.anchor_signature_hash
                for item in anchors
            ):
                raise OfflineDataError("crossfit source has no incumbent")
            if not any(item.prequery.is_safe_anchor for item in anchors):
                raise OfflineDataError("crossfit source has no safe anchor")
        for heldout_fold in range(folds):
            source_by_agent = {}
            target_by_agent = {}
            action_tables = []
            for agent_id, rows in by_agent.items():
                source = []
                target = []
                for item in rows:
                    anchor = (
                        item.prequery.signature_hash
                        == item.prequery.anchor_signature_hash
                        or item.prequery.is_safe_anchor
                    )
                    if anchor or _stable_fold(item.prequery.signature_hash, folds) != heldout_fold:
                        source.append(item)
                    else:
                        target.append(item)
                if not target:
                    raise OfflineDataError(
                        "seed %d fold %d agent %d has no held-out candidates"
                        % (seed, heldout_fold, agent_id)
                    )
                source_by_agent[agent_id] = tuple(source)
                target_by_agent[agent_id] = tuple(target)
                incumbent_hash = rows[0].prequery.anchor_signature_hash
                action_tables.append(
                    [_action(item, incumbent_hash) for item in source]
                )
            source_factor = solve_chain_max_sum(
                action_tables,
                links,
                require_incumbent=True,
                require_safe_anchor=True,
            )
            pressure_rows = []
            messages = {}
            for agent_id in range(int(offline.partitions.k)):
                source = source_by_agent[agent_id]
                signatures = []
                for item in source:
                    signatures.append(
                        type("SignatureView", (), {"selected_ids": item.prequery.selected_boundary})()
                    )
                agent_pressures = _pressure_labels(
                    seed=seed,
                    agent_id=agent_id,
                    boundary_order=source[0].prequery.boundary_order,
                    boundary_order_hash=source[0].prequery.boundary_order_hash,
                    incumbent_selected=set(source[0].prequery.anchor_selected_boundary),
                    signatures=signatures,
                    forced_values=source_factor.forced_best_completions[agent_id],
                )
                pressure_rows.extend(agent_pressures)
                messages[agent_id] = _message_from_source(
                    agent_id, source, agent_pressures
                )
            heldout_examples = []
            for target_agent in range(int(offline.partitions.k)):
                expanded_tables = [list(table) for table in action_tables]
                incumbent_hash = by_agent[target_agent][0].prequery.anchor_signature_hash
                existing = {item.candidate_id for item in expanded_tables[target_agent]}
                targets = target_by_agent[target_agent]
                for item in targets:
                    if item.prequery.signature_hash in existing:
                        raise OfflineDataError("held-out candidate leaked into source table")
                    expanded_tables[target_agent].append(_action(item, incumbent_hash))
                factor = solve_chain_max_sum(
                    expanded_tables,
                    links,
                    require_incumbent=True,
                    require_safe_anchor=True,
                )
                index_by_id = {
                    action.candidate_id: index
                    for index, action in enumerate(expanded_tables[target_agent])
                }
                for item in targets:
                    diagnostics = factor.diagnostics(
                        target_agent, index_by_id[item.prequery.signature_hash]
                    )
                    finite_advantage = math.isfinite(
                        diagnostics.best_completion_advantage
                    )
                    finite_externality = math.isfinite(diagnostics.externality)
                    finite_forced = math.isfinite(
                        diagnostics.forced_best_completion
                    )
                    target = CandidateTargets(
                        solver_status=item.targets.solver_status,
                        local_objective=item.targets.local_objective,
                        local_gain=(
                            diagnostics.local_gain
                            if math.isfinite(diagnostics.local_gain)
                            else None
                        ),
                        local_gain_mask=(
                            item.targets.solver_status == "feasible"
                            and math.isfinite(diagnostics.local_gain)
                        ),
                        forced_best_completion=(
                            diagnostics.forced_best_completion
                            if finite_forced
                            else None
                        ),
                        best_completion_advantage=(
                            diagnostics.best_completion_advantage
                            if finite_advantage
                            else None
                        ),
                        advantage_mask=finite_advantage,
                        externality=(
                            diagnostics.externality if finite_externality else None
                        ),
                        externality_mask=finite_externality,
                        left_opportunity_cost=(
                            diagnostics.left_opportunity_cost
                            if math.isfinite(diagnostics.left_opportunity_cost)
                            else None
                        ),
                        right_opportunity_cost=(
                            diagnostics.right_opportunity_cost
                            if math.isfinite(diagnostics.right_opportunity_cost)
                            else None
                        ),
                        opportunity_cost=(
                            diagnostics.opportunity_cost
                            if math.isfinite(diagnostics.opportunity_cost)
                            else None
                        ),
                        completion_feasible=diagnostics.completion_feasible,
                        completion_mask=True,
                        uncertainty_mask=finite_advantage,
                        teacher_source="crossfit_exact_chain_message_v3",
                    )
                    heldout_examples.append(
                        replace(item, targets=target)
                    )
            source_ids = {
                str(agent_id): [
                    item.prequery.signature_hash for item in source_by_agent[agent_id]
                ]
                for agent_id in source_by_agent
            }
            target_ids = {
                str(agent_id): [
                    item.prequery.signature_hash for item in target_by_agent[agent_id]
                ]
                for agent_id in target_by_agent
            }
            if any(set(source_ids[key]).intersection(target_ids[key]) for key in source_ids):
                raise OfflineDataError("crossfit source/target signatures overlap")
            receipt = {
                "seed": seed,
                "heldout_fold": heldout_fold,
                "fold_count": folds,
                "message_source_candidate_counts": {
                    key: len(value) for key, value in source_ids.items()
                },
                "target_candidate_counts": {
                    key: len(value) for key, value in target_ids.items()
                },
                "source_signature_hash": hashlib.sha256(
                    json.dumps(source_ids, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "target_signature_hash": hashlib.sha256(
                    json.dumps(target_ids, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "candidate_objective_in_own_message": False,
                "target_count": len(heldout_examples),
                "pressure_total": len(pressure_rows),
                "pressure_identifiable": sum(
                    item.pressure_identifiable for item in pressure_rows
                ),
                "message_hashes": {
                    str(agent_id): message.message_hash
                    for agent_id, message in messages.items()
                },
            }
            output.append(
                MessageTeacherFold(
                    seed=seed,
                    heldout_fold=heldout_fold,
                    target_examples=tuple(heldout_examples),
                    messages=messages,
                    arc_pressures=tuple(pressure_rows),
                    receipt=receipt,
                )
            )
    expected = {
        (item.prequery.seed, item.prequery.agent_id, item.prequery.signature_hash)
        for item in offline.examples
        if not (
            item.prequery.signature_hash == item.prequery.anchor_signature_hash
            or item.prequery.is_safe_anchor
        )
    }
    observed = {
        (item.prequery.seed, item.prequery.agent_id, item.prequery.signature_hash)
        for fold in output
        for item in fold.target_examples
    }
    if expected != observed:
        raise OfflineDataError(
            "crossfit targets do not cover every non-anchor candidate exactly once"
        )
    return tuple(output)


def zero_target_examples(
    examples: Sequence[OfflineExample], source: str = "paired_zero_message_v3"
) -> Tuple[OfflineExample, ...]:
    """Use local gain as the zero-message acquisition target."""

    result = []
    for item in examples:
        target = replace(
            item.targets,
            best_completion_advantage=item.targets.local_gain,
            advantage_mask=item.targets.local_gain_mask,
            externality=0.0 if item.targets.local_gain_mask else None,
            externality_mask=item.targets.local_gain_mask,
            uncertainty_mask=item.targets.local_gain_mask,
            teacher_source=source,
        )
        result.append(replace(item, targets=target))
    return tuple(result)


def write_crossfit_manifest(folds: Sequence[MessageTeacherFold], path: str) -> None:
    payload = {
        "schema_version": 3,
        "teacher": "crossfit_exact_chain_message_v3",
        "candidate_objective_in_own_message": False,
        "folds": [dict(item.receipt) for item in folds],
        "total_target_candidates": sum(len(item.target_examples) for item in folds),
        "total_identifiable_pressures": sum(
            label.pressure_identifiable for item in folds for label in item.arc_pressures
        ),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(destination))


__all__ = [
    "MessageTeacherFold",
    "build_crossfit_message_folds",
    "write_crossfit_manifest",
    "zero_target_examples",
]
