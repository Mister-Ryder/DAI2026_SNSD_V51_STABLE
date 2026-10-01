"""Cross-fitted rolling communication teacher for SNSD v1.5.

The v3 teacher is a sound one-snapshot warm start, but every example is
conditioned on the same warm-up incumbent and the same exact message.  The
v4.4 runner instead consumes a message produced by the preceding factor pass
and anchors the next proposal at that pass' joint selection.  This module
implements that temporal contract without changing the v3 artifacts.

For each seed and held-out fold, complementary-fold candidates plus the public
incumbent/safe actions form the initially revealed domain.  At each of six
query steps all still-held-out candidates receive counterfactual exact labels.
A caller then reveals at most one candidate per agent.  Exact chain Max-Sum is
run on the enlarged revealed domain, its joint optimum becomes the next
rolling anchor, and its pressure message is consumed by the next observation.

The leakage invariant is sequential: a candidate's worker objective is absent
from the message used to score that candidate.  Once a candidate has been
selected and revealed, its objective may legitimately affect later messages,
but that candidate is removed from the target pool and is never scored again.
Unidentifiable pressure retains the v3 value=0, uncertainty=1 transport rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from snsd_v51.factor_graph import CandidateAction, solve_chain_max_sum
from snsd_v51.messages import BoundaryMessage

from .message_teacher import _message_from_source, _stable_fold
from .offline_dataset import (
    ArcPressureLabel,
    CandidateTargets,
    OfflineDataError,
    OfflineDataset,
    OfflineExample,
    _pressure_labels,
    cut_edges_by_link,
)


ROLLING_TEACHER_SCHEMA = "SNSD-v1.5-crossfit-rolling-message-teacher-v4.5"


def _stable_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _candidate_action(item: OfflineExample, incumbent_hash: str) -> CandidateAction:
    return CandidateAction(
        agent_id=int(item.prequery.agent_id),
        candidate_id=str(item.prequery.signature_hash),
        local_objective=float(item.targets.local_objective),
        selected_boundary=frozenset(item.prequery.selected_boundary),
        is_incumbent=str(item.prequery.signature_hash) == str(incumbent_hash),
        is_safe=bool(item.prequery.is_safe_anchor),
        is_feasible=str(item.targets.solver_status) == "feasible",
    )


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


@dataclass(frozen=True)
class RollingTeacherObservation:
    """One pre-query state in a rolling cross-fitted trajectory."""

    seed: int
    heldout_fold: int
    step: int
    target_examples: Tuple[OfflineExample, ...]
    messages: Mapping[int, BoundaryMessage]
    arc_pressures: Tuple[ArcPressureLabel, ...]
    anchor_signature_hashes: Mapping[int, str]
    anchor_selected_boundary: Mapping[int, Tuple[int, ...]]
    receipt: Mapping[str, Any]

    @property
    def key(self) -> Tuple[int, int, int]:
        return (self.seed, self.heldout_fold, self.step)

    @property
    def consumes_message(self) -> bool:
        return bool(self.messages)


class CrossfitRollingEnvironment(object):
    """Finite offline environment with online-style reveal/anchor transitions."""

    def __init__(
        self,
        offline: OfflineDataset,
        seed: int,
        heldout_fold: int,
        folds: int = 2,
        negotiation_steps: int = 6,
    ) -> None:
        self.offline = offline
        self.seed = int(seed)
        self.heldout_fold = int(heldout_fold)
        self.folds = int(folds)
        self.negotiation_steps = int(negotiation_steps)
        if self.folds < 2:
            raise OfflineDataError("rolling message cross-fitting requires at least two folds")
        if not 0 <= self.heldout_fold < self.folds:
            raise OfflineDataError("heldout_fold is outside the configured fold range")
        if self.negotiation_steps < 1:
            raise OfflineDataError("negotiation_steps must be positive")
        if offline.partitions is None:
            raise OfflineDataError("rolling teacher requires reconstructed partitions")
        self.links = cut_edges_by_link(offline.partitions)
        self.num_agents = int(offline.partitions.k)
        self._source = {}  # type: Dict[int, List[OfflineExample]]
        self._pool = {}  # type: Dict[int, Dict[str, OfflineExample]]
        self._anchors = {}  # type: Dict[int, str]
        self._messages = {}  # type: Dict[int, BoundaryMessage]
        self._pressures = tuple()  # type: Tuple[ArcPressureLabel, ...]
        self._transitions = []  # type: List[Mapping[str, Any]]
        self.step = 0

        for agent_id in range(self.num_agents):
            rows = tuple(
                sorted(
                    (
                        item
                        for item in offline.examples
                        if int(item.prequery.seed) == self.seed
                        and int(item.prequery.agent_id) == agent_id
                    ),
                    key=lambda item: item.prequery.signature_hash,
                )
            )
            if not rows:
                raise OfflineDataError(
                    "seed %d agent %d has an empty rolling table"
                    % (self.seed, agent_id)
                )
            anchor_hashes = {
                str(item.prequery.anchor_signature_hash) for item in rows
            }
            if len(anchor_hashes) != 1:
                raise OfflineDataError("rolling table has inconsistent public anchors")
            anchor_hash = next(iter(anchor_hashes))
            source = []
            target = []
            for item in rows:
                protected = (
                    str(item.prequery.signature_hash) == anchor_hash
                    or bool(item.prequery.is_safe_anchor)
                )
                if protected or _stable_fold(item.prequery.signature_hash, self.folds) != self.heldout_fold:
                    source.append(item)
                else:
                    target.append(item)
            if not any(
                str(item.prequery.signature_hash) == anchor_hash for item in source
            ):
                raise OfflineDataError("rolling source has no incumbent anchor")
            if not any(bool(item.prequery.is_safe_anchor) for item in source):
                raise OfflineDataError("rolling source has no safe anchor")
            self._source[agent_id] = list(source)
            self._pool[agent_id] = {
                str(item.prequery.signature_hash): item for item in target
            }
            self._anchors[agent_id] = anchor_hash

        # The public anchor came from the full warm-up factor.  Because it is
        # retained while candidates are removed, it remains a feasible tuple.
        self._solve_source()

    @property
    def anchor_signature_hashes(self) -> Dict[int, str]:
        return dict(self._anchors)

    @property
    def transitions(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self._transitions)

    def _source_tables(self) -> Tuple[Tuple[CandidateAction, ...], ...]:
        return tuple(
            tuple(
                _candidate_action(item, self._anchors[agent_id])
                for item in self._source[agent_id]
            )
            for agent_id in range(self.num_agents)
        )

    def _solve_source(self):
        return solve_chain_max_sum(
            self._source_tables(),
            self.links,
            require_incumbent=True,
            require_safe_anchor=True,
        )

    def _anchor_boundaries(self) -> Dict[int, Tuple[int, ...]]:
        result = {}
        for agent_id in range(self.num_agents):
            matches = [
                item
                for item in self._source[agent_id]
                if str(item.prequery.signature_hash) == self._anchors[agent_id]
            ]
            if len(matches) != 1:
                raise OfflineDataError("rolling anchor is not unique in source domain")
            result[agent_id] = tuple(matches[0].prequery.selected_boundary)
        return result

    def _relabel_targets(self) -> Tuple[OfflineExample, ...]:
        base_tables = [list(table) for table in self._source_tables()]
        anchors = self._anchor_boundaries()
        output = []
        for target_agent in range(self.num_agents):
            targets = tuple(
                sorted(
                    self._pool[target_agent].values(),
                    key=lambda item: item.prequery.signature_hash,
                )
            )
            if not targets:
                continue
            expanded = [list(table) for table in base_tables]
            existing = {item.candidate_id for item in expanded[target_agent]}
            for item in targets:
                if item.prequery.signature_hash in existing:
                    raise OfflineDataError("unrevealed rolling target leaked into source")
                expanded[target_agent].append(
                    _candidate_action(item, self._anchors[target_agent])
                )
            factor = solve_chain_max_sum(
                expanded,
                self.links,
                require_incumbent=True,
                require_safe_anchor=True,
            )
            index_by_id = {
                action.candidate_id: index
                for index, action in enumerate(expanded[target_agent])
            }
            for item in targets:
                diagnostics = factor.diagnostics(
                    target_agent, index_by_id[item.prequery.signature_hash]
                )
                advantage = _finite_or_none(diagnostics.best_completion_advantage)
                externality = _finite_or_none(diagnostics.externality)
                forced = _finite_or_none(diagnostics.forced_best_completion)
                local_gain = _finite_or_none(diagnostics.local_gain)
                prequery = replace(
                    item.prequery,
                    round_id=self.step,
                    anchor_signature_hash=self._anchors[target_agent],
                    anchor_selected_boundary=anchors[target_agent],
                    source="rolling_crossfit_prequery_v4_5",
                    is_safe_anchor=False,
                )
                targets_row = CandidateTargets(
                    solver_status=item.targets.solver_status,
                    local_objective=item.targets.local_objective,
                    local_gain=local_gain,
                    local_gain_mask=(
                        str(item.targets.solver_status) == "feasible"
                        and local_gain is not None
                    ),
                    forced_best_completion=forced,
                    best_completion_advantage=advantage,
                    advantage_mask=advantage is not None,
                    externality=externality,
                    externality_mask=externality is not None,
                    left_opportunity_cost=_finite_or_none(
                        diagnostics.left_opportunity_cost
                    ),
                    right_opportunity_cost=_finite_or_none(
                        diagnostics.right_opportunity_cost
                    ),
                    opportunity_cost=_finite_or_none(diagnostics.opportunity_cost),
                    completion_feasible=bool(diagnostics.completion_feasible),
                    completion_mask=True,
                    uncertainty_mask=advantage is not None,
                    teacher_source="rolling_crossfit_exact_chain_v4_5",
                )
                output.append(OfflineExample(prequery=prequery, targets=targets_row))
        return tuple(output)

    def observe(self) -> RollingTeacherObservation:
        if self.step >= self.negotiation_steps:
            raise OfflineDataError("rolling trajectory is already complete")
        targets = self._relabel_targets()
        source_ids = {
            str(agent_id): sorted(
                str(item.prequery.signature_hash)
                for item in self._source[agent_id]
            )
            for agent_id in range(self.num_agents)
        }
        target_ids = {
            str(agent_id): sorted(self._pool[agent_id])
            for agent_id in range(self.num_agents)
        }
        disjoint = all(
            set(source_ids[str(agent_id)]).isdisjoint(target_ids[str(agent_id)])
            for agent_id in range(self.num_agents)
        )
        if not disjoint:
            raise OfflineDataError("rolling source and target domains overlap")
        anchors = self._anchor_boundaries()
        receipt = {
            "schema_version": ROLLING_TEACHER_SCHEMA,
            "seed": self.seed,
            "heldout_fold": self.heldout_fold,
            "step": self.step,
            "consumes_message_from_previous_factor": bool(self._messages),
            "candidate_objective_in_own_message": False,
            "source_target_disjoint": True,
            "source_signature_hash": _stable_digest(source_ids),
            "target_signature_hash": _stable_digest(target_ids),
            "source_candidate_counts": {
                key: len(value) for key, value in source_ids.items()
            },
            "target_candidate_counts": {
                key: len(value) for key, value in target_ids.items()
            },
            "anchor_signature_hashes": {
                str(key): value for key, value in self._anchors.items()
            },
            "anchor_hash": _stable_digest(
                {str(key): value for key, value in self._anchors.items()}
            ),
            "message_hashes": {
                str(key): value.message_hash for key, value in self._messages.items()
            },
            "pressure_total": len(self._pressures),
            "pressure_identifiable": sum(
                item.pressure_identifiable for item in self._pressures
            ),
            "target_count": len(targets),
        }
        return RollingTeacherObservation(
            seed=self.seed,
            heldout_fold=self.heldout_fold,
            step=self.step,
            target_examples=targets,
            messages=dict(self._messages),
            arc_pressures=tuple(self._pressures),
            anchor_signature_hashes=dict(self._anchors),
            anchor_selected_boundary=anchors,
            receipt=receipt,
        )

    def advance(self, selected_by_agent: Mapping[int, Optional[str]]) -> Mapping[str, Any]:
        """Reveal selected candidates and construct the next rolling state."""

        if self.step >= self.negotiation_steps:
            raise OfflineDataError("cannot advance a completed rolling trajectory")
        before = dict(self._anchors)
        revealed = {}
        for agent_id in range(self.num_agents):
            candidate_id = selected_by_agent.get(agent_id)
            if candidate_id is None:
                continue
            candidate_id = str(candidate_id)
            if candidate_id not in self._pool[agent_id]:
                raise OfflineDataError(
                    "agent %d selected a candidate outside its unrevealed pool"
                    % agent_id
                )
            item = self._pool[agent_id].pop(candidate_id)
            self._source[agent_id].append(item)
            self._source[agent_id].sort(key=lambda row: row.prequery.signature_hash)
            revealed[agent_id] = candidate_id

        factor = self._solve_source()
        self._anchors = {
            agent_id: str(factor.best_actions[agent_id].candidate_id)
            for agent_id in range(self.num_agents)
        }
        anchors = self._anchor_boundaries()
        pressures = []  # type: List[ArcPressureLabel]
        messages = {}
        tables = self._source_tables()
        for agent_id in range(self.num_agents):
            source = tuple(self._source[agent_id])
            signatures = tuple(
                SimpleNamespace(selected_ids=item.prequery.selected_boundary)
                for item in source
            )
            labels = _pressure_labels(
                seed=self.seed,
                agent_id=agent_id,
                boundary_order=source[0].prequery.boundary_order,
                boundary_order_hash=source[0].prequery.boundary_order_hash,
                incumbent_selected=set(anchors[agent_id]),
                signatures=signatures,
                forced_values=factor.forced_best_completions[agent_id],
            )
            labels = tuple(
                replace(
                    item,
                    round_id=self.step,
                    teacher_source="rolling_crossfit_exact_pressure_v4_5",
                )
                for item in labels
            )
            pressures.extend(labels)
            rolling_source = tuple(
                replace(
                    item,
                    prequery=replace(
                        item.prequery,
                        round_id=self.step,
                        anchor_signature_hash=self._anchors[agent_id],
                        anchor_selected_boundary=anchors[agent_id],
                        source="rolling_crossfit_revealed_v4_5",
                    ),
                )
                for item in source
            )
            messages[agent_id] = _message_from_source(
                agent_id, rolling_source, labels
            )
        self._messages = messages
        self._pressures = tuple(pressures)
        transition = {
            "step": self.step,
            "revealed_signature_hashes": {
                str(key): value for key, value in revealed.items()
            },
            "anchor_before": {str(key): value for key, value in before.items()},
            "anchor_after": {
                str(key): value for key, value in self._anchors.items()
            },
            "changed_agents": [
                agent_id
                for agent_id in range(self.num_agents)
                if before[agent_id] != self._anchors[agent_id]
            ],
            "factor_value": float(factor.best_value),
            "message_hashes": {
                str(key): value.message_hash for key, value in messages.items()
            },
            "pressure_identifiable": sum(
                item.pressure_identifiable for item in pressures
            ),
        }
        self._transitions.append(transition)
        self.step += 1
        return transition


def _exact_greedy_selection(
    observation: RollingTeacherObservation, num_agents: int
) -> Dict[int, Optional[str]]:
    grouped = {}  # type: Dict[int, List[OfflineExample]]
    for item in observation.target_examples:
        grouped.setdefault(int(item.prequery.agent_id), []).append(item)
    selected = {}
    for agent_id in range(num_agents):
        rows = grouped.get(agent_id, [])
        if not rows:
            selected[agent_id] = None
            continue
        selected[agent_id] = min(
            rows,
            key=lambda item: (
                -float(
                    item.targets.best_completion_advantage
                    if item.targets.best_completion_advantage is not None
                    else float("-inf")
                ),
                item.prequery.signature_hash,
            ),
        ).prequery.signature_hash
    return selected


def build_crossfit_rolling_trajectories(
    offline: OfflineDataset,
    folds: int = 2,
    negotiation_steps: int = 6,
) -> Tuple[RollingTeacherObservation, ...]:
    """Materialize deterministic exact-greedy trajectories for audit/tests.

    Training uses the same environment with graph_layers-selected actions, so this
    helper is deliberately labelled exact-greedy rather than on-policy.
    """

    output = []
    for seed in offline.seeds:
        for heldout_fold in range(int(folds)):
            environment = CrossfitRollingEnvironment(
                offline=offline,
                seed=seed,
                heldout_fold=heldout_fold,
                folds=folds,
                negotiation_steps=negotiation_steps,
            )
            for _ in range(int(negotiation_steps)):
                observation = environment.observe()
                output.append(observation)
                environment.advance(
                    _exact_greedy_selection(observation, environment.num_agents)
                )
    return tuple(output)


def write_rolling_teacher_manifest(
    observations: Sequence[RollingTeacherObservation], path: str
) -> None:
    payload = {
        "schema_version": ROLLING_TEACHER_SCHEMA,
        "trajectory_policy": "exact-greedy-audit-only",
        "candidate_objective_in_own_message": False,
        "steps": [dict(item.receipt) for item in observations],
        "observation_count": len(observations),
        "consumed_message_observations": sum(
            item.consumes_message for item in observations
        ),
        "rolling_anchor_observations": sum(
            int(item.step > 0) for item in observations
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
    "ROLLING_TEACHER_SCHEMA",
    "CrossfitRollingEnvironment",
    "RollingTeacherObservation",
    "build_crossfit_rolling_trajectories",
    "write_rolling_teacher_manifest",
]
