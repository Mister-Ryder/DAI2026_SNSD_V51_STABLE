"""Strict depth-conditioned cross-fitted rolling teacher for SNSD v1.5.

This additive module closes an audit gap in :mod:`rolling_teacher`.
Merely calling ``observe`` repeatedly is not a deep acquisition trajectory
when a held-out pool has already become empty.  Here candidate folds are
balanced *within every (seed, agent)* table and construction fails unless each
agent can reveal a distinct held-out candidate at every one of the requested
steps.

The sequential leakage contract is unchanged: the objective of an unrevealed
candidate is absent from the factor message used to score it.  Once selected,
the candidate is moved into the revealed/source domain and may affect only
later observations.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .rolling_teacher import (
    CrossfitRollingEnvironment,
    RollingTeacherObservation,
    _exact_greedy_selection,
    _stable_digest,
)
from .offline_dataset import (
    ArcPressureLabel,
    OfflineDataError,
    OfflineDataset,
    OfflineExample,
    cut_edges_by_link,
)


STRICT_ROLLING_TEACHER_SCHEMA = (
    "SNSD-v1.5-crossfit-balanced-deep-transition-message-teacher-v4.5-strict"
)


def _balanced_fold_assignment(
    signature_hashes: Sequence[str], folds: int
) -> Dict[str, int]:
    """Deterministically stratify one agent's candidates across folds.

    Sorting by a domain-separated hash avoids dependence on input row order;
    round-robin assignment makes fold sizes differ by at most one.
    """

    if int(folds) < 2:
        raise OfflineDataError("strict rolling cross-fitting requires >=2 folds")
    unique = tuple(set(str(value) for value in signature_hashes))
    if len(unique) != len(signature_hashes):
        raise OfflineDataError("candidate signatures must be unique per agent")
    ordered = sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(
                ("message-crossfit-v4.5-strict|" + value).encode("ascii")
            ).digest(),
            value,
        ),
    )
    return {value: index % int(folds) for index, value in enumerate(ordered)}


class StrictCrossfitRollingEnvironment(CrossfitRollingEnvironment):
    """Balanced environment with at least six genuine agent transitions."""

    def __init__(
        self,
        offline: OfflineDataset,
        seed: int,
        heldout_fold: int,
        folds: int = 2,
        negotiation_steps: int = 12,
    ) -> None:
        # Do not call the v4.5 constructor: it performs unstratified hash-fold
        # assignment before we could enforce the full-activity invariant.
        self.offline = offline
        self.seed = int(seed)
        self.heldout_fold = int(heldout_fold)
        self.folds = int(folds)
        self.negotiation_steps = int(negotiation_steps)
        if self.folds < 2:
            raise OfflineDataError("strict rolling cross-fitting requires >=2 folds")
        if not 0 <= self.heldout_fold < self.folds:
            raise OfflineDataError("heldout_fold is outside the configured range")
        if self.negotiation_steps < 6:
            raise OfflineDataError("strict rolling training requires at least six steps")
        if offline.partitions is None:
            raise OfflineDataError("strict rolling teacher requires partitions")

        self.links = cut_edges_by_link(offline.partitions)
        self.num_agents = int(offline.partitions.k)
        self._source = {}  # type: Dict[int, List[OfflineExample]]
        self._pool = {}  # type: Dict[int, Dict[str, OfflineExample]]
        self._anchors = {}  # type: Dict[int, str]
        self._messages = {}  # type: Dict[int, Any]
        self._pressures = tuple()  # type: Tuple[ArcPressureLabel, ...]
        self._transitions = []  # type: List[Mapping[str, Any]]
        self._initial_target_counts = {}  # type: Dict[int, int]
        self._fold_assignment_hashes = {}  # type: Dict[int, str]
        self._original_full_anchor_hashes = {}  # type: Dict[int, str]
        self._initial_source_anchor_hashes = {}  # type: Dict[int, str]
        self._initial_source_factor_value = float("nan")
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
                    "seed %d agent %d has an empty strict rolling table"
                    % (self.seed, agent_id)
                )
            anchor_hashes = {
                str(item.prequery.anchor_signature_hash) for item in rows
            }
            if len(anchor_hashes) != 1:
                raise OfflineDataError("rolling table has inconsistent anchors")
            anchor_hash = next(iter(anchor_hashes))
            # Only a certified safe action is protected in every source fold.
            # The original manifest incumbent was optimized over the *entire*
            # warm-up table.  Protecting it would make every subsequent reveal
            # incapable of improving the rolling anchor and turn the unroll
            # into static repetition.
            protected = {
                str(item.prequery.signature_hash)
                for item in rows
                if bool(item.prequery.is_safe_anchor)
            }
            safe_hashes = tuple(sorted(protected))
            if not safe_hashes:
                raise OfflineDataError("strict source has no safe anchor")
            eligible = tuple(
                str(item.prequery.signature_hash)
                for item in rows
                if str(item.prequery.signature_hash) not in protected
            )
            assignment = _balanced_fold_assignment(eligible, self.folds)
            source = [
                item
                for item in rows
                if str(item.prequery.signature_hash) in protected
                or assignment[str(item.prequery.signature_hash)]
                != self.heldout_fold
            ]
            target = [
                item
                for item in rows
                if str(item.prequery.signature_hash) not in protected
                and assignment[str(item.prequery.signature_hash)]
                == self.heldout_fold
            ]
            if len(target) < self.negotiation_steps:
                raise OfflineDataError(
                    "strict rolling pool exhausted before training: seed=%d "
                    "agent=%d fold=%d targets=%d required=%d; materialize more "
                    "distinct warm-up candidates"
                    % (
                        self.seed,
                        agent_id,
                        self.heldout_fold,
                        len(target),
                        self.negotiation_steps,
                    )
                )
            if not any(bool(item.prequery.is_safe_anchor) for item in source):
                raise OfflineDataError("strict source has no safe anchor")
            self._source[agent_id] = list(source)
            self._pool[agent_id] = {
                str(item.prequery.signature_hash): item for item in target
            }
            # A safe tuple supplies the first feasible incumbent flags.  After
            # all agent tables have been assembled, exact Max-Sum promotes the
            # source-only joint optimum to I0.
            self._anchors[agent_id] = safe_hashes[0]
            self._original_full_anchor_hashes[agent_id] = anchor_hash
            self._initial_target_counts[agent_id] = len(target)
            self._fold_assignment_hashes[agent_id] = _stable_digest(assignment)

        source_factor = self._solve_source()
        self._anchors = {
            agent_id: str(source_factor.best_actions[agent_id].candidate_id)
            for agent_id in range(self.num_agents)
        }
        self._initial_source_anchor_hashes = dict(self._anchors)
        self._initial_source_factor_value = float(source_factor.best_value)
        # Revalidate that the promoted tuple is present, jointly feasible, and
        # marked as the unique source incumbent for every table.
        promoted = self._solve_source()
        if tuple(
            str(promoted.best_actions[agent_id].candidate_id)
            for agent_id in range(self.num_agents)
        ) != tuple(
            self._anchors[agent_id] for agent_id in range(self.num_agents)
        ):
            raise OfflineDataError("source-only anchor promotion is not stable")

    @property
    def initial_target_counts(self) -> Dict[int, int]:
        return dict(self._initial_target_counts)

    def observe(self) -> RollingTeacherObservation:
        observation = super().observe()
        active = {
            agent_id: len(self._pool[agent_id])
            for agent_id in range(self.num_agents)
        }
        if any(value < 1 for value in active.values()):
            raise OfflineDataError(
                "strict rolling observation is not active for every agent: %r"
                % active
            )
        receipt = dict(observation.receipt)
        receipt.update(
            {
                "schema_version": STRICT_ROLLING_TEACHER_SCHEMA,
                "balanced_crossfit": True,
                "full_agent_activity": True,
                "remaining_target_counts": {
                    str(key): value for key, value in active.items()
                },
                "initial_target_counts": {
                    str(key): value
                    for key, value in self._initial_target_counts.items()
                },
                "fold_assignment_hashes": {
                    str(key): value
                    for key, value in self._fold_assignment_hashes.items()
                },
                "anchor_initialization": "exact_source_only_max_sum",
                "original_full_table_anchor_signature_hashes": {
                    str(key): value
                    for key, value in self._original_full_anchor_hashes.items()
                },
                "initial_source_only_anchor_signature_hashes": {
                    str(key): value
                    for key, value in self._initial_source_anchor_hashes.items()
                },
                "initial_source_only_factor_value": self._initial_source_factor_value,
                "heldout_candidates_excluded_from_initial_anchor": True,
            }
        )
        return replace(observation, receipt=receipt)

    def advance(
        self, selected_by_agent: Mapping[int, Optional[str]]
    ) -> Mapping[str, Any]:
        missing = [
            agent_id
            for agent_id in range(self.num_agents)
            if selected_by_agent.get(agent_id) is None
        ]
        if missing:
            raise OfflineDataError(
                "strict rolling transition requires one new candidate per agent; "
                "missing=%r" % missing
            )
        transition = dict(super().advance(selected_by_agent))
        if len(transition["revealed_signature_hashes"]) != self.num_agents:
            raise OfflineDataError("strict rolling transition revealed too few agents")
        transition["full_agent_transition"] = True
        transition["schema_version"] = STRICT_ROLLING_TEACHER_SCHEMA
        # Replace the immutable mapping stored by the parent with the audited row.
        self._transitions[-1] = transition
        return transition


def build_strict_crossfit_rolling_trajectories(
    offline: OfflineDataset,
    folds: int = 2,
    negotiation_steps: int = 12,
) -> Tuple[RollingTeacherObservation, ...]:
    """Materialize deterministic exact-greedy trajectories for audit only."""

    output = []
    for seed in offline.seeds:
        for heldout_fold in range(int(folds)):
            environment = StrictCrossfitRollingEnvironment(
                offline=offline,
                seed=seed,
                heldout_fold=heldout_fold,
                folds=folds,
                negotiation_steps=negotiation_steps,
            )
            for _ in range(int(negotiation_steps)):
                observation = environment.observe()
                transition = environment.advance(
                    _exact_greedy_selection(observation, environment.num_agents)
                )
                receipt = dict(observation.receipt)
                receipt["transition_after_observation"] = dict(transition)
                output.append(replace(observation, receipt=receipt))
    return tuple(output)


def write_strict_rolling_teacher_manifest(
    observations: Sequence[RollingTeacherObservation], path: str
) -> None:
    num_agents = (
        len(observations[0].anchor_signature_hashes) if observations else 0
    )
    per_step = {}
    for step in sorted(set(item.step for item in observations)):
        rows = [item for item in observations if item.step == step]
        transitions = [
            item.receipt.get("transition_after_observation", {}) for item in rows
        ]
        changes = sum(len(row.get("changed_agents", ())) for row in transitions)
        slots = len(rows) * num_agents
        per_step[str(step)] = {
            "observations": len(rows),
            "candidate_samples": sum(len(item.target_examples) for item in rows),
            "minimum_candidates_per_agent": min(
                int(value)
                for item in rows
                for value in item.receipt["remaining_target_counts"].values()
            ),
            "revealed_candidates": sum(
                len(row.get("revealed_signature_hashes", {})) for row in transitions
            ),
            "anchor_change_events": changes,
            "anchor_change_rate": (
                float(changes) / float(slots) if slots else None
            ),
            "full_agent_activity": all(
                len(row.get("revealed_signature_hashes", {})) == num_agents
                for row in transitions
            ),
        }
    payload = {
        "schema_version": STRICT_ROLLING_TEACHER_SCHEMA,
        "trajectory_policy": "balanced-crossfit-exact-greedy-audit-only",
        "candidate_objective_in_own_message": False,
        "fabricated_zero_labels": False,
        "observation_count": len(observations),
        "consumed_message_observations": sum(
            item.consumes_message for item in observations
        ),
        "all_observations_full_agent_activity": all(
            bool(item.receipt.get("full_agent_activity")) for item in observations
        ),
        "all_transitions_full_agent_activity": all(
            len(
                item.receipt.get("transition_after_observation", {}).get(
                    "revealed_signature_hashes", {}
                )
            )
            == num_agents
            for item in observations
        ),
        "per_step": per_step,
        "steps": [dict(item.receipt) for item in observations],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(destination))


__all__ = [
    "STRICT_ROLLING_TEACHER_SCHEMA",
    "StrictCrossfitRollingEnvironment",
    "_balanced_fold_assignment",
    "build_strict_crossfit_rolling_trajectories",
    "write_strict_rolling_teacher_manifest",
]
