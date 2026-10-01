"""Exact candidate-level factor graph for the six-agent C3 chain.

The module deliberately has no dependency on CPLEX or the learned policy.  It
solves the *restricted* problem induced by realized local candidates:

    max sum_i q_i(c_i)
    s.t. adjacent boundary signatures are compatible.

Compatibility is exact with respect to the supplied cut edges.  The dynamic
programming messages exclude the receiving candidate's local objective, which
matches the notation in ``V1_5_METHOD_SPEC.md`` and makes the opportunity-cost
decomposition directly auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Hashable, Iterable, Optional, Sequence, Tuple

import math
import numpy as np


NEG_INF = float("-inf")


class FactorGraphError(ValueError):
    """Base class for invalid candidate factor graphs."""


class InvalidCandidateTableError(FactorGraphError):
    """Raised when candidate-table contents violate the data contract."""


class InvalidIncumbentError(FactorGraphError):
    """Raised when the verified incumbent anchors do not form a feasible tuple."""


class InvalidSafeAnchorError(FactorGraphError):
    """Raised when safe anchors cannot complete every feasible local action."""


class NoFeasibleJointAssignmentError(FactorGraphError):
    """Raised when the restricted candidate domains have no feasible tuple."""


@dataclass(frozen=True)
class CandidateAction:
    """One realized, internally feasible-or-rejected action for one agent.

    Parameters
    ----------
    agent_id:
        Integer agent identifier.  Candidate tables determine chain order; all
        candidates in one table must have the same identifier.
    candidate_id:
        Stable identifier, unique within the agent's current candidate table.
    local_objective:
        True local objective returned by the common worker.  It must be finite;
        infeasible actions are excluded through ``is_feasible`` instead of a
        sentinel objective.
    selected_boundary:
        Globally identified boundary nodes selected by this action.  Only this
        signature participates in adjacent compatibility tests.
    is_incumbent:
        Marks the verified incumbent projection used as the advantage baseline.
    is_safe:
        Marks the zero-boundary (or equivalently universal) safe anchor.
    is_feasible:
        Whether the local worker declared the action internally feasible.
        Infeasible actions remain auditable but receive a forced value of
        negative infinity and can never enter the optimal tuple.
    """

    agent_id: int
    candidate_id: str
    local_objective: float
    selected_boundary: FrozenSet[Hashable] = frozenset()
    is_incumbent: bool = False
    is_safe: bool = False
    is_feasible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, int):
            raise TypeError("agent_id must be an int")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        objective = float(self.local_objective)
        if not math.isfinite(objective):
            raise ValueError("local_objective must be finite")
        try:
            boundary = frozenset(self.selected_boundary)
        except TypeError as exc:
            raise TypeError("selected_boundary must be an iterable of hashable IDs") from exc
        object.__setattr__(self, "local_objective", objective)
        object.__setattr__(self, "selected_boundary", boundary)


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Auditable conflict-benefit quantities for one candidate action."""

    forced_best_completion: float
    best_completion_advantage: float
    local_gain: float
    externality: float
    left_opportunity_cost: float
    right_opportunity_cost: float
    opportunity_cost: float
    completion_feasible: bool


@dataclass(frozen=True)
class ChainMaxSumResult:
    """Exact forward/backward solution and per-candidate diagnostics."""

    candidate_tables: Tuple[Tuple[CandidateAction, ...], ...]
    compatibility_matrices: Tuple[np.ndarray, ...]
    left_messages: Tuple[np.ndarray, ...]
    right_messages: Tuple[np.ndarray, ...]
    left_backpointers: Tuple[np.ndarray, ...]
    right_backpointers: Tuple[np.ndarray, ...]
    forced_best_completions: Tuple[np.ndarray, ...]
    best_completion_advantages: Tuple[np.ndarray, ...]
    local_gains: Tuple[np.ndarray, ...]
    externalities: Tuple[np.ndarray, ...]
    left_opportunity_costs: Tuple[np.ndarray, ...]
    right_opportunity_costs: Tuple[np.ndarray, ...]
    opportunity_costs: Tuple[np.ndarray, ...]
    completion_feasible: Tuple[np.ndarray, ...]
    free_left_branch_values: Tuple[float, ...]
    free_right_branch_values: Tuple[float, ...]
    incumbent_indices: Tuple[int, ...]
    safe_indices: Tuple[Optional[int], ...]
    best_indices: Tuple[int, ...]
    best_actions: Tuple[CandidateAction, ...]
    best_value: float

    def diagnostics(self, agent_position: int, candidate_index: int) -> CandidateDiagnostics:
        """Return scalar diagnostics for one table position and candidate index."""

        return CandidateDiagnostics(
            forced_best_completion=float(
                self.forced_best_completions[agent_position][candidate_index]
            ),
            best_completion_advantage=float(
                self.best_completion_advantages[agent_position][candidate_index]
            ),
            local_gain=float(self.local_gains[agent_position][candidate_index]),
            externality=float(self.externalities[agent_position][candidate_index]),
            left_opportunity_cost=float(
                self.left_opportunity_costs[agent_position][candidate_index]
            ),
            right_opportunity_cost=float(
                self.right_opportunity_costs[agent_position][candidate_index]
            ),
            opportunity_cost=float(
                self.opportunity_costs[agent_position][candidate_index]
            ),
            completion_feasible=bool(
                self.completion_feasible[agent_position][candidate_index]
            ),
        )


def _readonly_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    array.setflags(write=False)
    return array


def _normalize_cut_edges(
    cut_edges: Iterable[Tuple[Hashable, Hashable]]
) -> Tuple[Tuple[Hashable, Hashable], ...]:
    normalized = []
    for edge in cut_edges:
        try:
            left_node, right_node = edge
        except (TypeError, ValueError) as exc:
            raise ValueError("each cut edge must contain exactly two endpoint IDs") from exc
        try:
            hash(left_node)
            hash(right_node)
        except TypeError as exc:
            raise TypeError("cut-edge endpoint IDs must be hashable") from exc
        normalized.append((left_node, right_node))
    return tuple(normalized)


def build_compatibility_matrix(
    left_actions: Sequence[CandidateAction],
    right_actions: Sequence[CandidateAction],
    cut_edges: Iterable[Tuple[Hashable, Hashable]],
) -> np.ndarray:
    """Build the exact compatibility matrix for one oriented chain edge.

    Each cut edge must be supplied as ``(left_node_id, right_node_id)``.  A
    matrix entry is true exactly when both actions are locally feasible and no
    cut edge has both endpoints selected.
    """

    edges = _normalize_cut_edges(cut_edges)
    matrix = np.zeros((len(left_actions), len(right_actions)), dtype=np.bool_)

    # Pre-indexing the cut graph avoids scanning all cut edges for every action
    # pair.  Candidate domains are small, while C3 has thousands of cut edges.
    right_neighbors = {}
    for left_node, right_node in edges:
        right_neighbors.setdefault(left_node, set()).add(right_node)

    forbidden_by_left = []
    for action in left_actions:
        forbidden = set()
        for node_id in action.selected_boundary:
            forbidden.update(right_neighbors.get(node_id, ()))
        forbidden_by_left.append(frozenset(forbidden))

    for left_index, left_action in enumerate(left_actions):
        if not left_action.is_feasible:
            continue
        forbidden = forbidden_by_left[left_index]
        for right_index, right_action in enumerate(right_actions):
            if not right_action.is_feasible:
                continue
            matrix[left_index, right_index] = forbidden.isdisjoint(
                right_action.selected_boundary
            )

    return _readonly_array(matrix)


def build_adjacent_compatibility_matrices(
    candidate_tables: Sequence[Sequence[CandidateAction]],
    cut_edges_by_link: Sequence[Iterable[Tuple[Hashable, Hashable]]],
) -> Tuple[np.ndarray, ...]:
    """Build compatibility matrices for every adjacent pair in chain order."""

    if len(cut_edges_by_link) != max(0, len(candidate_tables) - 1):
        raise ValueError("cut_edges_by_link must contain exactly n_agents - 1 entries")
    return tuple(
        build_compatibility_matrix(candidate_tables[i], candidate_tables[i + 1], edges)
        for i, edges in enumerate(cut_edges_by_link)
    )


def _validate_tables(
    candidate_tables: Sequence[Sequence[CandidateAction]],
    require_incumbent: bool,
) -> Tuple[Tuple[CandidateAction, ...], ...]:
    if not candidate_tables:
        raise InvalidCandidateTableError("at least one candidate table is required")

    normalized = []
    seen_agent_ids = set()
    for position, raw_table in enumerate(candidate_tables):
        table = tuple(raw_table)
        if not table:
            raise InvalidCandidateTableError(
                "candidate table at chain position %d is empty" % position
            )
        agent_id = table[0].agent_id
        if agent_id in seen_agent_ids:
            raise InvalidCandidateTableError("agent IDs must be unique across tables")
        seen_agent_ids.add(agent_id)
        if any(action.agent_id != agent_id for action in table):
            raise InvalidCandidateTableError(
                "all candidates in one table must have the same agent_id"
            )
        candidate_ids = [action.candidate_id for action in table]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise InvalidCandidateTableError(
                "candidate_id values must be unique within an agent table"
            )
        incumbent_count = sum(action.is_incumbent for action in table)
        if require_incumbent and incumbent_count != 1:
            raise InvalidIncumbentError(
                "each table must contain exactly one incumbent candidate"
            )
        if incumbent_count > 1:
            raise InvalidIncumbentError(
                "a table cannot contain more than one incumbent candidate"
            )
        normalized.append(table)
    return tuple(normalized)


def _anchor_indices(
    tables: Tuple[Tuple[CandidateAction, ...], ...],
    flag_name: str,
) -> Tuple[Optional[int], ...]:
    result = []
    for table in tables:
        indices = [index for index, action in enumerate(table) if getattr(action, flag_name)]
        if len(indices) > 1:
            raise InvalidCandidateTableError(
                "each table may contain at most one %s candidate" % flag_name
            )
        result.append(indices[0] if indices else None)
    return tuple(result)


def _validate_incumbent(
    tables: Tuple[Tuple[CandidateAction, ...], ...],
    compatibility: Tuple[np.ndarray, ...],
    incumbent_indices: Tuple[Optional[int], ...],
    require_incumbent: bool,
) -> None:
    if not require_incumbent:
        return
    for position, incumbent_index in enumerate(incumbent_indices):
        if incumbent_index is None:
            raise InvalidIncumbentError("missing incumbent candidate")
        if not tables[position][incumbent_index].is_feasible:
            raise InvalidIncumbentError("incumbent candidates must be locally feasible")
    for link, matrix in enumerate(compatibility):
        left_index = incumbent_indices[link]
        right_index = incumbent_indices[link + 1]
        assert left_index is not None and right_index is not None
        if not matrix[left_index, right_index]:
            raise InvalidIncumbentError(
                "incumbent candidates are incompatible across chain link %d" % link
            )


def _validate_safe_anchors(
    tables: Tuple[Tuple[CandidateAction, ...], ...],
    compatibility: Tuple[np.ndarray, ...],
    safe_indices: Tuple[Optional[int], ...],
    require_safe_anchor: bool,
) -> None:
    if not require_safe_anchor:
        return
    for position, safe_index in enumerate(safe_indices):
        if safe_index is None:
            raise InvalidSafeAnchorError(
                "candidate table at position %d has no safe anchor" % position
            )
        if not tables[position][safe_index].is_feasible:
            raise InvalidSafeAnchorError("safe anchors must be locally feasible")

    # A safe anchor is stronger than merely being a feasible tuple: replacing
    # every other agent by its safe action must complete any feasible action.
    for link, matrix in enumerate(compatibility):
        left_safe = safe_indices[link]
        right_safe = safe_indices[link + 1]
        assert left_safe is not None and right_safe is not None
        for left_index, action in enumerate(tables[link]):
            if action.is_feasible and not matrix[left_index, right_safe]:
                raise InvalidSafeAnchorError(
                    "right safe anchor does not complete candidate %s"
                    % action.candidate_id
                )
        for right_index, action in enumerate(tables[link + 1]):
            if action.is_feasible and not matrix[left_safe, right_index]:
                raise InvalidSafeAnchorError(
                    "left safe anchor does not complete candidate %s"
                    % action.candidate_id
                )


def _masked_argmax(scores: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.max(scores, axis=axis)
    indices = np.argmax(scores, axis=axis).astype(np.int64, copy=False)
    indices = np.asarray(indices)
    indices[~np.isfinite(values)] = -1
    return values, indices


def _opportunity_cost(free_value: float, constrained_values: np.ndarray) -> np.ndarray:
    result = np.empty(constrained_values.shape, dtype=np.float64)
    if np.isneginf(free_value):
        result.fill(np.nan)
        return result
    finite = np.isfinite(constrained_values)
    result[finite] = free_value - constrained_values[finite]
    # Tiny negative values can only be floating-point cancellation: a
    # constrained optimum cannot exceed its corresponding free optimum.
    if np.any(result[finite] < -1.0e-9):
        raise ArithmeticError("constrained branch value exceeds its free optimum")
    result[finite] = np.maximum(result[finite], 0.0)
    result[~finite] = np.inf
    return result


class ChainFactorGraph:
    """Exact Max-Sum solver for ordered candidate tables on a chain."""

    def __init__(
        self,
        candidate_tables: Sequence[Sequence[CandidateAction]],
        cut_edges_by_link: Sequence[Iterable[Tuple[Hashable, Hashable]]],
        require_incumbent: bool = True,
        require_safe_anchor: bool = True,
    ) -> None:
        self.candidate_tables = _validate_tables(candidate_tables, require_incumbent)
        self.cut_edges_by_link = tuple(
            _normalize_cut_edges(edges) for edges in cut_edges_by_link
        )
        self.compatibility_matrices = build_adjacent_compatibility_matrices(
            self.candidate_tables, self.cut_edges_by_link
        )
        self.require_incumbent = bool(require_incumbent)
        self.require_safe_anchor = bool(require_safe_anchor)
        self.incumbent_indices = _anchor_indices(
            self.candidate_tables, "is_incumbent"
        )
        self.safe_indices = _anchor_indices(self.candidate_tables, "is_safe")
        _validate_incumbent(
            self.candidate_tables,
            self.compatibility_matrices,
            self.incumbent_indices,
            self.require_incumbent,
        )
        _validate_safe_anchors(
            self.candidate_tables,
            self.compatibility_matrices,
            self.safe_indices,
            self.require_safe_anchor,
        )

    def solve(self) -> ChainMaxSumResult:
        """Run exact forward/backward Max-Sum and recover the best tuple."""

        tables = self.candidate_tables
        compatibility = self.compatibility_matrices
        n_agents = len(tables)
        local_values = tuple(
            np.asarray(
                [
                    action.local_objective if action.is_feasible else NEG_INF
                    for action in table
                ],
                dtype=np.float64,
            )
            for table in tables
        )

        left_messages = [np.zeros(len(tables[0]), dtype=np.float64)]
        left_backpointers = [np.full(len(tables[0]), -1, dtype=np.int64)]
        for position in range(1, n_agents):
            base = left_messages[position - 1] + local_values[position - 1]
            scores = np.broadcast_to(
                base[:, None], compatibility[position - 1].shape
            ).copy()
            scores[~compatibility[position - 1]] = NEG_INF
            values, pointers = _masked_argmax(scores, axis=0)
            left_messages.append(values)
            left_backpointers.append(pointers)

        right_messages = [None] * n_agents  # type: ignore
        right_backpointers = [None] * n_agents  # type: ignore
        right_messages[-1] = np.zeros(len(tables[-1]), dtype=np.float64)
        right_backpointers[-1] = np.full(len(tables[-1]), -1, dtype=np.int64)
        for position in range(n_agents - 2, -1, -1):
            base = local_values[position + 1] + right_messages[position + 1]
            scores = np.broadcast_to(
                base[None, :], compatibility[position].shape
            ).copy()
            scores[~compatibility[position]] = NEG_INF
            values, pointers = _masked_argmax(scores, axis=1)
            right_messages[position] = values
            right_backpointers[position] = pointers

        forced = tuple(
            local_values[i] + left_messages[i] + right_messages[i]
            for i in range(n_agents)
        )

        terminal_scores = left_messages[-1] + local_values[-1]
        best_last = int(np.argmax(terminal_scores))
        best_value = float(terminal_scores[best_last])
        if not math.isfinite(best_value):
            raise NoFeasibleJointAssignmentError(
                "the restricted candidate domains contain no feasible tuple"
            )

        best_indices = [-1] * n_agents
        best_indices[-1] = best_last
        for position in range(n_agents - 1, 0, -1):
            predecessor = int(left_backpointers[position][best_indices[position]])
            if predecessor < 0:
                raise ArithmeticError("missing backpointer on a finite optimal path")
            best_indices[position - 1] = predecessor
        best_indices_tuple = tuple(best_indices)
        best_actions = tuple(
            tables[position][candidate_index]
            for position, candidate_index in enumerate(best_indices_tuple)
        )

        free_left = [0.0] * n_agents
        for position in range(1, n_agents):
            free_left[position] = float(
                np.max(left_messages[position - 1] + local_values[position - 1])
            )
        free_right = [0.0] * n_agents
        for position in range(n_agents - 2, -1, -1):
            free_right[position] = float(
                np.max(local_values[position + 1] + right_messages[position + 1])
            )

        left_costs = tuple(
            _opportunity_cost(free_left[i], left_messages[i])
            for i in range(n_agents)
        )
        right_costs = tuple(
            _opportunity_cost(free_right[i], right_messages[i])
            for i in range(n_agents)
        )
        total_costs = tuple(left_costs[i] + right_costs[i] for i in range(n_agents))

        advantages = []
        local_gains = []
        externalities = []
        for position, table in enumerate(tables):
            incumbent_index = self.incumbent_indices[position]
            if incumbent_index is None:
                advantage = np.full(len(table), np.nan, dtype=np.float64)
                local_gain = np.full(len(table), np.nan, dtype=np.float64)
            else:
                incumbent_forced = forced[position][incumbent_index]
                if not np.isfinite(incumbent_forced):
                    raise InvalidIncumbentError(
                        "incumbent candidate has no finite chain completion"
                    )
                advantage = forced[position] - incumbent_forced
                incumbent_local = table[incumbent_index].local_objective
                local_gain = np.asarray(
                    [action.local_objective - incumbent_local for action in table],
                    dtype=np.float64,
                )
            advantages.append(advantage)
            local_gains.append(local_gain)
            externalities.append(advantage - local_gain)

        return ChainMaxSumResult(
            candidate_tables=tables,
            compatibility_matrices=tuple(
                _readonly_array(matrix) for matrix in compatibility
            ),
            left_messages=tuple(_readonly_array(value) for value in left_messages),
            right_messages=tuple(_readonly_array(value) for value in right_messages),
            left_backpointers=tuple(
                _readonly_array(value) for value in left_backpointers
            ),
            right_backpointers=tuple(
                _readonly_array(value) for value in right_backpointers
            ),
            forced_best_completions=tuple(_readonly_array(value) for value in forced),
            best_completion_advantages=tuple(
                _readonly_array(value) for value in advantages
            ),
            local_gains=tuple(_readonly_array(value) for value in local_gains),
            externalities=tuple(_readonly_array(value) for value in externalities),
            left_opportunity_costs=tuple(
                _readonly_array(value) for value in left_costs
            ),
            right_opportunity_costs=tuple(
                _readonly_array(value) for value in right_costs
            ),
            opportunity_costs=tuple(
                _readonly_array(value) for value in total_costs
            ),
            completion_feasible=tuple(
                _readonly_array(np.isfinite(value)) for value in forced
            ),
            free_left_branch_values=tuple(free_left),
            free_right_branch_values=tuple(free_right),
            incumbent_indices=tuple(
                int(index) if index is not None else -1
                for index in self.incumbent_indices
            ),
            safe_indices=self.safe_indices,
            best_indices=best_indices_tuple,
            best_actions=best_actions,
            best_value=best_value,
        )


def solve_chain_max_sum(
    candidate_tables: Sequence[Sequence[CandidateAction]],
    cut_edges_by_link: Sequence[Iterable[Tuple[Hashable, Hashable]]],
    require_incumbent: bool = True,
    require_safe_anchor: bool = True,
) -> ChainMaxSumResult:
    """Convenience function for constructing and solving a chain factor graph."""

    return ChainFactorGraph(
        candidate_tables=candidate_tables,
        cut_edges_by_link=cut_edges_by_link,
        require_incumbent=require_incumbent,
        require_safe_anchor=require_safe_anchor,
    ).solve()


# Short alias for callers that describe the operation rather than the algorithm.
max_sum = solve_chain_max_sum


__all__ = [
    "CandidateAction",
    "CandidateDiagnostics",
    "ChainFactorGraph",
    "ChainMaxSumResult",
    "FactorGraphError",
    "InvalidCandidateTableError",
    "InvalidIncumbentError",
    "InvalidSafeAnchorError",
    "NoFeasibleJointAssignmentError",
    "build_compatibility_matrix",
    "build_adjacent_compatibility_matrices",
    "solve_chain_max_sum",
    "max_sum",
]
