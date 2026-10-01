"""Read-only global feasibility checks for candidate schedules."""

from __future__ import absolute_import

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .graph import ConflictGraph


@dataclass(frozen=True)
class VerificationResult:
    feasible: bool
    selected_ids: Tuple[int, ...]
    conflicting_edges: Tuple[Tuple[int, int], ...]
    invalid_ids: Tuple[int, ...]
    duplicate_ids: Tuple[int, ...]

    def __bool__(self):  # type: () -> bool
        return self.feasible

    @property
    def message(self):  # type: () -> str
        if self.feasible:
            return "feasible independent set ({} selected)".format(len(self.selected_ids))
        parts = []
        if self.invalid_ids:
            parts.append("invalid IDs={}".format(list(self.invalid_ids)))
        if self.duplicate_ids:
            parts.append("duplicate IDs={}".format(list(self.duplicate_ids)))
        if self.conflicting_edges:
            preview = list(self.conflicting_edges[:10])
            suffix = "..." if len(self.conflicting_edges) > 10 else ""
            parts.append("conflicts={}{}".format(preview, suffix))
        return "; ".join(parts)


def _coerce_ids(selected_ids):  # type: (Iterable[int]) -> Tuple[int, ...]
    result = []
    for value in selected_ids:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("Selected IDs must be integers; got {!r}".format(value))
        result.append(int(value))
    return tuple(result)


def verify_selected_ids(graph, selected_ids):
    # type: (ConflictGraph, Iterable[int]) -> VerificationResult
    """Verify range, uniqueness and independence without repairing the set."""

    raw_ids = _coerce_ids(selected_ids)
    counts = {}  # type: dict
    for node_id in raw_ids:
        counts[node_id] = counts.get(node_id, 0) + 1
    duplicate_ids = tuple(sorted(node_id for node_id, count in counts.items() if count > 1))
    invalid_ids = tuple(
        sorted(node_id for node_id in counts if node_id < 0 or node_id >= graph.num_nodes)
    )
    selected = tuple(sorted(node_id for node_id in counts if node_id not in invalid_ids))

    mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    if selected:
        mask[np.asarray(selected, dtype=np.int64)] = True
    if graph.num_edges:
        conflict_mask = mask[graph.edges[:, 0]] & mask[graph.edges[:, 1]]
        conflicting_edges = tuple(
            (int(first), int(second)) for first, second in graph.edges[conflict_mask]
        )
    else:
        conflicting_edges = tuple()
    feasible = not invalid_ids and not duplicate_ids and not conflicting_edges
    return VerificationResult(
        feasible=feasible,
        selected_ids=selected,
        conflicting_edges=conflicting_edges,
        invalid_ids=invalid_ids,
        duplicate_ids=duplicate_ids,
    )


def is_independent_set(graph, selected_ids):  # type: (ConflictGraph, Iterable[int]) -> bool
    return verify_selected_ids(graph, selected_ids).feasible


def assert_independent_set(graph, selected_ids):  # type: (ConflictGraph, Iterable[int]) -> None
    result = verify_selected_ids(graph, selected_ids)
    if not result.feasible:
        raise ValueError("Invalid selected set: {}".format(result.message))

