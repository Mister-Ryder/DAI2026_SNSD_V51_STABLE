"""Canonical conflict graph and equal-time decomposition.

The conflict predicates intentionally reproduce ``DealData.find_conf_arc``.
In particular, the satellite partial-overlap predicate may look unusual; it
must remain unchanged for first-version comparability with KP-NLNS.
"""

from __future__ import absolute_import

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .data import Arc


EDGE_GROUND = 1
EDGE_SATELLITE = 2


@dataclass(frozen=True)
class ConflictParameters:
    ground_trans_time: int = 340
    satellite_change_time: int = 150
    satellite_trans_time: int = 300

    def __post_init__(self):
        for name in (
            "ground_trans_time",
            "satellite_change_time",
            "satellite_trans_time",
        ):
            if getattr(self, name) < 0:
                raise ValueError("{} must be non-negative".format(name))


@dataclass(frozen=True)
class ConflictGraph:
    """Undirected graph in both canonical edge-list and CSR form."""

    num_nodes: int
    edges: np.ndarray
    edge_types: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    conflict_duration: np.ndarray
    params: ConflictParameters
    graph_hash: str

    @property
    def num_edges(self):  # type: () -> int
        return int(self.edges.shape[0])

    @property
    def degree(self):  # type: () -> np.ndarray
        return np.diff(self.indptr)

    @property
    def edge_index(self):  # type: () -> np.ndarray
        """Return the two-way GNN edge index with shape ``[2, 2M]``."""
        if self.num_edges == 0:
            return np.empty((2, 0), dtype=np.int64)
        return np.concatenate((self.edges.T, self.edges[:, ::-1].T), axis=1)

    def neighbors(self, node_id):  # type: (int) -> np.ndarray
        if node_id < 0 or node_id >= self.num_nodes:
            raise IndexError("node id {} outside [0,{})".format(node_id, self.num_nodes))
        return self.indices[self.indptr[node_id] : self.indptr[node_id + 1]]

    def has_edge(self, first, second):  # type: (int, int) -> bool
        if first == second:
            return False
        if first < 0 or second < 0 or first >= self.num_nodes or second >= self.num_nodes:
            return False
        neighbours = self.neighbors(first)
        position = int(np.searchsorted(neighbours, second))
        return position < len(neighbours) and int(neighbours[position]) == second


@dataclass(frozen=True)
class TimePartitions:
    """A complete edge decomposition induced by equal-time node groups."""

    k: int
    horizon_start: int
    horizon_end: int
    spans: np.ndarray
    partition_id: np.ndarray
    nodes: Tuple[np.ndarray, ...]
    internal_edges: Tuple[np.ndarray, ...]
    cut_edges: np.ndarray
    boundary_left: Tuple[np.ndarray, ...]
    boundary_right: Tuple[np.ndarray, ...]
    non_adjacent_cut_edges: np.ndarray

    @property
    def boundary(self):  # type: () -> Tuple[np.ndarray, ...]
        result = []
        for left, right in zip(self.boundary_left, self.boundary_right):
            result.append(np.union1d(left, right).astype(np.int64, copy=False))
        return tuple(result)

    @property
    def has_non_adjacent_cuts(self):  # type: () -> bool
        return bool(self.non_adjacent_cut_edges.shape[0])


def _validate_dense_arcs(arcs):  # type: (Sequence[Arc]) -> Tuple[Arc, ...]
    ordered = tuple(sorted(arcs, key=lambda arc: arc.id))
    if any(arc.id != index for index, arc in enumerate(ordered)):
        raise ValueError("Arc IDs must be unique and dense in [0, N)")
    return ordered


def _add_edge(edge_flags, num_nodes, first, second, edge_type):
    # type: (Dict[int, int], int, int, int, int) -> None
    if first == second:
        return
    first, second = (first, second) if first < second else (second, first)
    key = first * num_nodes + second
    edge_flags[key] = edge_flags.get(key, 0) | edge_type


def _graph_digest(arcs, params, edges, edge_types):
    # type: (Sequence[Arc], ConflictParameters, np.ndarray, np.ndarray) -> str
    digest = hashlib.sha256()
    digest.update(b"SNSD-CONFLICT-GRAPH-v1\0")
    digest.update(
        json.dumps(
            {
                "ground_trans_time": params.ground_trans_time,
                "satellite_change_time": params.satellite_change_time,
                "satellite_trans_time": params.satellite_trans_time,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for arc in arcs:
        digest.update(
            ("{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\n".format(
                arc.id,
                arc.ground,
                arc.satellite,
                arc.link_st,
                arc.link_et,
                arc.trace_st,
                arc.trace_et,
                arc.ground_name,
                arc.satellite_name,
            )).encode("utf-8")
        )
    digest.update(np.asarray(edges, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(edge_types, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _build_csr(num_nodes, edges):  # type: (int, np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    if len(edges) == 0:
        return np.zeros(num_nodes + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
    sources = np.concatenate((edges[:, 0], edges[:, 1])).astype(np.int64, copy=False)
    targets = np.concatenate((edges[:, 1], edges[:, 0])).astype(np.int64, copy=False)
    order = np.lexsort((targets, sources))
    sources = sources[order]
    targets = targets[order]
    counts = np.bincount(sources, minlength=num_nodes)
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, targets


def build_conflict_graph(arcs, params=None):
    # type: (Sequence[Arc], Optional[ConflictParameters]) -> ConflictGraph
    """Build the one canonical graph using the legacy ``DealData`` rules."""

    ordered = _validate_dense_arcs(arcs)
    params = params or ConflictParameters()
    num_nodes = len(ordered)
    edge_flags = {}  # type: Dict[int, int]

    by_ground = defaultdict(list)  # type: Dict[int, List[Arc]]
    by_satellite = defaultdict(list)  # type: Dict[int, List[Arc]]
    for arc in ordered:
        by_ground[arc.ground].append(arc)
        by_satellite[arc.satellite].append(arc)

    for resource_arcs in by_ground.values():
        resource_arcs.sort(key=lambda arc: (arc.link_st, arc.id))
        for index, first in enumerate(resource_arcs):
            cutoff = first.link_et + params.ground_trans_time
            for second in resource_arcs[index + 1 :]:
                if second.link_st < cutoff:
                    _add_edge(
                        edge_flags,
                        num_nodes,
                        first.id,
                        second.id,
                        EDGE_GROUND,
                    )
                else:
                    break

    for resource_arcs in by_satellite.values():
        resource_arcs.sort(key=lambda arc: (arc.link_st, arc.id))
        for index, first in enumerate(resource_arcs):
            for second in resource_arcs[index + 1 :]:
                conflicts = False
                if second.link_st == first.link_st:
                    conflicts = True
                elif second.link_st < first.link_et:
                    if second.link_et <= first.link_et:
                        conflicts = True
                    elif first.link_et - second.link_st < params.satellite_change_time:
                        conflicts = True
                elif second.link_st - first.link_et < params.satellite_trans_time:
                    conflicts = True
                else:
                    break
                if conflicts:
                    _add_edge(
                        edge_flags,
                        num_nodes,
                        first.id,
                        second.id,
                        EDGE_SATELLITE,
                    )

    sorted_keys = sorted(edge_flags)
    if sorted_keys:
        edges = np.asarray(
            [(key // num_nodes, key % num_nodes) for key in sorted_keys],
            dtype=np.int64,
        )
        edge_types = np.asarray([edge_flags[key] for key in sorted_keys], dtype=np.uint8)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
        edge_types = np.empty(0, dtype=np.uint8)

    indptr, indices = _build_csr(num_nodes, edges)
    duration = np.asarray([arc.link_time for arc in ordered], dtype=np.int64)
    conflict_duration = np.zeros(num_nodes, dtype=np.int64)
    if len(edges):
        np.add.at(conflict_duration, edges[:, 0], duration[edges[:, 1]])
        np.add.at(conflict_duration, edges[:, 1], duration[edges[:, 0]])
    graph_hash = _graph_digest(ordered, params, edges, edge_types)
    return ConflictGraph(
        num_nodes=num_nodes,
        edges=edges,
        edge_types=edge_types,
        indptr=indptr,
        indices=indices,
        conflict_duration=conflict_duration,
        params=params,
        graph_hash=graph_hash,
    )


def _empty_edges():  # type: () -> np.ndarray
    return np.empty((0, 2), dtype=np.int64)


def partition_equal_time(
    arcs,
    graph,
    k=6,
    horizon_start=0,
    horizon_end=259200,
):
    # type: (Sequence[Arc], ConflictGraph, int, int, int) -> TimePartitions
    """Assign nodes by link start time and split every graph edge exactly once.

    Intervals are half open except that a node exactly at ``horizon_end`` is
    assigned to the final partition.  Starts outside the declared horizon are
    rejected to prevent silently changing the experimental decomposition.
    """

    ordered = _validate_dense_arcs(arcs)
    if len(ordered) != graph.num_nodes:
        raise ValueError("Graph and arc count differ")
    if k <= 0:
        raise ValueError("k must be positive")
    if horizon_end <= horizon_start:
        raise ValueError("horizon_end must be greater than horizon_start")

    starts = np.asarray([arc.link_st for arc in ordered], dtype=np.int64)
    if len(starts) and (int(starts.min()) < horizon_start or int(starts.max()) > horizon_end):
        raise ValueError(
            "Arc start outside declared horizon [{}, {}]".format(
                horizon_start, horizon_end
            )
        )
    span = horizon_end - horizon_start
    partition_id = ((starts - horizon_start) * k // span).astype(np.int64)
    if len(partition_id):
        np.clip(partition_id, 0, k - 1, out=partition_id)
    spans = np.linspace(horizon_start, horizon_end, k + 1, dtype=np.float64)
    nodes = tuple(
        np.flatnonzero(partition_id == partition).astype(np.int64, copy=False)
        for partition in range(k)
    )

    if graph.num_edges:
        same_partition = partition_id[graph.edges[:, 0]] == partition_id[graph.edges[:, 1]]
        cut_edges = graph.edges[~same_partition].copy()
        internal_edges = tuple(
            graph.edges[
                same_partition & (partition_id[graph.edges[:, 0]] == partition)
            ].copy()
            for partition in range(k)
        )
    else:
        cut_edges = _empty_edges()
        internal_edges = tuple(_empty_edges() for _ in range(k))

    left_sets = [set() for _ in range(k)]
    right_sets = [set() for _ in range(k)]
    non_adjacent = []
    for first, second in cut_edges:
        first = int(first)
        second = int(second)
        first_partition = int(partition_id[first])
        second_partition = int(partition_id[second])
        if first_partition < second_partition:
            right_sets[first_partition].add(first)
            left_sets[second_partition].add(second)
        else:
            left_sets[first_partition].add(first)
            right_sets[second_partition].add(second)
        if abs(first_partition - second_partition) > 1:
            non_adjacent.append((first, second))

    boundary_left = tuple(
        np.asarray(sorted(values), dtype=np.int64) for values in left_sets
    )
    boundary_right = tuple(
        np.asarray(sorted(values), dtype=np.int64) for values in right_sets
    )
    non_adjacent_cut_edges = (
        np.asarray(non_adjacent, dtype=np.int64)
        if non_adjacent
        else _empty_edges()
    )
    return TimePartitions(
        k=k,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        spans=spans,
        partition_id=partition_id,
        nodes=nodes,
        internal_edges=internal_edges,
        cut_edges=cut_edges,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        non_adjacent_cut_edges=non_adjacent_cut_edges,
    )


def save_graph_cache(path, graph, partitions=None):
    # type: (str, ConflictGraph, Optional[TimePartitions]) -> None
    """Write numeric graph/partition arrays to a pickle-free compressed NPZ."""

    payload = {
        "cache_version": np.asarray("SNSD-GRAPH-v1"),
        "num_nodes": np.asarray(graph.num_nodes, dtype=np.int64),
        "edges": graph.edges,
        "edge_types": graph.edge_types,
        "indptr": graph.indptr,
        "indices": graph.indices,
        "conflict_duration": graph.conflict_duration,
        "graph_hash": np.asarray(graph.graph_hash),
        "ground_trans_time": np.asarray(graph.params.ground_trans_time, dtype=np.int64),
        "satellite_change_time": np.asarray(
            graph.params.satellite_change_time, dtype=np.int64
        ),
        "satellite_trans_time": np.asarray(
            graph.params.satellite_trans_time, dtype=np.int64
        ),
        "has_partitions": np.asarray(1 if partitions is not None else 0, dtype=np.uint8),
    }
    if partitions is not None:
        payload.update(
            {
                "partition_k": np.asarray(partitions.k, dtype=np.int64),
                "horizon_start": np.asarray(partitions.horizon_start, dtype=np.int64),
                "horizon_end": np.asarray(partitions.horizon_end, dtype=np.int64),
                "spans": partitions.spans,
                "partition_id": partitions.partition_id,
                "cut_edges": partitions.cut_edges,
                "non_adjacent_cut_edges": partitions.non_adjacent_cut_edges,
            }
        )
        for partition in range(partitions.k):
            payload["nodes_{}".format(partition)] = partitions.nodes[partition]
            payload["internal_edges_{}".format(partition)] = partitions.internal_edges[partition]
            payload["boundary_left_{}".format(partition)] = partitions.boundary_left[partition]
            payload["boundary_right_{}".format(partition)] = partitions.boundary_right[partition]
    np.savez_compressed(os.fspath(path), **payload)


def load_graph_cache(path):
    # type: (str) -> Tuple[ConflictGraph, Optional[TimePartitions]]
    """Load :func:`save_graph_cache` output without permitting object pickle."""

    with np.load(os.fspath(path), allow_pickle=False) as cached:
        version = str(cached["cache_version"].item())
        if version != "SNSD-GRAPH-v1":
            raise ValueError("Unsupported graph cache version {!r}".format(version))
        params = ConflictParameters(
            ground_trans_time=int(cached["ground_trans_time"].item()),
            satellite_change_time=int(cached["satellite_change_time"].item()),
            satellite_trans_time=int(cached["satellite_trans_time"].item()),
        )
        graph = ConflictGraph(
            num_nodes=int(cached["num_nodes"].item()),
            edges=cached["edges"].copy(),
            edge_types=cached["edge_types"].copy(),
            indptr=cached["indptr"].copy(),
            indices=cached["indices"].copy(),
            conflict_duration=cached["conflict_duration"].copy(),
            params=params,
            graph_hash=str(cached["graph_hash"].item()),
        )
        if not bool(int(cached["has_partitions"].item())):
            return graph, None

        k = int(cached["partition_k"].item())
        partitions = TimePartitions(
            k=k,
            horizon_start=int(cached["horizon_start"].item()),
            horizon_end=int(cached["horizon_end"].item()),
            spans=cached["spans"].copy(),
            partition_id=cached["partition_id"].copy(),
            nodes=tuple(cached["nodes_{}".format(i)].copy() for i in range(k)),
            internal_edges=tuple(
                cached["internal_edges_{}".format(i)].copy() for i in range(k)
            ),
            cut_edges=cached["cut_edges"].copy(),
            boundary_left=tuple(
                cached["boundary_left_{}".format(i)].copy() for i in range(k)
            ),
            boundary_right=tuple(
                cached["boundary_right_{}".format(i)].copy() for i in range(k)
            ),
            non_adjacent_cut_edges=cached["non_adjacent_cut_edges"].copy(),
        )
    return graph, partitions

