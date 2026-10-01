"""Tensor adapter and leakage-safe inference runtime for TriGraph v1.5.

The optimization code deliberately keeps the public problem representation as
plain ``ArcDataset``/``ConflictGraph``/``TimePartitions`` objects.  This module
is the narrow bridge from those objects to :mod:`graph_layers`.  It performs three
jobs:

* build full, local-plus-halo, and boundary-cut tensors with stable ordering;
* cache static tensors and full-graph encodings keyed by graph, incumbent and
  feature mode;
* rank unqueried :class:`ProposalIntent` objects through
  ``score_intents_prequery`` without accepting any worker objective or exact
  candidate gain as an input.

The feature map is intentionally modest and auditable.  It uses arc metadata,
conflict topology, the accepted incumbent boundary bits, ``BoundaryState``
public fields, and optional peer ``BoundaryMessage`` channels.  In particular,
``BoundaryState.local_marginal`` is not consumed: it may contain a revealed
local-worker quantity in some runners and therefore cannot enter acquisition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .model import (
    BoundaryGraphState,
    FeatureMode,
    FullGraphState,
    LocalGraphState,
    PrequeryIntentScores,
    TriGraphConfig,
    TriGraphModel,
)
from .messages import BoundaryMessage, BoundaryState, ProposalIntent


Tensor = torch.Tensor


def _as_float(array: Any) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    result[~np.isfinite(result)] = 0.0
    return result


def _scale(array: Any) -> float:
    values = np.abs(_as_float(array).reshape(-1))
    if not len(values):
        return 1.0
    value = float(np.percentile(values, 95.0))
    return value if value > 1.0e-12 else 1.0


def _normalise(array: Any, scale: Optional[float] = None) -> np.ndarray:
    values = _as_float(array)
    denominator = _scale(values) if scale is None else max(float(scale), 1.0e-12)
    return np.tanh(values / denominator).astype(np.float32, copy=False)


def _fit_columns(array: np.ndarray, columns: int) -> np.ndarray:
    """Truncate/pad a public feature table to a configured feature group."""

    values = _as_float(array)
    if values.ndim != 2:
        raise ValueError("feature table must be two-dimensional")
    columns = int(columns)
    if values.shape[1] == columns:
        return values
    if values.shape[1] > columns:
        return values[:, :columns]
    padding = np.zeros((values.shape[0], columns - values.shape[1]), dtype=np.float32)
    return np.concatenate((values, padding), axis=1)


def _tensor(array: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def _selected_ids(value: Any) -> Tuple[int, ...]:
    """Extract selected IDs without inspecting any objective-valued field."""

    if value is None:
        return tuple()
    if hasattr(value, "signature"):
        return _selected_ids(value.signature)
    if hasattr(value, "result"):
        return _selected_ids(value.result)
    if hasattr(value, "selected_ids"):
        return tuple(int(node) for node in value.selected_ids)
    if isinstance(value, Mapping):
        return tuple(int(node) for node, bit in value.items() if bool(bit))
    return tuple(int(node) for node in value)


def _incumbent_set(
    incumbent: Optional[Any], boundary_states: Mapping[int, BoundaryState]
) -> Tuple[int, ...]:
    selected = set()  # type: set
    if incumbent is None:
        for state in boundary_states.values():
            selected.update(int(node) for node in state.incumbent_selected)
    elif isinstance(incumbent, Mapping):
        for value in incumbent.values():
            selected.update(_selected_ids(value))
    else:
        selected.update(_selected_ids(incumbent))
    return tuple(sorted(selected))


def _model_digest(graph_layers: TriGraphModel) -> str:
    digest = hashlib.sha256()
    digest.update(b"SNSD-TRIGRAPH-MODEL-v2\0")
    for name, value in sorted(graph_layers.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ModelMetadata:
    model_hash: str
    parameter_count: int
    trainable_parameter_count: int
    checkpoint_path: Optional[str]
    checkpoint_loaded: bool
    device: str
    dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PreparedTriGraph:
    """Encoded three-view state for one incumbent/message snapshot."""

    mode: FeatureMode
    full_state: FullGraphState
    local_states: Mapping[int, LocalGraphState]
    boundary_state: BoundaryGraphState
    boundary_indices: Mapping[int, Tensor]
    boundary_orders: Mapping[int, Tuple[int, ...]]
    boundary_states: Mapping[int, BoundaryState]
    boundary_messages: Mapping[int, BoundaryMessage]
    incumbent_selected: Tuple[int, ...]
    benefit_scale: float
    cache_key: str


@dataclass(frozen=True)
class RankedIntent:
    rank: int
    intent: ProposalIntent
    selection_logit: float
    predicted_local_benefit: float
    learned_externality: float
    predicted_best_completion_advantage: float
    completion_probability: float
    uncertainty: float
    raw_arc_benefit_proxy: float
    score_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "intent_hash": self.intent.intent_hash,
            "agent_id": self.intent.agent_id,
            "selection_logit": self.selection_logit,
            "predicted_local_benefit": self.predicted_local_benefit,
            "learned_externality": self.learned_externality,
            "predicted_best_completion_advantage": self.predicted_best_completion_advantage,
            "completion_probability": self.completion_probability,
            "uncertainty": self.uncertainty,
            "raw_arc_benefit_proxy": self.raw_arc_benefit_proxy,
            "score_hash": self.score_hash,
        }


@dataclass
class _StaticTensors:
    graph_key: str
    full_structural_base: np.ndarray
    full_benefit: np.ndarray
    full_conflict: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_features: np.ndarray
    weights: np.ndarray
    priority: np.ndarray
    degree: np.ndarray
    cut_degree: np.ndarray
    owner: np.ndarray


@dataclass
class _LocalTopology:
    """Immutable local-plus-halo topology and device tensors.

    These values depend only on the C3 graph/partition/candidate boundaries,
    not on an incumbent, a message, or graph_layers parameters.  Keeping them beside
    the policy avoids rebuilding roughly one million edge mappings for every
    Real/Zero rollout step.
    """

    node_ids: np.ndarray
    row_map: Mapping[int, int]
    global_node_ids: Tensor
    structural_features: Tensor
    benefit_features_base: Tensor
    benefit_selected_column: Optional[int]
    conflict_features: Tensor
    edge_index: Tensor
    core_mask: Tensor
    halo_mask: Tensor
    boundary_mask: Tensor
    edge_type: Tensor
    edge_features: Tensor
    shuffle_index: Optional[Tensor]
    boundary_local_indices: Tensor


@dataclass
class _TopologyTensors:
    """Graph-static tensors shared by every prepare call in one policy."""

    cache_key: str
    full_structural_base: Tensor
    full_selected_column: Optional[int]
    full_benefit: Tensor
    full_conflict: Tensor
    full_edge_index: Tensor
    full_edge_type: Tensor
    full_edge_features: Tensor
    local: Mapping[int, _LocalTopology]
    boundary_orders: Mapping[int, Tuple[int, ...]]
    boundary_global: np.ndarray
    boundary_row: Mapping[int, int]
    boundary_global_tensor: Tensor
    boundary_owners: np.ndarray
    boundary_owner_tensor: Tensor
    boundary_structural_base: Tensor
    boundary_selected_column: Optional[int]
    boundary_benefit: Tensor
    boundary_conflict: Tensor
    cut_edge_index: Tensor
    cut_edge_type: Tensor
    cut_edge_features: Tensor
    boundary_shuffle_index: Optional[Tensor]
    boundary_indices: Mapping[int, Tensor]


class TriGraphPolicy(object):
    """Inference facade used by the base_protocol runner.

    A checkpoint is optional for wiring/smoke tests.  ``metadata`` always
    exposes the exact initialized weights, so an untrained smoke run cannot be
    mistaken for a trained policy in base_protocol logs.
    """

    def __init__(
        self,
        graph_layers: Optional[TriGraphModel] = None,
        config: Optional[TriGraphConfig] = None,
        device: Union[str, torch.device] = "cpu",
        checkpoint_path: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        if graph_layers is None:
            torch.manual_seed(int(seed))
            graph_layers = TriGraphModel(config=config)
        self.graph_layers = graph_layers.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_loaded = False
        if checkpoint_path:
            payload = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(payload, Mapping) and "state_dict" in payload:
                payload = payload["state_dict"]
            self.graph_layers.load_state_dict(payload, strict=True)
            self.checkpoint_loaded = True
        self.graph_layers.eval()
        self._static_cache = {}  # type: Dict[str, _StaticTensors]
        self._topology_cache = {}  # type: Dict[str, _TopologyTensors]
        self._full_state_cache = {}  # type: Dict[str, FullGraphState]
        self._refresh_metadata()

    def _refresh_metadata(self) -> None:
        parameters = list(self.graph_layers.parameters())
        self.metadata = ModelMetadata(
            model_hash=_model_digest(self.graph_layers),
            parameter_count=sum(int(item.numel()) for item in parameters),
            trainable_parameter_count=sum(
                int(item.numel()) for item in parameters if item.requires_grad
            ),
            checkpoint_path=self.checkpoint_path,
            checkpoint_loaded=self.checkpoint_loaded,
            device=str(self.device),
            dtype=str(next(self.graph_layers.parameters()).dtype),
        )

    def invalidate_encoded_cache(self) -> None:
        """Call after optimizer updates or loading different graph_layers weights."""

        self._full_state_cache.clear()
        self._refresh_metadata()

    @staticmethod
    def _graph_key(dataset: Any, graph: Any, partitions: Any) -> str:
        return _stable_digest(
            {
                "graph_hash": str(getattr(graph, "graph_hash", "")),
                "nodes": int(graph.num_nodes),
                "edges": int(graph.num_edges),
                "partition_k": int(partitions.k),
                "source": str(getattr(dataset, "source_path", "")),
            }
        )

    def _build_static(self, dataset: Any, graph: Any, partitions: Any) -> _StaticTensors:
        graph_key = self._graph_key(dataset, graph, partitions)
        cached = self._static_cache.get(graph_key)
        if cached is not None:
            return cached
        arcs = tuple(dataset.arcs)
        if len(arcs) != int(graph.num_nodes):
            raise ValueError("dataset and graph node counts differ")
        weights = _as_float([float(arc.weight) for arc in arcs])
        priority = _as_float([float(getattr(arc, "priority", 0.0)) for arc in arcs])
        degree = _as_float(graph.degree)
        conflict_duration = _as_float(graph.conflict_duration)
        owner = np.asarray(partitions.partition_id, dtype=np.int64)
        cut_degree = np.zeros(int(graph.num_nodes), dtype=np.float32)
        if len(partitions.cut_edges):
            np.add.at(cut_degree, partitions.cut_edges[:, 0], 1.0)
            np.add.at(cut_degree, partitions.cut_edges[:, 1], 1.0)
        horizon = max(1.0, float(partitions.horizon_end - partitions.horizon_start))
        start = _as_float([arc.link_st for arc in arcs])
        end = _as_float([arc.link_et for arc in arcs])
        trace_start = _as_float([arc.trace_st for arc in arcs])
        trace_end = _as_float([arc.trace_et for arc in arcs])
        ground = _as_float([arc.ground for arc in arcs])
        satellite = _as_float([arc.satellite for arc in arcs])
        structural = np.column_stack(
            (
                (start - float(partitions.horizon_start)) / horizon,
                (end - float(partitions.horizon_start)) / horizon,
                (trace_start - float(partitions.horizon_start)) / horizon,
                (trace_end - float(partitions.horizon_start)) / horizon,
                _normalise(degree),
                ground / max(1.0, float(max(1, getattr(dataset, "num_grounds", 1)) - 1)),
                satellite
                / max(1.0, float(max(1, getattr(dataset, "num_satellites", 1)) - 1)),
                owner / max(1.0, float(int(partitions.k) - 1)),
            )
        ).astype(np.float32, copy=False)
        benefit = np.column_stack((_normalise(weights), _normalise(priority)))
        # Eight explicit topology channels.  No worker/candidate label appears.
        types = np.asarray(graph.edge_types, dtype=np.int64)
        type_degree = []
        for relation in (1, 2, 3):
            values = np.zeros(int(graph.num_nodes), dtype=np.float32)
            mask = types == relation
            if bool(np.any(mask)):
                edges = graph.edges[mask]
                np.add.at(values, edges[:, 0], 1.0)
                np.add.at(values, edges[:, 1], 1.0)
            type_degree.append(_normalise(values))
        boundary_flag = (cut_degree > 0).astype(np.float32)
        conflict = np.column_stack(
            (
                _normalise(degree),
                _normalise(conflict_duration),
                _normalise(cut_degree),
                boundary_flag,
                type_degree[0],
                type_degree[1],
                type_degree[2],
                _normalise(conflict_duration / np.maximum(weights, 1.0)),
            )
        )
        edges = np.asarray(graph.edges, dtype=np.int64)
        static = _StaticTensors(
            graph_key=graph_key,
            full_structural_base=structural,
            full_benefit=benefit.astype(np.float32, copy=False),
            full_conflict=conflict.astype(np.float32, copy=False),
            edge_index=edges.T.copy(),
            edge_type=np.asarray(graph.edge_types, dtype=np.int64).copy(),
            edge_features=np.ones((len(edges), 1), dtype=np.float32),
            weights=weights,
            priority=priority,
            degree=degree,
            cut_degree=cut_degree,
            owner=owner,
        )
        self._static_cache[graph_key] = static
        return static

    def _build_topology(
        self,
        dataset: Any,
        graph: Any,
        partitions: Any,
        generators: Sequence[Any],
        static: _StaticTensors,
    ) -> _TopologyTensors:
        """Build invariant graph views once, with exact legacy row ordering."""

        boundary_orders = {
            agent_id: tuple(int(node) for node in generators[agent_id].boundary)
            for agent_id in range(int(partitions.k))
        }
        cache_key = _stable_digest(
            {
                "graph": static.graph_key,
                "device": str(self.device),
                "boundaries": {
                    str(key): list(value) for key, value in boundary_orders.items()
                },
                "feature_widths": {
                    "full_struct": int(self.graph_layers.config.full_struct_dim),
                    "full_benefit": int(self.graph_layers.config.full_benefit_dim),
                    "full_conflict": int(self.graph_layers.config.full_conflict_dim),
                    "full_edge": int(self.graph_layers.config.full_edge_dim),
                    "local_struct": int(self.graph_layers.config.local_struct_dim),
                    "local_benefit": int(self.graph_layers.config.local_benefit_dim),
                    "local_conflict": int(self.graph_layers.config.local_conflict_dim),
                    "local_edge": int(self.graph_layers.config.local_edge_dim),
                    "boundary_struct": int(self.graph_layers.config.boundary_struct_dim),
                    "boundary_benefit": int(self.graph_layers.config.boundary_benefit_dim),
                    "boundary_conflict": int(self.graph_layers.config.boundary_conflict_dim),
                    "boundary_edge": int(self.graph_layers.config.boundary_edge_dim),
                },
            }
        )
        cached = self._topology_cache.get(cache_key)
        if cached is not None:
            return cached

        full_structural_np = _fit_columns(
            static.full_structural_base, self.graph_layers.config.full_struct_dim
        )
        full_selected_column = (
            int(static.full_structural_base.shape[1]) - 1
            if int(self.graph_layers.config.full_struct_dim)
            >= int(static.full_structural_base.shape[1])
            and int(static.full_structural_base.shape[1]) > 0
            else None
        )
        graph_edges = np.asarray(graph.edges, dtype=np.int64)
        local = {}  # type: Dict[int, _LocalTopology]
        for agent_id in range(int(partitions.k)):
            incident = (
                (static.owner[graph_edges[:, 0]] == agent_id)
                | (static.owner[graph_edges[:, 1]] == agent_id)
            ) if len(graph_edges) else np.zeros(0, dtype=bool)
            edges = graph_edges[incident]
            if len(edges):
                node_ids = np.unique(edges.reshape(-1)).astype(
                    np.int64, copy=False
                )
            else:
                node_ids = np.asarray(partitions.nodes[agent_id], dtype=np.int64)
            node_ids = np.union1d(node_ids, partitions.nodes[agent_id]).astype(
                np.int64, copy=False
            )
            row_map = {int(node): row for row, node in enumerate(node_ids)}
            if len(edges):
                # Vectorized inverse mapping is exactly equivalent to the old
                # Python list comprehension and preserves graph edge order.
                inverse = np.full(int(graph.num_nodes), -1, dtype=np.int64)
                inverse[node_ids] = np.arange(len(node_ids), dtype=np.int64)
                local_edge_index = inverse[edges].T.copy()
                if bool(np.any(local_edge_index < 0)):
                    raise ValueError("local edge endpoint is absent from node_ids")
                edge_types = static.edge_type[incident]
            else:
                local_edge_index = np.empty((2, 0), dtype=np.int64)
                edge_types = np.empty(0, dtype=np.int64)
            core_mask = static.owner[node_ids] == agent_id
            boundary_set = set(boundary_orders[agent_id])
            boundary_mask = np.asarray(
                [int(node) in boundary_set for node in node_ids], dtype=bool
            )
            halo_mask = ~core_mask
            local_structural = np.column_stack(
                (
                    static.full_structural_base[node_ids, 0],
                    _normalise(static.weights[node_ids]),
                    core_mask.astype(np.float32),
                    boundary_mask.astype(np.float32),
                )
            )
            local_benefit = np.column_stack(
                (
                    _normalise(static.weights[node_ids]),
                    _normalise(static.priority[node_ids]),
                    np.zeros(len(node_ids), dtype=np.float32),
                )
            )
            local_conflict = np.column_stack(
                (
                    _normalise(static.degree[node_ids]),
                    _normalise(static.cut_degree[node_ids]),
                    (static.owner[node_ids] != agent_id).astype(np.float32),
                )
            )
            benefit_selected_column = (
                2 if int(self.graph_layers.config.local_benefit_dim) >= 3 else None
            )
            shuffle_index = (
                torch.roll(
                    torch.arange(
                        len(node_ids), device=self.device, dtype=torch.long
                    ),
                    shifts=-1,
                )
                if len(node_ids)
                else None
            )
            boundary_local_indices = _tensor(
                np.asarray(
                    [row_map[int(node)] for node in boundary_orders[agent_id]],
                    dtype=np.int64,
                ),
                self.device,
                torch.long,
            )
            local[agent_id] = _LocalTopology(
                node_ids=node_ids,
                row_map=row_map,
                global_node_ids=_tensor(node_ids, self.device, torch.long),
                structural_features=_tensor(
                    _fit_columns(
                        local_structural, self.graph_layers.config.local_struct_dim
                    ),
                    self.device,
                ),
                benefit_features_base=_tensor(
                    _fit_columns(
                        local_benefit, self.graph_layers.config.local_benefit_dim
                    ),
                    self.device,
                ),
                benefit_selected_column=benefit_selected_column,
                conflict_features=_tensor(
                    _fit_columns(
                        local_conflict, self.graph_layers.config.local_conflict_dim
                    ),
                    self.device,
                ),
                edge_index=_tensor(
                    local_edge_index, self.device, torch.long
                ),
                core_mask=_tensor(core_mask, self.device, torch.bool),
                halo_mask=_tensor(halo_mask, self.device, torch.bool),
                boundary_mask=_tensor(boundary_mask, self.device, torch.bool),
                edge_type=_tensor(edge_types, self.device, torch.long),
                edge_features=_tensor(
                    np.ones(
                        (len(edges), self.graph_layers.config.local_edge_dim),
                        dtype=np.float32,
                    ),
                    self.device,
                ),
                shuffle_index=shuffle_index,
                boundary_local_indices=boundary_local_indices,
            )

        boundary_global = np.asarray(
            [
                node
                for agent_id in range(int(partitions.k))
                for node in boundary_orders[agent_id]
            ],
            dtype=np.int64,
        )
        if len(set(int(node) for node in boundary_global)) != len(boundary_global):
            raise ValueError("a boundary node is owned by more than one agent")
        boundary_row = {
            int(node): row for row, node in enumerate(boundary_global)
        }
        owners = (
            static.owner[boundary_global]
            if len(boundary_global)
            else np.empty(0, dtype=np.int64)
        )
        cut_edges = np.asarray(partitions.cut_edges, dtype=np.int64)
        if len(cut_edges):
            cut_index = np.asarray(
                [
                    [boundary_row[int(u)], boundary_row[int(v)]]
                    for u, v in cut_edges
                ],
                dtype=np.int64,
            ).T
            cut_mask = (
                static.owner[graph_edges[:, 0]]
                != static.owner[graph_edges[:, 1]]
            )
            cut_types = static.edge_type[cut_mask]
            if len(cut_types) != len(cut_edges) or not np.array_equal(
                graph_edges[cut_mask], cut_edges
            ):
                raise ValueError(
                    "partition cut edges are not aligned with graph edges"
                )
        else:
            cut_index = np.empty((2, 0), dtype=np.int64)
            cut_types = np.empty(0, dtype=np.int64)
        boundary_structural = (
            np.column_stack(
                (
                    owners / max(1.0, float(int(partitions.k) - 1)),
                    np.zeros(len(boundary_global), dtype=np.float32),
                )
            )
            if len(boundary_global)
            else np.zeros((0, 2), dtype=np.float32)
        )
        boundary_benefit = (
            np.column_stack(
                (
                    _normalise(static.weights[boundary_global]),
                    _normalise(static.priority[boundary_global]),
                )
            )
            if len(boundary_global)
            else np.zeros((0, 2), dtype=np.float32)
        )
        boundary_conflict = (
            np.column_stack(
                (
                    _normalise(static.degree[boundary_global]),
                    _normalise(static.cut_degree[boundary_global]),
                )
            )
            if len(boundary_global)
            else np.zeros((0, 2), dtype=np.float32)
        )
        boundary_selected_column = (
            1 if int(self.graph_layers.config.boundary_struct_dim) >= 2 else None
        )
        boundary_indices = {}  # type: Dict[int, Tensor]
        cursor = 0
        for agent_id in range(int(partitions.k)):
            width = len(boundary_orders[agent_id])
            boundary_indices[agent_id] = torch.arange(
                cursor,
                cursor + width,
                device=self.device,
                dtype=torch.long,
            )
            cursor += width
        topology = _TopologyTensors(
            cache_key=cache_key,
            full_structural_base=_tensor(
                full_structural_np, self.device
            ),
            full_selected_column=full_selected_column,
            full_benefit=_tensor(
                _fit_columns(
                    static.full_benefit, self.graph_layers.config.full_benefit_dim
                ),
                self.device,
            ),
            full_conflict=_tensor(
                _fit_columns(
                    static.full_conflict, self.graph_layers.config.full_conflict_dim
                ),
                self.device,
            ),
            full_edge_index=_tensor(
                static.edge_index, self.device, torch.long
            ),
            full_edge_type=_tensor(
                static.edge_type, self.device, torch.long
            ),
            full_edge_features=_tensor(
                _fit_columns(
                    static.edge_features, self.graph_layers.config.full_edge_dim
                ),
                self.device,
            ),
            local=local,
            boundary_orders=boundary_orders,
            boundary_global=boundary_global,
            boundary_row=boundary_row,
            boundary_global_tensor=_tensor(
                boundary_global, self.device, torch.long
            ),
            boundary_owners=owners,
            boundary_owner_tensor=_tensor(owners, self.device, torch.long),
            boundary_structural_base=_tensor(
                _fit_columns(
                    boundary_structural,
                    self.graph_layers.config.boundary_struct_dim,
                ),
                self.device,
            ),
            boundary_selected_column=boundary_selected_column,
            boundary_benefit=_tensor(
                _fit_columns(
                    boundary_benefit, self.graph_layers.config.boundary_benefit_dim
                ),
                self.device,
            ),
            boundary_conflict=_tensor(
                _fit_columns(
                    boundary_conflict,
                    self.graph_layers.config.boundary_conflict_dim,
                ),
                self.device,
            ),
            cut_edge_index=_tensor(cut_index, self.device, torch.long),
            cut_edge_type=_tensor(cut_types, self.device, torch.long),
            cut_edge_features=_tensor(
                np.ones(
                    (len(cut_edges), self.graph_layers.config.boundary_edge_dim),
                    dtype=np.float32,
                ),
                self.device,
            ),
            boundary_shuffle_index=(
                torch.roll(
                    torch.arange(
                        len(boundary_global),
                        device=self.device,
                        dtype=torch.long,
                    ),
                    shifts=-1,
                )
                if len(boundary_global)
                else None
            ),
            boundary_indices=boundary_indices,
        )
        self._topology_cache[cache_key] = topology
        return topology

    @staticmethod
    def _message_tables(
        node_ids: np.ndarray,
        owner: np.ndarray,
        boundary_states: Mapping[int, BoundaryState],
        messages: Mapping[int, BoundaryMessage],
        row_map: Optional[Mapping[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = int(len(node_ids))
        conflict = np.zeros((rows, 3), dtype=np.float32)
        benefit = np.zeros((rows, 3), dtype=np.float32)
        rows_by_node = (
            dict(row_map)
            if row_map is not None
            else {int(node): row for row, node in enumerate(node_ids)}
        )

        def scatter(values: Mapping[int, float], agent_id: int, table: np.ndarray, column: int) -> None:
            for raw_node, raw_value in values.items():
                node = int(raw_node)
                row = rows_by_node.get(node)
                if row is not None and int(owner[node]) == int(agent_id):
                    table[int(row), int(column)] = float(raw_value)

        # State fallbacks are written first; real message channels overwrite
        # only the keys they actually contain, exactly matching dict.get in
        # the former dense all-node loop.  local_marginal remains prohibited.
        for raw_agent_id, state in boundary_states.items():
            agent_id = int(raw_agent_id)
            scatter(state.value_map("conflict_score"), agent_id, conflict, 1)
            scatter(state.value_map("uncertainty"), agent_id, conflict, 2)
            scatter(state.value_map("raw_benefit"), agent_id, benefit, 2)
        for raw_agent_id, message in messages.items():
            agent_id = int(raw_agent_id)
            scatter(message.value_map("pressure"), agent_id, conflict, 0)
            scatter(message.value_map("conflict_score"), agent_id, conflict, 1)
            scatter(message.value_map("uncertainty"), agent_id, conflict, 2)
            scatter(message.value_map("opportunity_cost"), agent_id, benefit, 0)
            scatter(message.value_map("neighbor_recovery"), agent_id, benefit, 1)
        for column in range(3):
            conflict[:, column] = _normalise(conflict[:, column])
            benefit[:, column] = _normalise(benefit[:, column])
        return conflict, benefit

    def prepare(
        self,
        dataset: Any,
        graph: Any,
        partitions: Any,
        generators: Sequence[Any],
        boundary_states: Mapping[int, BoundaryState],
        boundary_messages: Optional[Mapping[int, BoundaryMessage]] = None,
        incumbent: Optional[Any] = None,
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
    ) -> PreparedTriGraph:
        """Encode one acquisition snapshot from public/pre-query information."""

        parsed = FeatureMode.parse(mode)
        if len(generators) != int(self.graph_layers.config.num_agents):
            raise ValueError("generator count must equal graph_layers.config.num_agents")
        if int(partitions.k) != int(self.graph_layers.config.num_agents):
            raise ValueError("partition count must equal graph_layers.config.num_agents")
        messages = dict(boundary_messages or {})
        static = self._build_static(dataset, graph, partitions)
        topology = self._build_topology(
            dataset, graph, partitions, generators, static
        )
        selected = _incumbent_set(incumbent, boundary_states)
        selected_mask = np.zeros(int(graph.num_nodes), dtype=np.float32)
        if selected:
            selected_array = np.asarray(selected, dtype=np.int64)
            if int(selected_array.min()) < 0 or int(selected_array.max()) >= int(graph.num_nodes):
                raise ValueError("incumbent contains an out-of-range node")
            selected_mask[selected_array] = 1.0
        # Accepted-incumbent bit replaces the final source structural channel.
        # The fitted static tensor is cloned only when that channel survives
        # the configured feature-width truncation.
        full_structural = topology.full_structural_base
        if topology.full_selected_column is not None:
            full_structural = full_structural.clone()
            full_structural[:, topology.full_selected_column] = _tensor(
                selected_mask, self.device
            )
        incumbent_hash = _stable_digest(selected)
        full_key = _stable_digest(
            (static.graph_key, parsed.value, incumbent_hash, self.metadata.model_hash)
        )
        full_state = self._full_state_cache.get(full_key)
        if full_state is None:
            with torch.no_grad():
                full_state = self.graph_layers.encode_full(
                    full_structural,
                    topology.full_benefit,
                    topology.full_conflict,
                    topology.full_edge_index,
                    edge_type=topology.full_edge_type,
                    edge_features=topology.full_edge_features,
                    mode=parsed,
                    undirected=True,
                )
            self._full_state_cache[full_key] = full_state

        local_states = {}  # type: Dict[int, LocalGraphState]
        for agent_id in range(int(partitions.k)):
            local_topology = topology.local[agent_id]
            message_conflict, message_benefit = self._message_tables(
                local_topology.node_ids,
                static.owner,
                boundary_states,
                messages,
                row_map=local_topology.row_map,
            )
            local_benefit = local_topology.benefit_features_base
            if local_topology.benefit_selected_column is not None:
                local_benefit = local_benefit.clone()
                local_benefit[:, local_topology.benefit_selected_column] = _tensor(
                    selected_mask[local_topology.node_ids], self.device
                )
            with torch.no_grad():
                local_states[agent_id] = self.graph_layers.encode_local(
                    agent_id=agent_id,
                    full_state=full_state,
                    global_node_ids=local_topology.global_node_ids,
                    structural_features=local_topology.structural_features,
                    benefit_features=local_benefit,
                    conflict_features=local_topology.conflict_features,
                    edge_index=local_topology.edge_index,
                    core_mask=local_topology.core_mask,
                    halo_mask=local_topology.halo_mask,
                    boundary_mask=local_topology.boundary_mask,
                    edge_type=local_topology.edge_type,
                    edge_features=local_topology.edge_features,
                    message_conflict=_tensor(
                        _fit_columns(message_conflict, self.graph_layers.config.message_conflict_dim), self.device
                    ),
                    message_benefit=_tensor(
                        _fit_columns(message_benefit, self.graph_layers.config.message_benefit_dim), self.device
                    ),
                    mode=parsed,
                    shuffle_index=(
                        local_topology.shuffle_index
                        if parsed == FeatureMode.SHUFFLED
                        else None
                    ),
                    undirected=True,
                )

        boundary_orders = topology.boundary_orders
        boundary_global = topology.boundary_global
        owners = topology.boundary_owners
        if len(boundary_global):
            stacked_local = torch.cat(
                [
                    local_states[agent_id].node_embeddings[
                        topology.local[agent_id].boundary_local_indices
                    ]
                    for agent_id in range(int(partitions.k))
                ],
                dim=0,
            )
        else:
            stacked_local = torch.zeros(
                (0, self.graph_layers.config.hidden_dim), device=self.device, dtype=torch.float32
            )
        boundary_messages_conflict, boundary_messages_benefit = self._message_tables(
            boundary_global,
            static.owner,
            boundary_states,
            messages,
            row_map=topology.boundary_row,
        )
        boundary_structural = topology.boundary_structural_base
        if topology.boundary_selected_column is not None:
            boundary_structural = boundary_structural.clone()
            boundary_structural[:, topology.boundary_selected_column] = _tensor(
                selected_mask[boundary_global], self.device
            )
        agent_contexts = torch.stack(
            [local_states[agent_id].agent_context for agent_id in range(int(partitions.k))], dim=0
        )
        with torch.no_grad():
            boundary_graph_state = self.graph_layers.encode_boundary(
                full_state=full_state,
                global_node_ids=topology.boundary_global_tensor,
                agent_ids=topology.boundary_owner_tensor,
                local_node_embeddings=stacked_local,
                agent_contexts=agent_contexts,
                structural_features=boundary_structural,
                benefit_features=topology.boundary_benefit,
                conflict_features=topology.boundary_conflict,
                cut_edge_index=topology.cut_edge_index,
                cut_edge_type=topology.cut_edge_type,
                cut_edge_features=topology.cut_edge_features,
                message_conflict=_tensor(
                    _fit_columns(boundary_messages_conflict, self.graph_layers.config.message_conflict_dim), self.device
                ),
                message_benefit=_tensor(
                    _fit_columns(boundary_messages_benefit, self.graph_layers.config.message_benefit_dim), self.device
                ),
                mode=parsed,
                shuffle_index=(
                    topology.boundary_shuffle_index
                    if parsed == FeatureMode.SHUFFLED
                    else None
                ),
                undirected=True,
            )
        cache_key = _stable_digest(
            {
                "full": full_key,
                "states": {str(key): value.state_hash for key, value in boundary_states.items()},
                "messages": {str(key): value.message_hash for key, value in messages.items()},
            }
        )
        return PreparedTriGraph(
            mode=parsed,
            full_state=full_state,
            local_states=local_states,
            boundary_state=boundary_graph_state,
            boundary_indices=topology.boundary_indices,
            boundary_orders=boundary_orders,
            boundary_states=dict(boundary_states),
            boundary_messages=messages,
            incumbent_selected=selected,
            benefit_scale=max(1.0, _scale(static.weights)),
            cache_key=cache_key,
        )

    def rank_intents(
        self,
        prepared: PreparedTriGraph,
        agent_id: int,
        intents: Sequence[ProposalIntent],
    ) -> List[RankedIntent]:
        """Return deterministic graph_layers ranking using no revealed candidate label."""

        agent_id = int(agent_id)
        if agent_id not in prepared.local_states:
            raise ValueError("agent_id was not prepared")
        if not intents:
            return []
        order = prepared.boundary_orders[agent_id]
        state = prepared.boundary_states[agent_id]
        if tuple(state.boundary_order) != tuple(order):
            raise ValueError("BoundaryState order differs from CandidateGenerator order")
        order_set = set(order)
        requested = []
        structural = []
        benefit = []
        conflict = []
        raw_weights = np.asarray(
            [float(state.raw_benefit[index]) for index in range(len(order))], dtype=np.float32
        )
        conflict_map = state.value_map("conflict_score")
        for intent in intents:
            if int(intent.agent_id) != agent_id:
                raise ValueError("intent belongs to another agent")
            if tuple(intent.boundary_order) != tuple(order):
                raise ValueError("intent boundary order differs from prepared order")
            selected_set = set(int(node) for node in intent.requested_selected_boundary)
            if not selected_set.issubset(order_set):
                raise ValueError("intent selects a node outside its boundary")
            requested.append([node in selected_set for node in order])
            added = set(int(node) for node in intent.added_ids)
            removed = set(int(node) for node in intent.removed_ids)
            width = float(max(1, len(order)))
            structural.append(
                [len(added) / width, len(removed) / width, len(selected_set) / width, (len(added) + len(removed)) / width]
            )
            added_value = sum(float(state.raw_benefit[order.index(node)]) for node in added)
            removed_value = sum(float(state.raw_benefit[order.index(node)]) for node in removed)
            benefit.append(
                [
                    (added_value - removed_value) / prepared.benefit_scale,
                    added_value / prepared.benefit_scale,
                    removed_value / prepared.benefit_scale,
                ]
            )
            conflict.append(
                [
                    sum(float(conflict_map.get(node, 0.0)) for node in added) / width,
                    sum(float(conflict_map.get(node, 0.0)) for node in removed) / width,
                    len(set(intent.hotspot_ids)) / width,
                ]
            )
        incumbent_bits = torch.tensor(
            [node in set(state.incumbent_selected) for node in order],
            dtype=torch.bool,
            device=self.device,
        )
        with torch.no_grad():
            scores = self.graph_layers.score_intents_prequery(
                local_state=prepared.local_states[agent_id],
                boundary_state=prepared.boundary_state,
                boundary_indices=prepared.boundary_indices[agent_id],
                requested_bits=torch.tensor(requested, dtype=torch.bool, device=self.device),
                incumbent_bits=incumbent_bits,
                raw_arc_benefit=torch.tensor(raw_weights, dtype=torch.float32, device=self.device),
                benefit_scale=prepared.benefit_scale,
                structural_features=_tensor(
                    _fit_columns(np.asarray(structural, dtype=np.float32), self.graph_layers.config.candidate_struct_dim), self.device
                ),
                benefit_features=_tensor(
                    _fit_columns(np.asarray(benefit, dtype=np.float32), self.graph_layers.config.candidate_benefit_dim), self.device
                ),
                conflict_features=_tensor(
                    _fit_columns(np.asarray(conflict, dtype=np.float32), self.graph_layers.config.candidate_conflict_dim), self.device
                ),
                mode=prepared.mode,
            )
        rows = []
        for index, intent in enumerate(intents):
            values = {
                "selection_logit": float(scores.selection_logit[index].item()),
                "predicted_local_benefit": float(scores.predicted_local_benefit[index].item()),
                "learned_externality": float(scores.learned_externality[index].item()),
                "predicted_best_completion_advantage": float(
                    scores.predicted_best_completion_advantage[index].item()
                ),
                "completion_probability": float(scores.completion_probability[index].item()),
                "uncertainty": float(scores.uncertainty[index].item()),
                "raw_arc_benefit_proxy": float(scores.raw_arc_benefit_proxy[index].item()),
            }
            rows.append((intent, values))
        rows.sort(key=lambda item: (-item[1]["selection_logit"], str(item[0].intent_hash)))
        ranked = []
        for rank, (intent, values) in enumerate(rows, 1):
            score_hash = _stable_digest(
                {
                    "model_hash": self.metadata.model_hash,
                    "prepared": prepared.cache_key,
                    "intent_hash": intent.intent_hash,
                    "values": values,
                }
            )
            ranked.append(RankedIntent(rank=rank, intent=intent, score_hash=score_hash, **values))
        return ranked


__all__ = [
    "ModelMetadata",
    "PreparedTriGraph",
    "RankedIntent",
    "TriGraphPolicy",
]
