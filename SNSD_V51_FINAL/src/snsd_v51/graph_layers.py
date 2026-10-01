"""Lightweight TriGraph encoders and scoring heads for SNSD v1.5.

The module intentionally depends only on PyTorch.  In particular, it does not
use :mod:`torch_geometric` or materialise a doubled, directed copy of the C3
conflict graph.  ``RelationMeanConv`` aggregates one categorical relation at a
time from a canonical undirected edge list.  Its largest persistent activation
is therefore ``O(|V| * hidden_dim)`` per relation instead of
``O(|E| * hidden_dim)``.  The default hidden size of 64 is designed for the C3
graph (69,923 nodes and 979,960 canonical edges) on a 32 GB device.

The public API follows the three graph views in ``V1_5_METHOD_SPEC.md``:

``encode_full``
    Encode the public complete conflict graph and the accepted incumbent.
``encode_local``
    Encode an agent-owned core together with its read-only one-hop halo and
    expose separate core, halo, and boundary pools.
``encode_boundary``
    Fuse full/local representations with candidate-independent neighbour
    messages and propagate them only over the canonical cut-edge graph.

Candidate scoring keeps exact local benefit separate from all learned values.
The exact value is detached and copied to the output unchanged; the graph_layers only
learns externality, completion, uncertainty, and ranking residuals.  Thus the
interpretable predicted best-completion advantage is always
``exact_local_benefit + learned_externality``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional


Tensor = torch.Tensor


class FeatureMode(str, Enum):
    """Feature/communication ablations used by the v1.5 protocol.

    ``ZERO`` means zero *dynamic peer messages*.  It deliberately retains the
    public static graph, accepted incumbent, and local observations, matching
    the equal-compute ``NoPeerComm`` control.  Exact feasibility constraints
    and exact local benefit are outside these learned feature masks.
    """

    FULL = "Full"
    NO_GLOBAL = "NoGlobal"
    CONFLICT_ONLY = "ConflictOnly"
    BENEFIT_ONLY = "BenefitOnly"
    ZERO = "Zero"
    SHUFFLED = "Shuffled"

    @classmethod
    def parse(cls, value: Union["FeatureMode", str]) -> "FeatureMode":
        if isinstance(value, cls):
            return value
        normalized = "".join(character for character in str(value).lower() if character.isalnum())
        aliases = {
            "full": cls.FULL,
            "noglobal": cls.NO_GLOBAL,
            "conflictonly": cls.CONFLICT_ONLY,
            "benefitonly": cls.BENEFIT_ONLY,
            "zero": cls.ZERO,
            "nopeercomm": cls.ZERO,
            "nocomm": cls.ZERO,
            "shuffled": cls.SHUFFLED,
            "shuffledcomm": cls.SHUFFLED,
        }
        if normalized not in aliases:
            raise ValueError(
                "unknown feature mode %r; expected one of %s"
                % (value, ", ".join(item.value for item in cls))
            )
        return aliases[normalized]


@dataclass(frozen=True)
class TriGraphConfig:
    """Dimensions and lightweight architecture choices.

    Feature groups are supplied separately so that conflict/benefit ablations
    cannot accidentally leave information available through a concatenated
    tensor.  A dimension may be zero when a graph view has no such feature.
    """

    full_struct_dim: int = 8
    full_benefit_dim: int = 2
    full_conflict_dim: int = 8
    local_struct_dim: int = 4
    local_benefit_dim: int = 3
    local_conflict_dim: int = 3
    boundary_struct_dim: int = 2
    boundary_benefit_dim: int = 2
    boundary_conflict_dim: int = 2
    message_benefit_dim: int = 3
    message_conflict_dim: int = 3
    candidate_struct_dim: int = 0
    candidate_benefit_dim: int = 0
    candidate_conflict_dim: int = 0
    full_edge_dim: int = 1
    local_edge_dim: int = 1
    boundary_edge_dim: int = 1
    hidden_dim: int = 64
    num_agents: int = 6
    full_relations: int = 8
    local_relations: int = 8
    boundary_relations: int = 4
    full_layers: int = 2
    local_layers: int = 2
    boundary_layers: int = 1
    dropout: float = 0.1
    candidate_pool_chunk: int = 32

    def __post_init__(self) -> None:
        dimension_names = (
            "full_struct_dim",
            "full_benefit_dim",
            "full_conflict_dim",
            "local_struct_dim",
            "local_benefit_dim",
            "local_conflict_dim",
            "boundary_struct_dim",
            "boundary_benefit_dim",
            "boundary_conflict_dim",
            "message_benefit_dim",
            "message_conflict_dim",
            "candidate_struct_dim",
            "candidate_benefit_dim",
            "candidate_conflict_dim",
            "full_edge_dim",
            "local_edge_dim",
            "boundary_edge_dim",
        )
        for name in dimension_names:
            if int(getattr(self, name)) < 0:
                raise ValueError("%s must be non-negative" % name)
        positive_names = (
            "hidden_dim",
            "num_agents",
            "full_relations",
            "local_relations",
            "boundary_relations",
            "full_layers",
            "local_layers",
            "boundary_layers",
            "candidate_pool_chunk",
        )
        for name in positive_names:
            if int(getattr(self, name)) <= 0:
                raise ValueError("%s must be positive" % name)
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.full_struct_dim + self.full_benefit_dim + self.full_conflict_dim <= 0:
            raise ValueError("the full graph needs at least one node feature")


@dataclass(frozen=True)
class FullGraphState:
    node_embeddings: Tensor
    graph_embedding: Tensor
    mode: FeatureMode


@dataclass(frozen=True)
class LocalGraphState:
    agent_id: int
    global_node_ids: Tensor
    node_embeddings: Tensor
    core_pool: Tensor
    halo_pool: Tensor
    boundary_pool: Tensor
    agent_context: Tensor
    message_embeddings: Tensor
    mode: FeatureMode


@dataclass(frozen=True)
class BoundaryGraphState:
    global_node_ids: Tensor
    agent_ids: Tensor
    node_embeddings: Tensor
    agent_embeddings: Tensor
    message_embeddings: Tensor
    mode: FeatureMode


@dataclass(frozen=True)
class ArcScores:
    add_logit: Tensor
    delete_logit: Tensor


@dataclass(frozen=True)
class CandidateScores:
    """Candidate-level outputs with an auditable benefit decomposition."""

    exact_local_benefit: Tensor
    learned_externality: Tensor
    predicted_best_completion_advantage: Tensor
    completion_logit: Tensor
    completion_probability: Tensor
    uncertainty: Tensor
    learned_selection_residual: Tensor
    selection_logit: Tensor


def _require_matrix(name: str, value: Tensor, rows: Optional[int], columns: int) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError("%s must be a torch.Tensor" % name)
    if value.dim() != 2:
        raise ValueError("%s must have shape [rows, features]" % name)
    if rows is not None and int(value.shape[0]) != int(rows):
        raise ValueError("%s has %d rows, expected %d" % (name, value.shape[0], rows))
    if int(value.shape[1]) != int(columns):
        raise ValueError(
            "%s has feature dimension %d, expected %d"
            % (name, value.shape[1], columns)
        )
    return value


def _optional_feature(
    name: str,
    value: Optional[Tensor],
    rows: int,
    columns: int,
    reference: Tensor,
) -> Tensor:
    if value is None:
        return reference.new_zeros((rows, columns))
    checked = _require_matrix(name, value, rows, columns)
    if checked.device != reference.device:
        raise ValueError("%s and the reference tensor must be on the same device" % name)
    if checked.dtype != reference.dtype:
        raise ValueError("%s and the reference tensor must have the same dtype" % name)
    return checked


def _mask_benefit_conflict(
    benefit: Tensor, conflict: Tensor, mode: FeatureMode
) -> Tuple[Tensor, Tensor]:
    # Multiplication by zero deliberately preserves a zero-gradient autograd
    # path, which makes mask audits unambiguous (gradient == 0, not None).
    if mode == FeatureMode.CONFLICT_ONLY:
        benefit = benefit * 0.0
    elif mode == FeatureMode.BENEFIT_ONLY:
        conflict = conflict * 0.0
    return benefit, conflict


def _validate_shuffle_index(index: Tensor, rows: int, device: torch.device) -> Tensor:
    if not torch.is_tensor(index) or index.dim() != 1 or int(index.numel()) != rows:
        raise ValueError("shuffle_index must be a one-dimensional permutation of all rows")
    index = index.to(device=device, dtype=torch.long)
    if rows:
        if int(index.min().item()) < 0 or int(index.max().item()) >= rows:
            raise ValueError("shuffle_index contains an out-of-range row")
        if int(torch.unique(index).numel()) != rows:
            raise ValueError("shuffle_index must not repeat rows")
    return index


def _mean_max_pool(values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if values.dim() != 2:
        raise ValueError("values must have shape [nodes, hidden]")
    if mask is not None:
        if mask.dim() != 1 or int(mask.numel()) != int(values.shape[0]):
            raise ValueError("pool mask must have one value per node")
        selected = values[mask.to(device=values.device, dtype=torch.bool)]
    else:
        selected = values
    if int(selected.shape[0]) == 0:
        return values.new_zeros(2 * int(values.shape[1]))
    return torch.cat((selected.mean(dim=0), selected.max(dim=0).values), dim=0)


def _segment_mean(values: Tensor, segment_ids: Tensor, segments: int) -> Tensor:
    if segment_ids.dim() != 1 or int(segment_ids.numel()) != int(values.shape[0]):
        raise ValueError("segment_ids must have one entry per row")
    ids = segment_ids.to(device=values.device, dtype=torch.long)
    result = values.new_zeros((segments, int(values.shape[1])))
    counts = values.new_zeros((segments, 1))
    if int(values.shape[0]):
        result.index_add_(0, ids, values)
        counts.index_add_(0, ids, values.new_ones((int(values.shape[0]), 1)))
    return result / counts.clamp_min(1.0)


class RelationMeanConv(nn.Module):
    """Memory-bounded relation-aware mean GraphSAGE convolution.

    ``edge_index`` is normally the canonical undirected list with shape
    ``[2, E]``.  Both directions are aggregated without constructing a
    ``[2, 2E]`` tensor.  Since a linear transform commutes with a mean, raw
    neighbour features are first aggregated per relation and are transformed
    only at node level.  Optional continuous edge features produce a positive
    scalar gate and therefore do not create an ``E x hidden`` message tensor
    for all relations at once.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relations: int,
        edge_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_relations = int(num_relations)
        self.edge_feature_dim = int(edge_feature_dim)
        self.self_linear = nn.Linear(self.input_dim, self.output_dim)
        self.relation_linears = nn.ModuleList(
            nn.Linear(self.input_dim, self.output_dim, bias=False)
            for _ in range(self.num_relations)
        )
        if self.edge_feature_dim:
            self.edge_gates = nn.ModuleList(
                nn.Linear(self.edge_feature_dim, 1, bias=False)
                for _ in range(self.num_relations)
            )
        else:
            self.edge_gates = None

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
        undirected: bool = True,
    ) -> Tensor:
        _require_matrix("node_features", node_features, None, self.input_dim)
        if not torch.is_tensor(edge_index) or edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long indices")
        if edge_index.device != node_features.device:
            raise ValueError("edge_index and node_features must be on the same device")
        edge_count = int(edge_index.shape[1])
        if edge_type is None:
            edge_type = torch.zeros(edge_count, dtype=torch.long, device=node_features.device)
        elif (
            not torch.is_tensor(edge_type)
            or edge_type.dim() != 1
            or int(edge_type.numel()) != edge_count
        ):
            raise ValueError("edge_type must have one integer relation per edge")
        else:
            edge_type = edge_type.to(device=node_features.device, dtype=torch.long)
        if edge_count:
            minimum = int(edge_type.min().item())
            maximum = int(edge_type.max().item())
            if minimum < 0 or maximum >= self.num_relations:
                raise ValueError("edge_type contains a relation outside the configured range")

        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = node_features.new_zeros((edge_count, self.edge_feature_dim))
            else:
                edge_features = _require_matrix(
                    "edge_features", edge_features, edge_count, self.edge_feature_dim
                )
        elif edge_features is not None and int(edge_features.numel()):
            raise ValueError("edge features were supplied but edge_feature_dim is zero")

        output = self.self_linear(node_features)
        if not edge_count:
            return output

        all_sources = edge_index[0]
        all_targets = edge_index[1]
        for relation in range(self.num_relations):
            positions = torch.nonzero(edge_type == relation, as_tuple=False).flatten()
            if not int(positions.numel()):
                continue
            sources = all_sources.index_select(0, positions)
            targets = all_targets.index_select(0, positions)
            if self.edge_gates is None:
                weights = node_features.new_ones((int(positions.numel()), 1))
            else:
                assert edge_features is not None
                relation_edge_features = edge_features.index_select(0, positions)
                # Twice the sigmoid starts close to a neutral weight of one and
                # remains positive, which keeps the denominator well defined.
                weights = 2.0 * torch.sigmoid(
                    self.edge_gates[relation](relation_edge_features)
                )

            aggregate = node_features.new_zeros(node_features.shape)
            degree = node_features.new_zeros((int(node_features.shape[0]), 1))
            aggregate.index_add_(
                0,
                targets,
                node_features.index_select(0, sources) * weights,
            )
            degree.index_add_(0, targets, weights)
            if undirected:
                aggregate.index_add_(
                    0,
                    sources,
                    node_features.index_select(0, targets) * weights,
                )
                degree.index_add_(0, sources, weights)
            neighbour_mean = aggregate / degree.clamp_min(1.0e-12)
            output = output + self.relation_linears[relation](neighbour_mean)
        return output


class _GraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relations: int,
        edge_feature_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.convolutions = nn.ModuleList(
            RelationMeanConv(
                hidden_dim,
                hidden_dim,
                num_relations,
                edge_feature_dim=edge_feature_dim,
            )
            for _ in range(layers)
        )
        self.normalizations = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout = float(dropout)

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
        edge_type: Optional[Tensor],
        edge_features: Optional[Tensor],
        undirected: bool,
    ) -> Tensor:
        hidden = functional.gelu(self.input_projection(features))
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            update = convolution(
                hidden,
                edge_index,
                edge_type=edge_type,
                edge_features=edge_features,
                undirected=undirected,
            )
            update = functional.gelu(update)
            update = functional.dropout(update, p=self.dropout, training=self.training)
            hidden = normalization(hidden + update)
        return hidden


class BoundaryMessageEncoder(nn.Module):
    """Encode semantically separate conflict and benefit message channels."""

    def __init__(
        self,
        conflict_dim: int,
        benefit_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conflict_dim = int(conflict_dim)
        self.benefit_dim = int(benefit_dim)
        self.hidden_dim = int(hidden_dim)
        self.conflict_projection = (
            nn.Linear(self.conflict_dim, self.hidden_dim, bias=False)
            if self.conflict_dim
            else None
        )
        self.benefit_projection = (
            nn.Linear(self.benefit_dim, self.hidden_dim, bias=False)
            if self.benefit_dim
            else None
        )
        self.dropout = float(dropout)

    def forward(
        self,
        conflict: Optional[Tensor],
        benefit: Optional[Tensor],
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
        shuffle_index: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
    ) -> Tensor:
        parsed = FeatureMode.parse(mode)
        inferred = conflict if conflict is not None else benefit
        if inferred is None:
            if reference is None:
                raise ValueError("a reference tensor is required when both messages are absent")
            if reference.dim() != 2:
                raise ValueError("reference must have shape [rows, features]")
            rows = int(reference.shape[0])
            base = reference
        else:
            if inferred.dim() != 2:
                raise ValueError("message tensors must have shape [rows, features]")
            rows = int(inferred.shape[0])
            base = inferred
        conflict_tensor = _optional_feature(
            "message_conflict", conflict, rows, self.conflict_dim, base
        )
        benefit_tensor = _optional_feature(
            "message_benefit", benefit, rows, self.benefit_dim, base
        )
        benefit_tensor, conflict_tensor = _mask_benefit_conflict(
            benefit_tensor, conflict_tensor, parsed
        )

        if parsed == FeatureMode.SHUFFLED and rows:
            if shuffle_index is None:
                shuffle_index = torch.roll(
                    torch.arange(rows, device=base.device, dtype=torch.long), shifts=-1
                )
            permutation = _validate_shuffle_index(shuffle_index, rows, base.device)
            conflict_tensor = conflict_tensor.index_select(0, permutation)
            benefit_tensor = benefit_tensor.index_select(0, permutation)

        encoded = base.new_zeros((rows, self.hidden_dim))
        if self.conflict_projection is not None:
            encoded = encoded + self.conflict_projection(conflict_tensor)
        if self.benefit_projection is not None:
            encoded = encoded + self.benefit_projection(benefit_tensor)
        encoded = torch.tanh(encoded)
        encoded = functional.dropout(encoded, p=self.dropout, training=self.training)
        if parsed == FeatureMode.ZERO:
            encoded = encoded * 0.0
        return encoded


class TriGraphModel(nn.Module):
    """TriGraph-CBCR representation and multi-head scoring network."""

    def __init__(self, config: Optional[TriGraphConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else TriGraphConfig()
        cfg = self.config
        hidden = int(cfg.hidden_dim)

        full_input_dim = cfg.full_struct_dim + cfg.full_benefit_dim + cfg.full_conflict_dim
        self.full_encoder = _GraphEncoder(
            full_input_dim,
            hidden,
            cfg.full_relations,
            cfg.full_edge_dim,
            cfg.full_layers,
            cfg.dropout,
        )
        self.message_encoder = BoundaryMessageEncoder(
            cfg.message_conflict_dim,
            cfg.message_benefit_dim,
            hidden,
            cfg.dropout,
        )

        local_input_dim = (
            hidden
            + hidden
            + cfg.local_struct_dim
            + cfg.local_benefit_dim
            + cfg.local_conflict_dim
        )
        self.local_encoder = _GraphEncoder(
            local_input_dim,
            hidden,
            cfg.local_relations,
            cfg.local_edge_dim,
            cfg.local_layers,
            cfg.dropout,
        )
        bottleneck = max(4, hidden // 4)
        self.agent_adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, hidden),
            )
            for _ in range(cfg.num_agents)
        )
        self.agent_adapter_norms = nn.ModuleList(
            nn.LayerNorm(hidden) for _ in range(cfg.num_agents)
        )
        self.agent_identity = nn.Embedding(cfg.num_agents, hidden)
        self.local_context_projection = nn.Sequential(
            nn.Linear(8 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.local_context_norm = nn.LayerNorm(hidden)

        boundary_input_dim = (
            4 * hidden
            + cfg.boundary_struct_dim
            + cfg.boundary_benefit_dim
            + cfg.boundary_conflict_dim
        )
        self.boundary_encoder = _GraphEncoder(
            boundary_input_dim,
            hidden,
            cfg.boundary_relations,
            cfg.boundary_edge_dim,
            cfg.boundary_layers,
            cfg.dropout,
        )
        self.agent_from_boundary = nn.Linear(hidden, hidden, bias=False)
        self.node_from_agent = nn.Linear(hidden, hidden, bias=False)
        self.boundary_agent_norm = nn.LayerNorm(hidden)
        self.boundary_node_norm = nn.LayerNorm(hidden)

        self.add_head = self._scalar_head(hidden)
        self.delete_head = self._scalar_head(hidden)

        candidate_input_dim = (
            6 * hidden
            + cfg.candidate_struct_dim
            + cfg.candidate_benefit_dim
            + cfg.candidate_conflict_dim
            + 1
        )
        self.candidate_trunk = nn.Sequential(
            nn.Linear(candidate_input_dim, 2 * hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
        )
        self.externality_head = self._scalar_head(hidden)
        self.completion_head = self._scalar_head(hidden)
        self.uncertainty_head = self._scalar_head(hidden)
        self.selection_residual_head = self._scalar_head(hidden)

    @staticmethod
    def _scalar_head(hidden_dim: int) -> nn.Sequential:
        middle = max(4, hidden_dim // 2)
        return nn.Sequential(
            nn.Linear(hidden_dim, middle),
            nn.GELU(),
            nn.Linear(middle, 1),
        )

    def encode_messages(
        self,
        conflict: Optional[Tensor],
        benefit: Optional[Tensor],
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
        shuffle_index: Optional[Tensor] = None,
        reference: Optional[Tensor] = None,
    ) -> Tensor:
        """Public, independently auditable boundary-message encoder."""

        return self.message_encoder(
            conflict,
            benefit,
            mode=mode,
            shuffle_index=shuffle_index,
            reference=reference,
        )

    def encode_full(
        self,
        structural_features: Tensor,
        benefit_features: Optional[Tensor],
        conflict_features: Optional[Tensor],
        edge_index: Tensor,
        edge_type: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
        undirected: bool = True,
    ) -> FullGraphState:
        """Encode the complete public graph using all supplied canonical edges."""

        parsed = FeatureMode.parse(mode)
        structural = _require_matrix(
            "full_structural_features",
            structural_features,
            None,
            self.config.full_struct_dim,
        )
        rows = int(structural.shape[0])
        benefit = _optional_feature(
            "full_benefit_features",
            benefit_features,
            rows,
            self.config.full_benefit_dim,
            structural,
        )
        conflict = _optional_feature(
            "full_conflict_features",
            conflict_features,
            rows,
            self.config.full_conflict_dim,
            structural,
        )
        benefit, conflict = _mask_benefit_conflict(benefit, conflict, parsed)
        features = torch.cat((structural, benefit, conflict), dim=1)
        node_embeddings = self.full_encoder(
            features,
            edge_index,
            edge_type,
            edge_features,
            undirected=undirected,
        )
        graph_embedding = _mean_max_pool(node_embeddings)
        if parsed == FeatureMode.NO_GLOBAL:
            # Run the same encoder and then mask it.  This is more expensive
            # than an early exit but preserves equal compute in timing twins.
            node_embeddings = node_embeddings * 0.0
            graph_embedding = graph_embedding * 0.0
        return FullGraphState(node_embeddings, graph_embedding, parsed)

    def encode_local(
        self,
        agent_id: int,
        full_state: FullGraphState,
        global_node_ids: Tensor,
        structural_features: Optional[Tensor],
        benefit_features: Optional[Tensor],
        conflict_features: Optional[Tensor],
        edge_index: Tensor,
        core_mask: Tensor,
        halo_mask: Tensor,
        boundary_mask: Tensor,
        edge_type: Optional[Tensor] = None,
        edge_features: Optional[Tensor] = None,
        message_conflict: Optional[Tensor] = None,
        message_benefit: Optional[Tensor] = None,
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
        shuffle_index: Optional[Tensor] = None,
        undirected: bool = True,
    ) -> LocalGraphState:
        """Encode one controllable core plus its read-only halo."""

        parsed = FeatureMode.parse(mode)
        agent_id = int(agent_id)
        if not 0 <= agent_id < self.config.num_agents:
            raise ValueError("agent_id is outside the configured range")
        if global_node_ids.dim() != 1 or global_node_ids.dtype != torch.long:
            raise ValueError("global_node_ids must be a one-dimensional torch.long tensor")
        ids = global_node_ids.to(device=full_state.node_embeddings.device)
        rows = int(ids.numel())
        if rows:
            if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(
                full_state.node_embeddings.shape[0]
            ):
                raise ValueError("global_node_ids contains an out-of-range node")
        global_nodes = full_state.node_embeddings.index_select(0, ids)
        global_summary = full_state.graph_embedding
        if parsed == FeatureMode.NO_GLOBAL:
            global_nodes = global_nodes * 0.0
            global_summary = global_summary * 0.0

        reference = global_nodes
        structural = _optional_feature(
            "local_structural_features",
            structural_features,
            rows,
            self.config.local_struct_dim,
            reference,
        )
        benefit = _optional_feature(
            "local_benefit_features",
            benefit_features,
            rows,
            self.config.local_benefit_dim,
            reference,
        )
        conflict = _optional_feature(
            "local_conflict_features",
            conflict_features,
            rows,
            self.config.local_conflict_dim,
            reference,
        )
        benefit, conflict = _mask_benefit_conflict(benefit, conflict, parsed)
        messages = self.message_encoder(
            message_conflict,
            message_benefit,
            mode=parsed,
            shuffle_index=shuffle_index,
            reference=reference,
        )
        inputs = torch.cat((global_nodes, messages, structural, benefit, conflict), dim=1)
        hidden = self.local_encoder(
            inputs,
            edge_index,
            edge_type,
            edge_features,
            undirected=undirected,
        )
        hidden = self.agent_adapter_norms[agent_id](
            hidden + self.agent_adapters[agent_id](hidden)
        )

        core_pool = _mean_max_pool(hidden, core_mask)
        halo_pool = _mean_max_pool(hidden, halo_mask)
        boundary_pool = _mean_max_pool(hidden, boundary_mask)
        context_input = torch.cat(
            (global_summary, core_pool, halo_pool, boundary_pool), dim=0
        )
        identity_index = torch.tensor(agent_id, dtype=torch.long, device=hidden.device)
        context = self.local_context_projection(context_input)
        context = self.local_context_norm(context + self.agent_identity(identity_index))
        return LocalGraphState(
            agent_id=agent_id,
            global_node_ids=ids,
            node_embeddings=hidden,
            core_pool=core_pool,
            halo_pool=halo_pool,
            boundary_pool=boundary_pool,
            agent_context=context,
            message_embeddings=messages,
            mode=parsed,
        )

    def encode_boundary(
        self,
        full_state: FullGraphState,
        global_node_ids: Tensor,
        agent_ids: Tensor,
        local_node_embeddings: Tensor,
        agent_contexts: Tensor,
        structural_features: Optional[Tensor],
        benefit_features: Optional[Tensor],
        conflict_features: Optional[Tensor],
        cut_edge_index: Tensor,
        cut_edge_type: Optional[Tensor] = None,
        cut_edge_features: Optional[Tensor] = None,
        message_conflict: Optional[Tensor] = None,
        message_benefit: Optional[Tensor] = None,
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
        shuffle_index: Optional[Tensor] = None,
        undirected: bool = True,
    ) -> BoundaryGraphState:
        """Encode true cut edges and one membership exchange with agent tokens."""

        parsed = FeatureMode.parse(mode)
        if global_node_ids.dim() != 1 or global_node_ids.dtype != torch.long:
            raise ValueError("global_node_ids must be a one-dimensional torch.long tensor")
        ids = global_node_ids.to(device=full_state.node_embeddings.device)
        rows = int(ids.numel())
        if agent_ids.dim() != 1 or int(agent_ids.numel()) != rows:
            raise ValueError("agent_ids must have one value per boundary node")
        owners = agent_ids.to(device=ids.device, dtype=torch.long)
        if rows and (
            int(owners.min().item()) < 0
            or int(owners.max().item()) >= self.config.num_agents
        ):
            raise ValueError("agent_ids contains an out-of-range agent")
        _require_matrix(
            "local_node_embeddings",
            local_node_embeddings,
            rows,
            self.config.hidden_dim,
        )
        _require_matrix(
            "agent_contexts",
            agent_contexts,
            self.config.num_agents,
            self.config.hidden_dim,
        )
        global_nodes = full_state.node_embeddings.index_select(0, ids)
        if parsed == FeatureMode.NO_GLOBAL:
            global_nodes = global_nodes * 0.0
        reference = local_node_embeddings
        structural = _optional_feature(
            "boundary_structural_features",
            structural_features,
            rows,
            self.config.boundary_struct_dim,
            reference,
        )
        benefit = _optional_feature(
            "boundary_benefit_features",
            benefit_features,
            rows,
            self.config.boundary_benefit_dim,
            reference,
        )
        conflict = _optional_feature(
            "boundary_conflict_features",
            conflict_features,
            rows,
            self.config.boundary_conflict_dim,
            reference,
        )
        benefit, conflict = _mask_benefit_conflict(benefit, conflict, parsed)
        messages = self.message_encoder(
            message_conflict,
            message_benefit,
            mode=parsed,
            shuffle_index=shuffle_index,
            reference=reference,
        )
        owner_context = agent_contexts.index_select(0, owners)
        inputs = torch.cat(
            (
                global_nodes,
                local_node_embeddings,
                owner_context,
                messages,
                structural,
                benefit,
                conflict,
            ),
            dim=1,
        )
        hidden = self.boundary_encoder(
            inputs,
            cut_edge_index,
            cut_edge_type,
            cut_edge_features,
            undirected=undirected,
        )

        membership_pool = _segment_mean(hidden, owners, self.config.num_agents)
        updated_agents = self.boundary_agent_norm(
            agent_contexts + self.agent_from_boundary(membership_pool)
        )
        hidden = self.boundary_node_norm(
            hidden + self.node_from_agent(updated_agents.index_select(0, owners))
        )
        return BoundaryGraphState(
            global_node_ids=ids,
            agent_ids=owners,
            node_embeddings=hidden,
            agent_embeddings=updated_agents,
            message_embeddings=messages,
            mode=parsed,
        )

    def score_arcs(self, boundary_state: BoundaryGraphState) -> ArcScores:
        """Return add/delete logits; canonical feasibility remains external."""

        hidden = boundary_state.node_embeddings
        return ArcScores(
            add_logit=self.add_head(hidden).squeeze(-1),
            delete_logit=self.delete_head(hidden).squeeze(-1),
        )

    def _candidate_on_off_pool(
        self, embeddings: Tensor, candidate_bits: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if candidate_bits.dim() != 2 or int(candidate_bits.shape[1]) != int(
            embeddings.shape[0]
        ):
            raise ValueError("candidate_bits must have shape [candidates, boundary_nodes]")
        bits = candidate_bits.to(device=embeddings.device)
        if bits.dtype != torch.bool:
            if bool(torch.any((bits != 0) & (bits != 1)).item()):
                raise ValueError("candidate_bits must contain only zeros and ones")
            bits = bits.to(dtype=torch.bool)

        on_parts = []
        off_parts = []
        chunk_size = int(self.config.candidate_pool_chunk)
        negative = torch.finfo(embeddings.dtype).min
        for start in range(0, int(bits.shape[0]), chunk_size):
            local_bits = bits[start : start + chunk_size]
            as_float = local_bits.to(dtype=embeddings.dtype)
            off_float = 1.0 - as_float
            on_mean = as_float.matmul(embeddings) / as_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            off_mean = off_float.matmul(embeddings) / off_float.sum(dim=1, keepdim=True).clamp_min(1.0)

            expanded = embeddings.unsqueeze(0)
            on_max = expanded.masked_fill(~local_bits.unsqueeze(-1), negative).max(dim=1).values
            off_max = expanded.masked_fill(local_bits.unsqueeze(-1), negative).max(dim=1).values
            on_empty = ~local_bits.any(dim=1)
            off_empty = local_bits.all(dim=1)
            if bool(on_empty.any().item()):
                on_max = torch.where(on_empty.unsqueeze(1), torch.zeros_like(on_max), on_max)
            if bool(off_empty.any().item()):
                off_max = torch.where(off_empty.unsqueeze(1), torch.zeros_like(off_max), off_max)
            on_parts.append(torch.cat((on_mean, on_max), dim=1))
            off_parts.append(torch.cat((off_mean, off_max), dim=1))
        if not on_parts:
            width = 2 * int(embeddings.shape[1])
            empty = embeddings.new_zeros((0, width))
            return empty, empty
        return torch.cat(on_parts, dim=0), torch.cat(off_parts, dim=0)

    def score_candidates(
        self,
        local_state: LocalGraphState,
        boundary_state: BoundaryGraphState,
        boundary_indices: Tensor,
        candidate_bits: Tensor,
        exact_local_benefit: Tensor,
        structural_features: Optional[Tensor] = None,
        benefit_features: Optional[Tensor] = None,
        conflict_features: Optional[Tensor] = None,
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
    ) -> CandidateScores:
        """Score one agent's candidate table without re-estimating local benefit.

        ``exact_local_benefit`` and learned externality must already use the
        same robust normalization.  The exact tensor is detached before it is
        included in the learned readout and is returned unchanged.
        """

        parsed = FeatureMode.parse(mode)
        if boundary_indices.dim() != 1 or boundary_indices.dtype != torch.long:
            raise ValueError("boundary_indices must be a one-dimensional torch.long tensor")
        indices = boundary_indices.to(device=boundary_state.node_embeddings.device)
        boundary_embeddings = boundary_state.node_embeddings.index_select(0, indices)
        candidate_count = int(candidate_bits.shape[0])
        on_pool, off_pool = self._candidate_on_off_pool(
            boundary_embeddings, candidate_bits
        )

        exact = exact_local_benefit
        if exact.dim() == 2 and int(exact.shape[1]) == 1:
            exact = exact.squeeze(1)
        if exact.dim() != 1 or int(exact.numel()) != candidate_count:
            raise ValueError("exact_local_benefit must contain one scalar per candidate")
        exact = exact.to(
            device=boundary_embeddings.device, dtype=boundary_embeddings.dtype
        ).detach()

        reference = on_pool
        structural = _optional_feature(
            "candidate_structural_features",
            structural_features,
            candidate_count,
            self.config.candidate_struct_dim,
            reference,
        )
        benefit = _optional_feature(
            "candidate_benefit_features",
            benefit_features,
            candidate_count,
            self.config.candidate_benefit_dim,
            reference,
        )
        conflict = _optional_feature(
            "candidate_conflict_features",
            conflict_features,
            candidate_count,
            self.config.candidate_conflict_dim,
            reference,
        )
        benefit, conflict = _mask_benefit_conflict(benefit, conflict, parsed)
        local_context = local_state.agent_context.unsqueeze(0).expand(candidate_count, -1)
        agent_context = boundary_state.agent_embeddings[
            int(local_state.agent_id)
        ].unsqueeze(0).expand(candidate_count, -1)
        readout = torch.cat(
            (
                local_context,
                agent_context,
                on_pool,
                off_pool,
                structural,
                benefit,
                conflict,
                exact.unsqueeze(1),
            ),
            dim=1,
        )
        hidden = self.candidate_trunk(readout)
        externality = self.externality_head(hidden).squeeze(-1)
        completion_logit = self.completion_head(hidden).squeeze(-1)
        uncertainty = functional.softplus(
            self.uncertainty_head(hidden).squeeze(-1)
        ) + 1.0e-6
        selection_residual = self.selection_residual_head(hidden).squeeze(-1)
        advantage = exact + externality
        selection_logit = advantage + selection_residual
        return CandidateScores(
            exact_local_benefit=exact,
            learned_externality=externality,
            predicted_best_completion_advantage=advantage,
            completion_logit=completion_logit,
            completion_probability=torch.sigmoid(completion_logit),
            uncertainty=uncertainty,
            learned_selection_residual=selection_residual,
            selection_logit=selection_logit,
        )

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


__all__ = [
    "ArcScores",
    "BoundaryGraphState",
    "BoundaryMessageEncoder",
    "CandidateScores",
    "FeatureMode",
    "FullGraphState",
    "LocalGraphState",
    "RelationMeanConv",
    "TriGraphConfig",
    "TriGraphModel",
]
