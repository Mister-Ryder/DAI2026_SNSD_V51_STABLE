"""Leakage-safe pre-query scoring extension for the v1.5 TriGraph graph_layers.

``graph_layers.py`` contains the graph encoders and the post-reveal candidate heads.
Those post-reveal heads intentionally accept an exact local worker gain for
training and audit.  They cannot be used to choose which of 24 unqueried
intents receive three worker calls.  This module adds a separate acquisition
path whose inputs are all available before label reveal:

* requested complete boundary bits and their add/remove delta from incumbent;
* an exact *boundary-arc weight proxy* (not the unknown local-worker gain);
* public full/local/boundary graph representations and peer-message encoding;
* optional structural, benefit, and conflict intent features.

The output names keep the proxy, learned local response, externality, and
completion terms separate so a runner cannot accidentally log a proxy as a
realized ``Delta W`` label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .graph_layers import (
    ArcScores,
    BoundaryGraphState,
    BoundaryMessageEncoder,
    CandidateScores,
    FeatureMode,
    FullGraphState,
    LocalGraphState,
    RelationMeanConv,
    TriGraphConfig,
    TriGraphModel as _BaseTriGraphModel,
    _mask_benefit_conflict,
    _optional_feature,
)


Tensor = torch.Tensor


@dataclass(frozen=True)
class PrequeryIntentScores:
    """Acquisition outputs computed without a worker label.

    All learned benefit/externality values use the normalized scale selected by
    the runner.  ``raw_arc_benefit_proxy`` remains in original objective units
    for audit, while ``normalized_arc_benefit_proxy`` is the value seen by the
    network.
    """

    raw_arc_benefit_proxy: Tensor
    normalized_arc_benefit_proxy: Tensor
    learned_local_benefit_residual: Tensor
    predicted_local_benefit: Tensor
    learned_externality: Tensor
    predicted_best_completion_advantage: Tensor
    completion_logit: Tensor
    completion_probability: Tensor
    uncertainty: Tensor
    learned_selection_residual: Tensor
    selection_logit: Tensor


def _binary_matrix(name: str, value: Tensor, rows: Optional[int], columns: int) -> Tensor:
    if not torch.is_tensor(value) or value.dim() != 2:
        raise ValueError("%s must have shape [rows, boundary_nodes]" % name)
    if rows is not None and int(value.shape[0]) != int(rows):
        raise ValueError("%s has an unexpected number of rows" % name)
    if int(value.shape[1]) != int(columns):
        raise ValueError("%s has an unexpected boundary width" % name)
    if value.dtype == torch.bool:
        return value
    invalid = (value != 0) & (value != 1)
    if bool(torch.any(invalid).item()):
        raise ValueError("%s must contain only zeros and ones" % name)
    return value.to(dtype=torch.bool)


class TriGraphModel(_BaseTriGraphModel):
    """TriGraph graph_layers with a strictly pre-reveal intent acquisition head."""

    def __init__(self, config: Optional[TriGraphConfig] = None) -> None:
        super().__init__(config=config)
        cfg = self.config
        hidden = int(cfg.hidden_dim)
        # local token + boundary agent token + on/off/add/remove mean-max pools
        # = 10H, followed by optional intent features, one benefit proxy and
        # four explicit edit-structure statistics.
        input_dim = (
            10 * hidden
            + cfg.candidate_struct_dim
            + cfg.candidate_benefit_dim
            + cfg.candidate_conflict_dim
            + 5
        )
        self.prequery_trunk = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
        )
        self.prequery_benefit_residual_head = self._scalar_head(hidden)

    @staticmethod
    def _assert_state_mode(
        local_state: LocalGraphState,
        boundary_state: BoundaryGraphState,
        mode: FeatureMode,
    ) -> None:
        if local_state.mode != boundary_state.mode:
            raise ValueError("local and boundary states use different feature modes")
        if local_state.mode != mode:
            raise ValueError(
                "score mode must match the mode used to encode graph states; "
                "re-encode the states for this ablation"
            )

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
        """Post-reveal scorer with an additional ablation-consistency guard."""

        parsed = FeatureMode.parse(mode)
        self._assert_state_mode(local_state, boundary_state, parsed)
        return super().score_candidates(
            local_state=local_state,
            boundary_state=boundary_state,
            boundary_indices=boundary_indices,
            candidate_bits=candidate_bits,
            exact_local_benefit=exact_local_benefit,
            structural_features=structural_features,
            benefit_features=benefit_features,
            conflict_features=conflict_features,
            mode=parsed,
        )

    def score_intents_prequery(
        self,
        local_state: LocalGraphState,
        boundary_state: BoundaryGraphState,
        boundary_indices: Tensor,
        requested_bits: Tensor,
        incumbent_bits: Tensor,
        raw_arc_benefit: Tensor,
        benefit_scale: float = 1.0,
        structural_features: Optional[Tensor] = None,
        benefit_features: Optional[Tensor] = None,
        conflict_features: Optional[Tensor] = None,
        mode: Union[FeatureMode, str] = FeatureMode.FULL,
    ) -> PrequeryIntentScores:
        """Rank unqueried intents using only information available pre-reveal.

        Parameters
        ----------
        requested_bits:
            Complete materialized boundary signatures, shape ``[P, |B_i|]``.
            The caller should use the *projected* v1 ``BoundarySignature`` bits,
            not an intent that failed strict projection compliance.
        incumbent_bits:
            The verified incumbent boundary assignment, shape ``[|B_i|]``.
        raw_arc_benefit:
            Public raw weights aligned to the same boundary order.  Their
            add-minus-remove sum is a cheap proxy and is not a worker label.
        benefit_scale:
            A positive scale frozen before this proposal wave.  It may come
            from warm-up labels or fixed problem statistics, never current
            unrevealed worker outcomes.
        """

        parsed = FeatureMode.parse(mode)
        self._assert_state_mode(local_state, boundary_state, parsed)
        if not torch.is_tensor(boundary_indices) or boundary_indices.dim() != 1:
            raise ValueError("boundary_indices must be one-dimensional")
        indices = boundary_indices.to(
            device=boundary_state.node_embeddings.device, dtype=torch.long
        )
        width = int(indices.numel())
        if width == 0:
            raise ValueError("pre-query scoring requires a non-empty boundary")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(
            boundary_state.node_embeddings.shape[0]
        ):
            raise ValueError("boundary_indices contains an out-of-range row")
        owners = boundary_state.agent_ids.index_select(0, indices)
        if bool(torch.any(owners != int(local_state.agent_id)).item()):
            raise ValueError("boundary_indices contains nodes owned by another agent")

        boundary_embeddings = boundary_state.node_embeddings.index_select(0, indices)
        requested = _binary_matrix("requested_bits", requested_bits, None, width).to(
            device=boundary_embeddings.device
        )
        proposal_count = int(requested.shape[0])
        if not torch.is_tensor(incumbent_bits):
            raise TypeError("incumbent_bits must be a torch.Tensor")
        if incumbent_bits.dim() == 1:
            if int(incumbent_bits.numel()) != width:
                raise ValueError("incumbent_bits has an unexpected boundary width")
            incumbent_matrix = incumbent_bits.unsqueeze(0).expand(proposal_count, -1)
        elif incumbent_bits.dim() == 2 and tuple(incumbent_bits.shape) == (
            proposal_count,
            width,
        ):
            incumbent_matrix = incumbent_bits
        else:
            raise ValueError("incumbent_bits must have shape [B] or [P, B]")
        incumbent = _binary_matrix(
            "incumbent_bits", incumbent_matrix, proposal_count, width
        ).to(device=boundary_embeddings.device)

        if not torch.is_tensor(raw_arc_benefit) or raw_arc_benefit.dim() != 1:
            raise ValueError("raw_arc_benefit must have one value per boundary node")
        if int(raw_arc_benefit.numel()) != width:
            raise ValueError("raw_arc_benefit has an unexpected boundary width")
        raw_weights = raw_arc_benefit.to(
            device=boundary_embeddings.device, dtype=boundary_embeddings.dtype
        ).detach()
        scale = float(benefit_scale)
        if not scale > 0.0:
            raise ValueError("benefit_scale must be positive")

        added = requested & ~incumbent
        removed = ~requested & incumbent
        requested_on_pool, requested_off_pool = self._candidate_on_off_pool(
            boundary_embeddings, requested
        )
        added_pool, _ = self._candidate_on_off_pool(boundary_embeddings, added)
        removed_pool, _ = self._candidate_on_off_pool(boundary_embeddings, removed)

        added_float = added.to(dtype=boundary_embeddings.dtype)
        removed_float = removed.to(dtype=boundary_embeddings.dtype)
        raw_proxy = added_float.matmul(raw_weights) - removed_float.matmul(raw_weights)
        normalized_proxy = raw_proxy / scale

        reference = requested_on_pool
        structural = _optional_feature(
            "prequery_structural_features",
            structural_features,
            proposal_count,
            self.config.candidate_struct_dim,
            reference,
        )
        benefit = _optional_feature(
            "prequery_benefit_features",
            benefit_features,
            proposal_count,
            self.config.candidate_benefit_dim,
            reference,
        )
        conflict = _optional_feature(
            "prequery_conflict_features",
            conflict_features,
            proposal_count,
            self.config.candidate_conflict_dim,
            reference,
        )
        benefit, conflict = _mask_benefit_conflict(benefit, conflict, parsed)

        denominator = float(max(1, width))
        structure = torch.stack(
            (
                added_float.sum(dim=1) / denominator,
                removed_float.sum(dim=1) / denominator,
                requested.to(dtype=boundary_embeddings.dtype).sum(dim=1)
                / denominator,
                (added_float + removed_float).sum(dim=1) / denominator,
            ),
            dim=1,
        )
        local_context = local_state.agent_context.unsqueeze(0).expand(
            proposal_count, -1
        )
        agent_context = boundary_state.agent_embeddings[
            int(local_state.agent_id)
        ].unsqueeze(0).expand(proposal_count, -1)
        readout = torch.cat(
            (
                local_context,
                agent_context,
                requested_on_pool,
                requested_off_pool,
                added_pool,
                removed_pool,
                structural,
                benefit,
                conflict,
                normalized_proxy.unsqueeze(1),
                structure,
            ),
            dim=1,
        )
        hidden = self.prequery_trunk(readout)
        benefit_residual = self.prequery_benefit_residual_head(hidden).squeeze(-1)
        predicted_local = normalized_proxy + benefit_residual
        externality = self.externality_head(hidden).squeeze(-1)
        predicted_advantage = predicted_local + externality
        completion_logit = self.completion_head(hidden).squeeze(-1)
        uncertainty = functional.softplus(
            self.uncertainty_head(hidden).squeeze(-1)
        ) + 1.0e-6
        selection_residual = self.selection_residual_head(hidden).squeeze(-1)
        selection_logit = predicted_advantage + selection_residual
        return PrequeryIntentScores(
            raw_arc_benefit_proxy=raw_proxy,
            normalized_arc_benefit_proxy=normalized_proxy,
            learned_local_benefit_residual=benefit_residual,
            predicted_local_benefit=predicted_local,
            learned_externality=externality,
            predicted_best_completion_advantage=predicted_advantage,
            completion_logit=completion_logit,
            completion_probability=torch.sigmoid(completion_logit),
            uncertainty=uncertainty,
            learned_selection_residual=selection_residual,
            selection_logit=selection_logit,
        )

    # Runner-facing name that makes the acquisition phase explicit.
    score_proposals_prequery = score_intents_prequery


TriGraphModelV2 = TriGraphModel


__all__ = [
    "ArcScores",
    "BoundaryGraphState",
    "BoundaryMessageEncoder",
    "CandidateScores",
    "FeatureMode",
    "FullGraphState",
    "LocalGraphState",
    "PrequeryIntentScores",
    "RelationMeanConv",
    "TriGraphConfig",
    "TriGraphModel",
    "TriGraphModelV2",
]
