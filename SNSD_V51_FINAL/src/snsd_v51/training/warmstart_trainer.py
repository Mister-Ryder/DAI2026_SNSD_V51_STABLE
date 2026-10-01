"""Runnable frozen-encoder warm-start trainer for the v1.5 TriGraph policy.

This first offline checkpoint deliberately reuses ``runtime.prepare``.  That
inference adapter encodes full/local/boundary graphs under ``torch.no_grad()``,
so this trainer freezes every graph/message encoder and optimizes only the
pre-query candidate heads plus add/delete arc heads.  The resulting checkpoint
is a strict 456,075-parameter ``TriGraphModel`` checkpoint, but its manifest
must not be described as end-to-end encoder training.

The objective covers exact-chain local gain, externality, best-completion
advantage, pairwise ranking, completion, heteroscedastic uncertainty, and
masked identifiable arc pressure.  Missing pressure labels are omitted rather
than replaced by zeros.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional

from snsd_v51.model import FeatureMode, TriGraphConfig, TriGraphModel
from snsd_v51.messages import BoundaryState
from snsd_v51.runtime import PreparedTriGraph, TriGraphPolicy

from .offline_dataset import (
    ArcPressureLabel,
    DatasetSplit,
    OfflineDataError,
    OfflineDataset,
    OfflineExample,
)


EXPECTED_PARAMETER_COUNT = 456075
TRAINABLE_PREFIXES = (
    "prequery_trunk.",
    "prequery_benefit_residual_head.",
    "externality_head.",
    "completion_head.",
    "uncertainty_head.",
    "selection_residual_head.",
    "add_head.",
    "delete_head.",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    os.replace(str(temporary), str(path))


def _state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"SNSD-V15-OFFLINE-CHECKPOINT-v2\0")
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LossWeights:
    local_gain: float = 1.0
    externality: float = 1.0
    advantage: float = 1.0
    ranking: float = 0.5
    completion: float = 0.25
    uncertainty: float = 0.1
    arc_pressure: float = 0.5


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    patience: int = 12
    gradient_clip: float = 5.0
    seed: int = 20260803
    device: str = "cpu"
    mode: str = "Zero"
    loss_weights: LossWeights = LossWeights()


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: str
    model_manifest_path: str
    history_path: str
    best_epoch: int
    checkpoint_hash: str
    parameter_count: int
    trainable_parameter_count: int
    best_validation_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self.__dict__)


def _boundary_states(offline: OfflineDataset, seed: int) -> Dict[int, BoundaryState]:
    if offline.generators_by_seed is None or seed not in offline.generators_by_seed:
        raise OfflineDataError("missing reconstructed generators for seed %d" % seed)
    examples = [item for item in offline.examples if item.prequery.seed == seed]
    states = {}
    for agent_id, generator in enumerate(offline.generators_by_seed[seed]):
        rows = [item for item in examples if item.prequery.agent_id == agent_id]
        if not rows:
            raise OfflineDataError("seed %d has no examples for agent %d" % (seed, agent_id))
        anchor_hashes = {item.prequery.anchor_signature_hash for item in rows}
        anchors = {item.prequery.anchor_selected_boundary for item in rows}
        if len(anchor_hashes) != 1 or len(anchors) != 1:
            raise OfflineDataError("agent context has inconsistent warmup incumbent")
        anchor = next(iter(anchors))
        boundary = tuple(int(node) for node in generator.boundary)
        raw_benefit = {
            node: float(offline.dataset.arcs[node].weight) for node in boundary
        }
        conflict_score = {
            node: float(offline.graph.degree[node]) for node in boundary
        }
        zeros = {node: 0.0 for node in boundary}
        states[agent_id] = BoundaryState.from_maps(
            agent_id=agent_id,
            boundary_order=boundary,
            incumbent_selected=anchor,
            raw_benefit=raw_benefit,
            # Exact worker/factor labels are intentionally not supplied here.
            local_marginal=zeros,
            conflict_score=conflict_score,
            prior_opportunity_cost=zeros,
            uncertainty=zeros,
            prior_pressure=zeros,
        )
    return states


def prepare_frozen_zero_message_states(
    offline: OfflineDataset,
    graph_layers: TriGraphModel,
    device: str,
    mode: str = "Zero",
) -> Tuple[TriGraphPolicy, Dict[int, PreparedTriGraph]]:
    """Build detached inference states and make the frozen-encoder status explicit."""

    policy = TriGraphPolicy(graph_layers=graph_layers, device=device, seed=0)
    prepared = {}
    parsed = FeatureMode.parse(mode)
    for seed in offline.seeds:
        states = _boundary_states(offline, seed)
        prepared[seed] = policy.prepare(
            offline.dataset,
            offline.graph,
            offline.partitions,
            offline.generators_by_seed[seed],
            states,
            boundary_messages=None,
            incumbent=None,
            mode=parsed,
        )
    return policy, prepared


def _groups(examples: Sequence[OfflineExample]) -> Dict[Tuple[int, int], Tuple[OfflineExample, ...]]:
    grouped = {}  # type: Dict[Tuple[int, int], List[OfflineExample]]
    for item in examples:
        key = (item.prequery.seed, item.prequery.agent_id)
        grouped.setdefault(key, []).append(item)
    return {
        key: tuple(sorted(rows, key=lambda item: item.prequery.signature_hash))
        for key, rows in grouped.items()
    }


def _pressure_groups(
    rows: Sequence[ArcPressureLabel],
) -> Dict[Tuple[int, int], Tuple[ArcPressureLabel, ...]]:
    grouped = {}  # type: Dict[Tuple[int, int], List[ArcPressureLabel]]
    for item in rows:
        grouped.setdefault((item.seed, item.agent_id), []).append(item)
    return {
        key: tuple(sorted(values, key=lambda item: item.node_id))
        for key, values in grouped.items()
    }


def _score_group(
    graph_layers: TriGraphModel,
    prepared: PreparedTriGraph,
    rows: Sequence[OfflineExample],
) -> Any:
    if not rows:
        raise OfflineDataError("cannot score an empty group")
    agent_id = rows[0].prequery.agent_id
    order = prepared.boundary_orders[agent_id]
    order_set = set(order)
    requested = []
    for item in rows:
        if item.prequery.agent_id != agent_id:
            raise OfflineDataError("mixed agent IDs in one training group")
        if tuple(item.prequery.boundary_order) != tuple(order):
            raise OfflineDataError("teacher boundary order differs from runtime order")
        selected = set(item.prequery.selected_boundary)
        if not selected.issubset(order_set):
            raise OfflineDataError("candidate selects a node outside runtime boundary")
        requested.append([node in selected for node in order])
    anchor = set(rows[0].prequery.anchor_selected_boundary)
    for item in rows[1:]:
        if set(item.prequery.anchor_selected_boundary) != anchor:
            raise OfflineDataError("mixed incumbent contexts in one training group")
    device = prepared.boundary_state.node_embeddings.device
    raw_weights = torch.tensor(
        [float(prepared.boundary_states[agent_id].raw_benefit[index]) for index in range(len(order))],
        dtype=torch.float32,
        device=device,
    )
    return graph_layers.score_intents_prequery(
        local_state=prepared.local_states[agent_id],
        boundary_state=prepared.boundary_state,
        boundary_indices=prepared.boundary_indices[agent_id],
        requested_bits=torch.tensor(requested, dtype=torch.bool, device=device),
        incumbent_bits=torch.tensor(
            [node in anchor for node in order], dtype=torch.bool, device=device
        ),
        raw_arc_benefit=raw_weights,
        benefit_scale=prepared.benefit_scale,
        structural_features=None,
        benefit_features=None,
        conflict_features=None,
        mode=prepared.mode,
    )


def _masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if bool(mask.any().item()):
        return functional.smooth_l1_loss(prediction[mask], target[mask])
    return prediction.sum() * 0.0


def _ranking_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, int, int]:
    count = int(prediction.numel())
    if count < 2:
        return prediction.sum() * 0.0, 0, 0
    difference = target[:, None] - target[None, :]
    valid = mask[:, None] & mask[None, :]
    valid = valid & torch.triu(
        torch.ones((count, count), dtype=torch.bool, device=prediction.device), diagonal=1
    )
    valid = valid & (difference.abs() > 1.0e-9)
    if not bool(valid.any().item()):
        return prediction.sum() * 0.0, 0, 0
    sign = torch.sign(difference[valid])
    predicted_difference = (prediction[:, None] - prediction[None, :])[valid]
    margin = sign * predicted_difference
    correct = int((margin > 0.0).sum().item())
    pairs = int(margin.numel())
    return functional.softplus(-margin).mean(), correct, pairs


def _group_loss(
    graph_layers: TriGraphModel,
    prepared: PreparedTriGraph,
    rows: Sequence[OfflineExample],
    pressure_rows: Sequence[ArcPressureLabel],
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    scores = _score_group(graph_layers, prepared, rows)
    device = scores.selection_logit.device
    scale = float(prepared.benefit_scale)
    local = torch.tensor(
        [float(item.targets.local_gain or 0.0) / scale for item in rows],
        dtype=torch.float32,
        device=device,
    )
    local_mask = torch.tensor(
        [bool(item.targets.local_gain_mask) for item in rows],
        dtype=torch.bool,
        device=device,
    )
    externality = torch.tensor(
        [float(item.targets.externality or 0.0) / scale for item in rows],
        dtype=torch.float32,
        device=device,
    )
    externality_mask = torch.tensor(
        [bool(item.targets.externality_mask) for item in rows],
        dtype=torch.bool,
        device=device,
    )
    advantage = torch.tensor(
        [float(item.targets.best_completion_advantage or 0.0) / scale for item in rows],
        dtype=torch.float32,
        device=device,
    )
    advantage_mask = torch.tensor(
        [bool(item.targets.advantage_mask) for item in rows],
        dtype=torch.bool,
        device=device,
    )
    completion = torch.tensor(
        [float(item.targets.completion_feasible) for item in rows],
        dtype=torch.float32,
        device=device,
    )
    completion_mask = torch.tensor(
        [bool(item.targets.completion_mask) for item in rows],
        dtype=torch.bool,
        device=device,
    )
    local_loss = _masked_smooth_l1(scores.predicted_local_benefit, local, local_mask)
    externality_loss = _masked_smooth_l1(
        scores.learned_externality, externality, externality_mask
    )
    advantage_loss = _masked_smooth_l1(
        scores.predicted_best_completion_advantage, advantage, advantage_mask
    )
    rank_loss, rank_correct, rank_pairs = _ranking_loss(
        scores.selection_logit, advantage, advantage_mask
    )
    if bool(completion_mask.any().item()):
        completion_loss = functional.binary_cross_entropy_with_logits(
            scores.completion_logit[completion_mask], completion[completion_mask]
        )
    else:
        completion_loss = scores.completion_logit.sum() * 0.0
    if bool(advantage_mask.any().item()):
        sigma = scores.uncertainty[advantage_mask].clamp(1.0e-4, 100.0)
        residual = (
            scores.predicted_best_completion_advantage[advantage_mask]
            - advantage[advantage_mask]
        ).detach()
        uncertainty_loss = (0.5 * (residual / sigma).pow(2) + torch.log(sigma)).mean()
    else:
        uncertainty_loss = scores.uncertainty.sum() * 0.0

    agent_id = rows[0].prequery.agent_id
    arc_scores = graph_layers.score_arcs(prepared.boundary_state)
    indices = prepared.boundary_indices[agent_id]
    add = arc_scores.add_logit.index_select(0, indices)
    delete = arc_scores.delete_logit.index_select(0, indices)
    order = prepared.boundary_orders[agent_id]
    row_by_node = {item.node_id: item for item in pressure_rows}
    arc_predictions = []
    arc_targets = []
    for index, node_id in enumerate(order):
        label = row_by_node.get(int(node_id))
        if label is None or not label.pressure_identifiable:
            continue
        if label.pressure is None:
            raise OfflineDataError("identifiable pressure cannot be missing")
        if label.incumbent_bit:
            arc_predictions.append(delete[index])
            arc_targets.append(-float(label.pressure) / scale)
        else:
            arc_predictions.append(add[index])
            arc_targets.append(float(label.pressure) / scale)
    if arc_predictions:
        pressure_prediction = torch.stack(arc_predictions)
        pressure_target = torch.tensor(
            arc_targets, dtype=torch.float32, device=device
        )
        arc_loss = functional.smooth_l1_loss(pressure_prediction, pressure_target)
        pressure_mae = float(
            (pressure_prediction.detach() - pressure_target).abs().mean().item()
        )
    else:
        arc_loss = add.sum() * 0.0
        pressure_mae = float("nan")
    total = (
        float(weights.local_gain) * local_loss
        + float(weights.externality) * externality_loss
        + float(weights.advantage) * advantage_loss
        + float(weights.ranking) * rank_loss
        + float(weights.completion) * completion_loss
        + float(weights.uncertainty) * uncertainty_loss
        + float(weights.arc_pressure) * arc_loss
    )
    metrics = {
        "loss": float(total.detach().item()),
        "local_gain_loss": float(local_loss.detach().item()),
        "externality_loss": float(externality_loss.detach().item()),
        "advantage_loss": float(advantage_loss.detach().item()),
        "ranking_loss": float(rank_loss.detach().item()),
        "completion_loss": float(completion_loss.detach().item()),
        "uncertainty_loss": float(uncertainty_loss.detach().item()),
        "arc_pressure_loss": float(arc_loss.detach().item()),
        "rank_correct": rank_correct,
        "rank_pairs": rank_pairs,
        "pressure_count": len(arc_predictions),
        "pressure_mae_normalized": pressure_mae,
        "candidate_count": len(rows),
    }
    return total, metrics


def _aggregate(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"loss": float("nan"), "groups": 0.0}
    result = {"groups": float(len(rows))}
    sum_keys = ("rank_correct", "rank_pairs", "pressure_count", "candidate_count")
    for key in rows[0]:
        values = [float(row[key]) for row in rows if key in row]
        if key in sum_keys:
            result[key] = float(sum(values))
        else:
            finite = [value for value in values if math.isfinite(value)]
            result[key] = float(sum(finite) / len(finite)) if finite else float("nan")
    pairs = result.get("rank_pairs", 0.0)
    result["rank_accuracy"] = (
        result.get("rank_correct", 0.0) / pairs if pairs > 0.0 else float("nan")
    )
    return result


def _evaluate(
    graph_layers: TriGraphModel,
    prepared: Mapping[int, PreparedTriGraph],
    examples: Sequence[OfflineExample],
    pressures: Sequence[ArcPressureLabel],
    weights: LossWeights,
) -> Dict[str, float]:
    groups = _groups(examples)
    pressure_map = _pressure_groups(pressures)
    rows = []
    graph_layers.eval()
    with torch.no_grad():
        for key in sorted(groups):
            seed, _ = key
            _, metrics = _group_loss(
                graph_layers,
                prepared[seed],
                groups[key],
                pressure_map.get(key, ()),
                weights,
            )
            rows.append(metrics)
    return _aggregate(rows)


def train_head_warmstart(
    offline: OfflineDataset,
    split: DatasetSplit,
    output_dir: str,
    config: Optional[TrainerConfig] = None,
) -> TrainingResult:
    """Train leakage-safe heads and save a strict runtime-compatible checkpoint."""

    cfg = config if config is not None else TrainerConfig()
    destination = Path(output_dir).resolve()
    checkpoint_path = destination / "checkpoint.pt"
    if checkpoint_path.exists():
        raise OfflineDataError("refusing to overwrite existing checkpoint %s" % checkpoint_path)
    destination.mkdir(parents=True, exist_ok=True)
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model_config = TriGraphConfig()
    graph_layers = TriGraphModel(model_config)
    parameter_count = sum(int(value.numel()) for value in graph_layers.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise OfflineDataError(
            "strict architecture mismatch: expected %d parameters, got %d"
            % (EXPECTED_PARAMETER_COUNT, parameter_count)
        )
    for name, parameter in graph_layers.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)
    trainable_parameters = [value for value in graph_layers.parameters() if value.requires_grad]
    trainable_count = sum(int(value.numel()) for value in trainable_parameters)
    policy, prepared = prepare_frozen_zero_message_states(
        offline=offline,
        graph_layers=graph_layers,
        device=cfg.device,
        mode=cfg.mode,
    )
    graph_layers = policy.graph_layers
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    train_groups = _groups(split.train)
    train_pressure = _pressure_groups(split.train_pressures)
    if not train_groups:
        raise OfflineDataError("training split is empty")
    history = []
    best_state = copy.deepcopy(graph_layers.state_dict())
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        graph_layers.train()
        keys = list(train_groups)
        random.Random(int(cfg.seed) + epoch).shuffle(keys)
        training_rows = []
        for key in keys:
            seed, _ = key
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _group_loss(
                graph_layers,
                prepared[seed],
                train_groups[key],
                train_pressure.get(key, ()),
                cfg.loss_weights,
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite training loss at epoch %d" % epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(cfg.gradient_clip))
            optimizer.step()
            training_rows.append(metrics)
        train_metrics = _aggregate(training_rows)
        validation_metrics = _evaluate(
            graph_layers,
            prepared,
            split.validation,
            split.validation_pressures,
            cfg.loss_weights,
        )
        selection_loss = validation_metrics.get("loss", float("nan"))
        if not math.isfinite(selection_loss):
            selection_loss = train_metrics["loss"]
        improved = selection_loss < best_validation - 1.0e-9
        if improved:
            best_validation = float(selection_loss)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in graph_layers.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "improved": improved,
            }
        )
        if stale >= int(cfg.patience):
            break
    graph_layers.load_state_dict(best_state, strict=True)
    strict_probe = TriGraphModel(TriGraphConfig())
    strict_probe.load_state_dict(graph_layers.state_dict(), strict=True)
    state_hash = _state_dict_hash(graph_layers.state_dict())
    checkpoint = {
        "schema_version": 2,
        "state_dict": graph_layers.state_dict(),
        "model_config": asdict(model_config),
        "model_class": "snsd_v51.model.TriGraphModel",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "checkpoint_hash": state_hash,
        "training_mode": "frozen_runtime_v2_zero_message_embeddings_head_warmstart",
        "dataset_hash": offline.receipt.get("dataset_hash"),
        "split_manifest": dict(split.manifest),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
    }
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, str(temporary_checkpoint))
    os.replace(str(temporary_checkpoint), str(checkpoint_path))
    history_path = destination / "training_history.jsonl"
    _atomic_jsonl(history_path, history)
    test_metrics = _evaluate(
        graph_layers,
        prepared,
        split.test,
        split.test_pressures,
        cfg.loss_weights,
    )
    manifest = {
        "schema_version": 2,
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": state_hash,
        "strict_load_verified": True,
        "model_class": checkpoint["model_class"],
        "model_config": asdict(model_config),
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "trainable_parameter_count": trainable_count,
        "training_mode": checkpoint["training_mode"],
        "encoder_training": "frozen; runtime.prepare is torch.no_grad inference code",
        "message_training": "zero-message pretraining; message encoder remains frozen",
        "trainable_prefixes": list(TRAINABLE_PREFIXES),
        "prequery_objective_used": False,
        "teacher": offline.receipt,
        "split": dict(split.manifest),
        "trainer_config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "test_metrics": test_metrics,
        "completion_warning": (
            "warmup completion labels are single-class; retain the head for strict "
            "compatibility but do not claim discriminative completion performance"
            if bool(offline.receipt.get("completion_is_degenerate"))
            else None
        ),
        "pressure_policy": "loss includes only pressure_identifiable=true arcs",
    }
    manifest_path = destination / "model_manifest.json"
    _atomic_json(manifest_path, manifest)
    return TrainingResult(
        checkpoint_path=str(checkpoint_path),
        model_manifest_path=str(manifest_path),
        history_path=str(history_path),
        best_epoch=best_epoch,
        checkpoint_hash=state_hash,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        best_validation_loss=best_validation,
    )


__all__ = [
    "EXPECTED_PARAMETER_COUNT",
    "LossWeights",
    "TrainerConfig",
    "TrainingResult",
    "prepare_frozen_zero_message_states",
    "train_head_warmstart",
]
