"""Paired Real/Zero exact-message warm-start for the strict v1.5 graph_layers."""

from __future__ import annotations

import copy
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
from snsd_v51.messages import apply_message_intervention
from snsd_v51.runtime import PreparedTriGraph, TriGraphPolicy

from .message_teacher import MessageTeacherFold, zero_target_examples
from .offline_dataset import DatasetSplit, OfflineDataError, OfflineDataset
from .warmstart_trainer import (
    EXPECTED_PARAMETER_COUNT,
    TRAINABLE_PREFIXES,
    LossWeights,
    _boundary_states,
    _group_loss,
    _groups,
    _pressure_groups,
    _score_group,
    _state_dict_hash,
)


@dataclass(frozen=True)
class MessageTrainerConfig:
    epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    patience: int = 12
    gradient_clip: float = 5.0
    seed: int = 20260803
    device: str = "cpu"
    include_shuffled: bool = False
    zero_weight: float = 1.0
    shuffled_weight: float = 0.25
    paired_delta_weight: float = 1.0
    loss_weights: LossWeights = LossWeights()


@dataclass(frozen=True)
class MessageTrainingResult:
    checkpoint_path: str
    model_manifest_path: str
    history_path: str
    best_epoch: int
    checkpoint_hash: str
    parameter_count: int
    trainable_parameter_count: int
    best_validation_loss: float
    validation_real_rank_accuracy: Optional[float]
    validation_zero_rank_accuracy: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    os.replace(str(temporary), str(path))


def prepare_paired_message_states(
    offline: OfflineDataset,
    folds: Sequence[MessageTeacherFold],
    graph_layers: TriGraphModel,
    device: str,
    include_shuffled: bool = False,
) -> Tuple[TriGraphPolicy, Dict[Tuple[int, int, str], PreparedTriGraph]]:
    """Prepare detached Full-mode states for paired message interventions."""

    policy = TriGraphPolicy(graph_layers=graph_layers, device=device, seed=0)
    prepared = {}
    for fold in folds:
        states = _boundary_states(offline, fold.seed)
        variants = {"Real": dict(fold.messages)}
        variants["Zero"] = {
            agent_id: apply_message_intervention(
                message,
                "zero",
                experiment_seed=fold.seed,
                agent_id=agent_id,
                macro_round=fold.heldout_fold,
            )
            for agent_id, message in fold.messages.items()
        }
        if include_shuffled:
            variants["Shuffled"] = {
                agent_id: apply_message_intervention(
                    message,
                    "shuffled",
                    experiment_seed=fold.seed,
                    agent_id=agent_id,
                    macro_round=fold.heldout_fold,
                )
                for agent_id, message in fold.messages.items()
            }
        for variant, messages in variants.items():
            prepared[(fold.seed, fold.heldout_fold, variant)] = policy.prepare(
                offline.dataset,
                offline.graph,
                offline.partitions,
                offline.generators_by_seed[fold.seed],
                states,
                boundary_messages=messages,
                incumbent=None,
                mode=FeatureMode.FULL,
            )
    return policy, prepared


def _without_arc_weight(weights: LossWeights) -> LossWeights:
    return LossWeights(
        local_gain=weights.local_gain,
        externality=weights.externality,
        advantage=weights.advantage,
        ranking=weights.ranking,
        completion=weights.completion,
        uncertainty=weights.uncertainty,
        arc_pressure=0.0,
    )


def _paired_delta_loss(
    graph_layers: TriGraphModel,
    real: PreparedTriGraph,
    zero: PreparedTriGraph,
    rows: Sequence[Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    real_scores = _score_group(graph_layers, real, rows)
    zero_scores = _score_group(graph_layers, zero, rows)
    device = real_scores.selection_logit.device
    scale = float(real.benefit_scale)
    target = torch.tensor(
        [float(item.targets.externality or 0.0) / scale for item in rows],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.tensor(
        [bool(item.targets.externality_mask) for item in rows],
        dtype=torch.bool,
        device=device,
    )
    if bool(mask.any().item()):
        advantage_delta = (
            real_scores.predicted_best_completion_advantage
            - zero_scores.predicted_best_completion_advantage
        )
        selection_delta = real_scores.selection_logit - zero_scores.selection_logit
        head_delta = real_scores.learned_externality - zero_scores.learned_externality
        loss = (
            functional.smooth_l1_loss(advantage_delta[mask], target[mask])
            + 0.5 * functional.smooth_l1_loss(selection_delta[mask], target[mask])
            + 0.5 * functional.smooth_l1_loss(head_delta[mask], target[mask])
        )
        error = (advantage_delta[mask].detach() - target[mask]).abs()
        sign_mask = target[mask].abs() > 1.0e-9
        if bool(sign_mask.any().item()):
            sign_correct = int(
                (
                    torch.sign(advantage_delta[mask][sign_mask])
                    == torch.sign(target[mask][sign_mask])
                ).sum().item()
            )
            sign_count = int(sign_mask.sum().item())
        else:
            sign_correct = 0
            sign_count = 0
        metrics = {
            "paired_delta_loss": float(loss.detach().item()),
            "paired_delta_mae": float(error.mean().item()),
            "real_zero_delta_abs_mean": float(
                advantage_delta[mask].detach().abs().mean().item()
            ),
            "target_externality_abs_mean": float(target[mask].abs().mean().item()),
            "delta_sign_correct": float(sign_correct),
            "delta_sign_count": float(sign_count),
        }
        return loss, metrics
    zero_loss = real_scores.selection_logit.sum() * 0.0
    return zero_loss, {
        "paired_delta_loss": 0.0,
        "paired_delta_mae": float("nan"),
        "real_zero_delta_abs_mean": float("nan"),
        "target_externality_abs_mean": float("nan"),
        "delta_sign_correct": 0.0,
        "delta_sign_count": 0.0,
    }


def _one_group_loss(
    graph_layers: TriGraphModel,
    prepared: Mapping[Tuple[int, int, str], PreparedTriGraph],
    fold: MessageTeacherFold,
    rows: Sequence[Any],
    pressure_rows: Sequence[Any],
    config: MessageTrainerConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    key = (fold.seed, fold.heldout_fold)
    real_state = prepared[key + ("Real",)]
    zero_state = prepared[key + ("Zero",)]
    real_loss, real_metrics = _group_loss(
        graph_layers,
        real_state,
        rows,
        pressure_rows,
        config.loss_weights,
    )
    zero_rows = zero_target_examples(rows)
    zero_loss, zero_metrics = _group_loss(
        graph_layers,
        zero_state,
        zero_rows,
        (),
        _without_arc_weight(config.loss_weights),
    )
    paired_loss, paired_metrics = _paired_delta_loss(
        graph_layers, real_state, zero_state, rows
    )
    total = (
        real_loss
        + float(config.zero_weight) * zero_loss
        + float(config.paired_delta_weight) * paired_loss
    )
    shuffled_metrics = None
    if config.include_shuffled:
        shuffled_state = prepared[key + ("Shuffled",)]
        shuffled_rows = zero_target_examples(rows, source="paired_shuffled_message_v3")
        shuffled_loss, shuffled_metrics = _group_loss(
            graph_layers,
            shuffled_state,
            shuffled_rows,
            (),
            _without_arc_weight(config.loss_weights),
        )
        total = total + float(config.shuffled_weight) * shuffled_loss
    metrics = {
        "loss": float(total.detach().item()),
        "real_loss": real_metrics["loss"],
        "zero_loss": zero_metrics["loss"],
        "real_rank_correct": float(real_metrics["rank_correct"]),
        "real_rank_pairs": float(real_metrics["rank_pairs"]),
        "zero_rank_correct": float(zero_metrics["rank_correct"]),
        "zero_rank_pairs": float(zero_metrics["rank_pairs"]),
        "pressure_count": float(real_metrics["pressure_count"]),
        "candidate_count": float(real_metrics["candidate_count"]),
        "arc_pressure_loss": real_metrics["arc_pressure_loss"],
    }
    metrics.update(paired_metrics)
    if shuffled_metrics is not None:
        metrics["shuffled_loss"] = shuffled_metrics["loss"]
        metrics["shuffled_rank_correct"] = float(shuffled_metrics["rank_correct"])
        metrics["shuffled_rank_pairs"] = float(shuffled_metrics["rank_pairs"])
    return total, metrics


def _aggregate(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"loss": float("nan"), "groups": 0.0}
    sum_keys = {
        "real_rank_correct",
        "real_rank_pairs",
        "zero_rank_correct",
        "zero_rank_pairs",
        "shuffled_rank_correct",
        "shuffled_rank_pairs",
        "pressure_count",
        "candidate_count",
        "delta_sign_correct",
        "delta_sign_count",
    }
    keys = sorted(set(key for row in rows for key in row))
    result = {"groups": float(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if key in sum_keys:
            result[key] = float(sum(values))
        else:
            finite = [value for value in values if math.isfinite(value)]
            result[key] = float(sum(finite) / len(finite)) if finite else float("nan")
    for prefix in ("real", "zero", "shuffled"):
        pairs = result.get(prefix + "_rank_pairs", 0.0)
        result[prefix + "_rank_accuracy"] = (
            result.get(prefix + "_rank_correct", 0.0) / pairs
            if pairs > 0.0
            else float("nan")
        )
    sign_count = result.get("delta_sign_count", 0.0)
    result["delta_sign_accuracy"] = (
        result.get("delta_sign_correct", 0.0) / sign_count
        if sign_count > 0.0
        else float("nan")
    )
    return result


def _fold_groups(
    folds: Sequence[MessageTeacherFold], seeds: Sequence[int]
) -> Tuple[Tuple[MessageTeacherFold, int, Tuple[Any, ...]], ...]:
    wanted = set(int(value) for value in seeds)
    rows = []
    for fold in folds:
        if fold.seed not in wanted:
            continue
        for (_, agent_id), examples in sorted(_groups(fold.target_examples).items()):
            rows.append((fold, agent_id, examples))
    return tuple(rows)


def _evaluate(
    graph_layers: TriGraphModel,
    prepared: Mapping[Tuple[int, int, str], PreparedTriGraph],
    groups: Sequence[Tuple[MessageTeacherFold, int, Tuple[Any, ...]]],
    config: MessageTrainerConfig,
) -> Dict[str, float]:
    graph_layers.eval()
    metrics = []
    with torch.no_grad():
        for fold, agent_id, rows in groups:
            pressure = _pressure_groups(fold.arc_pressures).get(
                (fold.seed, agent_id), ()
            )
            _, row = _one_group_loss(
                graph_layers, prepared, fold, rows, pressure, config
            )
            metrics.append(row)
    return _aggregate(metrics)


def train_message_warmstart(
    offline: OfflineDataset,
    folds: Sequence[MessageTeacherFold],
    split: DatasetSplit,
    output_dir: str,
    config: Optional[MessageTrainerConfig] = None,
) -> MessageTrainingResult:
    cfg = config if config is not None else MessageTrainerConfig()
    destination = Path(output_dir).resolve()
    checkpoint_path = destination / "checkpoint_message_v3.pt"
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
    trainable = [parameter for parameter in graph_layers.parameters() if parameter.requires_grad]
    trainable_count = sum(int(value.numel()) for value in trainable)
    policy, prepared = prepare_paired_message_states(
        offline,
        folds,
        graph_layers,
        cfg.device,
        include_shuffled=cfg.include_shuffled,
    )
    graph_layers = policy.graph_layers
    train_seeds = tuple(int(value) for value in split.manifest["train_seeds"])
    validation_seeds = tuple(
        int(value) for value in split.manifest["validation_seeds"]
    )
    test_seeds = tuple(int(value) for value in split.manifest["test_seeds"])
    train_groups = _fold_groups(folds, train_seeds)
    validation_groups = _fold_groups(folds, validation_seeds)
    test_groups = _fold_groups(folds, test_seeds)
    if not train_groups:
        raise OfflineDataError("message training split is empty")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    best_state = copy.deepcopy(graph_layers.state_dict())
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    history = []
    for epoch in range(1, int(cfg.epochs) + 1):
        graph_layers.train()
        order = list(range(len(train_groups)))
        random.Random(int(cfg.seed) + epoch).shuffle(order)
        epoch_metrics = []
        for index in order:
            fold, agent_id, rows = train_groups[index]
            pressure = _pressure_groups(fold.arc_pressures).get(
                (fold.seed, agent_id), ()
            )
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _one_group_loss(
                graph_layers, prepared, fold, rows, pressure, cfg
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "non-finite message training loss at epoch %d" % epoch
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.gradient_clip))
            optimizer.step()
            epoch_metrics.append(metrics)
        training = _aggregate(epoch_metrics)
        validation = _evaluate(graph_layers, prepared, validation_groups, cfg)
        criterion = validation.get("loss", float("nan"))
        if not math.isfinite(criterion):
            criterion = training["loss"]
        improved = criterion < best_validation - 1.0e-9
        if improved:
            best_validation = float(criterion)
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
                "train": training,
                "validation": validation,
                "improved": improved,
            }
        )
        if stale >= int(cfg.patience):
            break
    graph_layers.load_state_dict(best_state, strict=True)
    strict_probe = TriGraphModel(TriGraphConfig())
    strict_probe.load_state_dict(graph_layers.state_dict(), strict=True)
    checkpoint_hash = _state_dict_hash(graph_layers.state_dict())
    validation = _evaluate(graph_layers, prepared, validation_groups, cfg)
    test = _evaluate(graph_layers, prepared, test_groups, cfg)
    checkpoint = {
        "schema_version": 3,
        "state_dict": graph_layers.state_dict(),
        "model_config": asdict(model_config),
        "model_class": "snsd_v51.model.TriGraphModel",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "checkpoint_hash": checkpoint_hash,
        "training_mode": (
            "frozen_encoder_crossfit_exact_teacher_real_zero_paired_message_warmstart_v3"
        ),
        "dataset_hash": offline.receipt.get("dataset_hash"),
        "split_manifest": dict(split.manifest),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, str(temporary))
    os.replace(str(temporary), str(checkpoint_path))
    history_path = destination / "training_history_message_v3.jsonl"
    _atomic_jsonl(history_path, history)
    manifest = {
        "schema_version": 3,
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "strict_load_verified": True,
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "trainable_parameter_count": trainable_count,
        "training_mode": checkpoint["training_mode"],
        "encoder_training": "frozen; runtime.prepare states are detached",
        "message_encoder_training": (
            "frozen random projection; heads trained on Real/Zero paired embeddings"
        ),
        "candidate_objective_in_own_message": False,
        "message_variants": (
            ["Real", "Zero", "Shuffled"]
            if cfg.include_shuffled
            else ["Real", "Zero"]
        ),
        "zero_target": "local_gain with externality=0",
        "real_target": "cross-fitted exact best-completion advantage/externality",
        "prequery_objective_used": False,
        "trainer_config": asdict(cfg),
        "split": dict(split.manifest),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "validation_metrics": validation,
        "test_metrics": test,
        "pressure_policy": (
            "unidentifiable pressure transported as value=0, uncertainty=1; "
            "excluded from pressure regression"
        ),
        "completion_warning": (
            "warmup completion labels are single-class and cannot support AUROC"
            if bool(offline.receipt.get("completion_is_degenerate"))
            else None
        ),
    }
    manifest_path = destination / "model_manifest_message_v3.json"
    _atomic_json(manifest_path, manifest)
    return MessageTrainingResult(
        checkpoint_path=str(checkpoint_path),
        model_manifest_path=str(manifest_path),
        history_path=str(history_path),
        best_epoch=best_epoch,
        checkpoint_hash=checkpoint_hash,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        best_validation_loss=best_validation,
        validation_real_rank_accuracy=(
            float(validation["real_rank_accuracy"])
            if math.isfinite(validation.get("real_rank_accuracy", float("nan")))
            else None
        ),
        validation_zero_rank_accuracy=(
            float(validation["zero_rank_accuracy"])
            if math.isfinite(validation.get("zero_rank_accuracy", float("nan")))
            else None
        ),
    )


__all__ = [
    "MessageTrainerConfig",
    "MessageTrainingResult",
    "prepare_paired_message_states",
    "train_message_warmstart",
]
