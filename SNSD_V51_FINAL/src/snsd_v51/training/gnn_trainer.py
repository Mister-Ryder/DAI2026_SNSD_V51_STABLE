"""Strict end-to-end trainer for deep rolling Real/Zero communication.

Differences from the additive v4.5 prototype are deliberately auditable:

* Real and Zero are separate on-policy branches, not static copies;
* every branch executes the configured non-empty, one-new-candidate-per-agent
  transitions (12 by default);
* both branches use exact factor labels for the same held-out candidates in
  their visited state -- Zero never substitutes ``local_gain``;
* the boundary state is field-equivalent to ``learned_policy_protocol._state``; and
* all graph/message encoders and acquisition heads train by default.

The exact factor/reveal transition is discrete.  Gradients span the full,
local, boundary and message encoders plus scoring heads inside each visited
observation, but intentionally do not pass through candidate selection or the
Max-Sum environment transition.  ``heads_only`` is retained as an ablation.
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
from dataclasses import replace

from snsd_v51.model import FeatureMode, TriGraphConfig, TriGraphModel
from snsd_v51.messages import BoundaryState, apply_message_intervention
from snsd_v51.runtime import PreparedTriGraph, TriGraphPolicy

from .rolling_teacher import RollingTeacherObservation
from .strict_teacher import StrictCrossfitRollingEnvironment
from .offline_dataset import DatasetSplit, OfflineDataError, OfflineDataset
from .message_warmstart_trainer import _aggregate, _without_arc_weight
from .warmstart_trainer import (
    EXPECTED_PARAMETER_COUNT,
    TRAINABLE_PREFIXES,
    LossWeights,
    _group_loss,
    _groups,
    _score_group,
    _state_dict_hash,
)


STRICT_TRAINING_SCHEMA = "SNSD-v1.5-e2e-on-policy-deep-real-zero-v4.5-strict"

ENCODER_PREFIXES = (
    "full_encoder.",
    "local_encoder.",
    "boundary_encoder.",
    "message_encoder.",
    "agent_adapters.",
    "agent_adapter_norms.",
    "agent_identity.",
    "local_context_projection.",
    "local_context_norm.",
    "agent_from_boundary.",
    "node_from_agent.",
    "boundary_agent_norm.",
    "boundary_node_norm.",
)
ACQUISITION_END_TO_END_PREFIXES = ENCODER_PREFIXES + TRAINABLE_PREFIXES


@dataclass(frozen=True)
class StrictRollingMessageTrainerConfig:
    epochs: int = 30
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    patience: int = 6
    gradient_clip: float = 5.0
    seed: int = 20260803
    device: str = "cpu"
    folds: int = 2
    negotiation_steps: int = 12
    heads_only: bool = False
    include_shuffled: bool = False
    exploration_epsilon: float = 0.05
    real_weight: float = 1.0
    zero_weight: float = 0.25
    shuffled_weight: float = 0.10
    zero_target_semantics: str = "intrinsic"
    initial_checkpoint: Optional[str] = None
    late_stage_weighted: bool = False
    priority_pair_weight: float = 1.0
    loss_weights: LossWeights = LossWeights()


@dataclass(frozen=True)
class StrictRollingMessageTrainingResult:
    checkpoint_path: str
    model_manifest_path: str
    history_path: str
    best_epoch: int
    checkpoint_hash: str
    parameter_count: int
    trainable_parameter_count: int
    encoder_trainable_parameter_count: int
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


def strict_online_boundary_states(
    offline: OfflineDataset, observation: RollingTeacherObservation
) -> Dict[int, BoundaryState]:
    """Reproduce the public fields constructed by ``learned_policy_protocol._state``."""

    if offline.generators_by_seed is None:
        raise OfflineDataError("strict feature construction requires generators")
    generators = offline.generators_by_seed.get(observation.seed)
    if generators is None:
        raise OfflineDataError("missing generators for seed %d" % observation.seed)
    cut_conflict = [dict() for _ in generators]  # type: List[Dict[int, float]]
    for agent_id, generator in enumerate(generators):
        cut_conflict[agent_id] = {
            int(node): 0.0 for node in generator.boundary
        }
    for raw_u, raw_v in offline.partitions.cut_edges:
        u, v = int(raw_u), int(raw_v)
        owner_u = int(offline.partitions.partition_id[u])
        owner_v = int(offline.partitions.partition_id[v])
        if u in cut_conflict[owner_u]:
            cut_conflict[owner_u][u] += 1.0
        if v in cut_conflict[owner_v]:
            cut_conflict[owner_v][v] += 1.0

    output = {}
    for agent_id, generator in enumerate(generators):
        boundary = tuple(int(node) for node in generator.boundary)
        raw = {
            node: float(offline.dataset.arcs[node].weight) for node in boundary
        }
        local_proxy = {
            node: float(offline.dataset.arcs[node].weight)
            / float(int(offline.graph.indptr[node + 1] - offline.graph.indptr[node]) + 1)
            for node in boundary
        }
        output[agent_id] = BoundaryState.from_maps(
            agent_id=agent_id,
            boundary_order=boundary,
            incumbent_selected=observation.anchor_selected_boundary[agent_id],
            raw_benefit=raw,
            local_marginal=local_proxy,
            conflict_score=cut_conflict[agent_id],
            prior_opportunity_cost={},
            uncertainty={},
            prior_pressure={},
        )
    return output


def _variant_messages(
    observation: RollingTeacherObservation,
    variant: str,
    negotiation_steps: int,
) -> Dict[int, Any]:
    if not observation.consumes_message:
        return {}
    if variant == "Real":
        return dict(observation.messages)
    intervention = "zero" if variant == "Zero" else "shuffled"
    return {
        agent_id: apply_message_intervention(
            message,
            intervention,
            experiment_seed=observation.seed,
            agent_id=agent_id,
            macro_round=(
                observation.heldout_fold * int(negotiation_steps)
                + observation.step
            ),
        )
        for agent_id, message in observation.messages.items()
    }


def _prepare_with_encoder_grad(
    policy: TriGraphPolicy, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> PreparedTriGraph:
    """Use runtime's exact feature path while re-enabling encoder autograd.

    ``runtime.prepare`` wraps its three encoder calls in ``no_grad`` for
    inference.  Instance wrappers establish a nested ``enable_grad`` context;
    this keeps one source of feature construction and avoids a drifting copy.
    The encoded cache is cleared before and after because an autograd graph may
    never be reused after backward.
    """

    graph_layers = policy.graph_layers
    names = ("encode_full", "encode_local", "encode_boundary")
    originals = {name: getattr(graph_layers, name) for name in names}

    def enabled(original):
        def call(*inner_args, **inner_kwargs):
            with torch.enable_grad():
                return original(*inner_args, **inner_kwargs)

        return call

    policy._full_state_cache.clear()
    try:
        for name in names:
            setattr(graph_layers, name, enabled(originals[name]))
        prepared = policy.prepare(*args, **kwargs)
    finally:
        for name in names:
            setattr(graph_layers, name, originals[name])
        policy._full_state_cache.clear()
    if not prepared.boundary_state.node_embeddings.requires_grad:
        raise OfflineDataError("strict end-to-end prepare detached the encoders")
    return prepared


def prepare_strict_rolling_state(
    offline: OfflineDataset,
    policy: TriGraphPolicy,
    observation: RollingTeacherObservation,
    variant: str,
    negotiation_steps: int = 6,
    differentiable: bool = False,
) -> PreparedTriGraph:
    if variant not in ("Real", "Zero", "Shuffled"):
        raise OfflineDataError("unknown strict communication variant %s" % variant)
    states = strict_online_boundary_states(offline, observation)
    messages = _variant_messages(observation, variant, negotiation_steps)
    args = (
        offline.dataset,
        offline.graph,
        offline.partitions,
        offline.generators_by_seed[observation.seed],
        states,
    )
    kwargs = {
        "boundary_messages": messages,
        # The union of state incumbent bits is the same boundary signature map
        # passed by the online runner; keeping one representation avoids a
        # private _RealizedRecord dependency in the trainer.
        "incumbent": None,
        "mode": FeatureMode.FULL,
    }
    if differentiable:
        return _prepare_with_encoder_grad(policy, args, kwargs)
    return policy.prepare(*args, **kwargs)


def _selection_rng(
    config: StrictRollingMessageTrainerConfig,
    epoch: int,
    observation: RollingTeacherObservation,
    agent_id: int,
    variant: str,
) -> random.Random:
    digest = hashlib.sha256(
        (
            "rolling-on-policy-v4.5-strict|%d|%d|%d|%d|%d|%s"
            % (
                int(config.seed),
                int(epoch),
                int(observation.seed),
                int(observation.heldout_fold),
                int(observation.step * 100 + agent_id),
                variant,
            )
        ).encode("ascii")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="little"))


def select_strict_on_policy_candidates(
    graph_layers: TriGraphModel,
    prepared: PreparedTriGraph,
    observation: RollingTeacherObservation,
    config: StrictRollingMessageTrainerConfig,
    epoch: int,
    variant: str,
) -> Dict[int, str]:
    grouped = _groups(observation.target_examples)
    selected = {}
    graph_layers.eval()
    with torch.no_grad():
        for agent_id in range(int(observation.receipt["source_candidate_counts"].__len__())):
            rows = grouped.get((observation.seed, agent_id), ())
            if not rows:
                raise OfflineDataError(
                    "strict on-policy step has no candidates for agent %d" % agent_id
                )
            scores = _score_group(graph_layers, prepared, rows)
            values = [float(value) for value in scores.selection_logit.detach().cpu()]
            rng = _selection_rng(config, epoch, observation, agent_id, variant)
            if rng.random() < float(config.exploration_epsilon):
                chosen = rng.randrange(len(rows))
            else:
                chosen = min(
                    range(len(rows)),
                    key=lambda index: (
                        -values[index], rows[index].prequery.signature_hash
                    ),
                )
            selected[agent_id] = str(rows[chosen].prequery.signature_hash)
    return selected


def _branch_weight(config: StrictRollingMessageTrainerConfig, variant: str) -> float:
    if variant == "Real":
        return float(config.real_weight)
    if variant == "Zero":
        return float(config.zero_weight)
    return float(config.shuffled_weight)


def _late_stage_step_weight(
    config: StrictRollingMessageTrainerConfig, step: int
) -> float:
    if not bool(config.late_stage_weighted):
        return 1.0
    step = int(step)
    if 0 <= step <= 5:
        return 1.0
    if 6 <= step <= 8:
        return 2.0
    if 9 <= step <= 11:
        return 3.0
    raise OfflineDataError("late-stage weighting requires steps 0 through 11")


def _priority_pair_loss(
    graph_layers: TriGraphModel,
    prepared: PreparedTriGraph,
    observation: RollingTeacherObservation,
    rows: Sequence[Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mine exact-label near-incumbent and hard-negative ranking pairs.

    Targets are never replaced: the pair sign is taken from the same exact
    advantage field consumed by the ordinary strict loss.  Mining only chooses
    which already-labelled comparisons receive extra ranking weight.
    """

    scores = _score_group(graph_layers, prepared, rows)
    reference = scores.selection_logit
    eligible = [
        index
        for index, item in enumerate(rows)
        if bool(item.targets.advantage_mask)
        and item.targets.best_completion_advantage is not None
    ]
    if len(eligible) < 2:
        return reference.sum() * 0.0, {
            "priority_pair_count": 0.0,
            "near_incumbent_pair_count": 0.0,
            "hard_negative_pair_count": 0.0,
        }
    target = {
        index: float(rows[index].targets.best_completion_advantage)
        for index in eligible
    }
    best = min(
        eligible,
        key=lambda index: (
            -target[index], rows[index].prequery.signature_hash
        ),
    )
    agent_id = int(rows[0].prequery.agent_id)
    anchor = set(observation.anchor_selected_boundary[agent_id])
    distance = {
        index: len(anchor.symmetric_difference(
            set(rows[index].prequery.selected_boundary)
        ))
        for index in eligible
    }
    near = sorted(
        (index for index in eligible if index != best),
        key=lambda index: (
            distance[index],
            -target[index],
            rows[index].prequery.signature_hash,
        ),
    )[:2]
    predicted = reference.detach().cpu().tolist()
    hard = sorted(
        (
            index
            for index in eligible
            if index != best and target[index] < target[best] - 1.0e-9
        ),
        key=lambda index: (
            -float(predicted[index]), rows[index].prequery.signature_hash
        ),
    )[:2]
    pair_sources = {}  # type: Dict[Tuple[int, int], set]
    for source, indices in (("near_incumbent", near), ("hard_negative", hard)):
        for index in indices:
            pair = (min(best, index), max(best, index))
            if abs(target[pair[0]] - target[pair[1]]) <= 1.0e-9:
                continue
            pair_sources.setdefault(pair, set()).add(source)
    losses = []
    for left, right in sorted(pair_sources):
        sign = 1.0 if target[left] > target[right] else -1.0
        margin = float(sign) * (reference[left] - reference[right])
        losses.append(functional.softplus(-margin))
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = reference.sum() * 0.0
    return loss, {
        "priority_pair_count": float(len(pair_sources)),
        "near_incumbent_pair_count": float(
            sum("near_incumbent" in sources for sources in pair_sources.values())
        ),
        "hard_negative_pair_count": float(
            sum("hard_negative" in sources for sources in pair_sources.values())
        ),
    }


def _exact_group_loss(
    graph_layers: TriGraphModel,
    prepared: PreparedTriGraph,
    observation: RollingTeacherObservation,
    rows: Sequence[Any],
    config: StrictRollingMessageTrainerConfig,
    variant: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervise one branch with an explicitly named exact target.

    Real uses the exact global completion advantage.  By default Zero uses
    the exact information-restricted intrinsic marginal.  This is not a
    purported zero-message global optimum: the factor identity
    ``global_advantage = local_gain + externality`` is checked before the
    intrinsic component is used.  ``zero_target_semantics=global`` is an
    available same-global-label sensitivity analysis.
    """

    agent_id = int(rows[0].prequery.agent_id)
    pressure_rows = (
        tuple(
            item
            for item in observation.arc_pressures
            if int(item.agent_id) == agent_id
        )
        if variant == "Real"
        else ()
    )
    target_rows = tuple(rows)
    target_semantics = "exact_global_completion_advantage"
    if variant != "Real" and config.zero_target_semantics == "intrinsic":
        converted = []
        for item in rows:
            target = item.targets
            if (
                target.best_completion_advantage is not None
                and target.local_gain is not None
                and target.externality is not None
                and not math.isclose(
                    float(target.best_completion_advantage),
                    float(target.local_gain) + float(target.externality),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-7,
                )
            ):
                raise OfflineDataError(
                    "exact advantage decomposition failed for %s"
                    % item.prequery.signature_hash
                )
            intrinsic = replace(
                target,
                best_completion_advantage=target.local_gain,
                advantage_mask=bool(target.local_gain_mask),
                externality=0.0 if target.local_gain_mask else None,
                externality_mask=bool(target.local_gain_mask),
                uncertainty_mask=bool(target.local_gain_mask),
                teacher_source="exact_information_restricted_intrinsic_v4_5_strict",
            )
            converted.append(replace(item, targets=intrinsic))
        target_rows = tuple(converted)
        target_semantics = "exact_information_restricted_intrinsic_marginal"
    weights = (
        config.loss_weights
        if variant == "Real"
        else _without_arc_weight(config.loss_weights)
    )
    loss, raw = _group_loss(graph_layers, prepared, target_rows, pressure_rows, weights)
    priority_loss = loss.sum() * 0.0
    priority_metrics = {
        "priority_pair_count": 0.0,
        "near_incumbent_pair_count": 0.0,
        "hard_negative_pair_count": 0.0,
    }
    if bool(config.late_stage_weighted):
        priority_loss, priority_metrics = _priority_pair_loss(
            graph_layers, prepared, observation, target_rows
        )
        loss = loss + (
            float(config.priority_pair_weight)
            * float(weights.ranking)
            * priority_loss
        )
    prefix = variant.lower()
    branch_weight = _branch_weight(config, variant)
    step_weight = _late_stage_step_weight(config, observation.step)
    weighted_loss = loss * branch_weight * step_weight
    metrics = {
        "loss": float(weighted_loss.detach().item()),
        prefix + "_loss": float(raw["loss"]),
        prefix + "_rank_correct": float(raw["rank_correct"]),
        prefix + "_rank_pairs": float(raw["rank_pairs"]),
        "pressure_count": float(raw["pressure_count"]),
        "candidate_count": float(raw["candidate_count"]),
        "arc_pressure_loss": float(raw["arc_pressure_loss"]),
        "trajectory_step": float(observation.step),
        "communication_input_group": float(
            variant == "Real" and observation.consumes_message
        ),
        "zero_input_group": float(
            variant == "Zero" and observation.consumes_message
        ),
        "exact_target_group": 1.0,
        "fabricated_zero_target_group": 0.0,
        "advantage_decomposition_checked_group": 1.0,
        "late_stage_step_weight": float(step_weight),
        "priority_pair_loss": float(priority_loss.detach().item()),
        "priority_pair_weight": (
            float(config.priority_pair_weight)
            if bool(config.late_stage_weighted)
            else 0.0
        ),
    }
    metrics.update(priority_metrics)
    metrics["intrinsic_target_group"] = float(
        target_semantics == "exact_information_restricted_intrinsic_marginal"
    )
    return weighted_loss, metrics


def _trajectory_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _step_audit(
    rows: Sequence[Mapping[str, Any]], num_agents: int
) -> Dict[str, Mapping[str, Any]]:
    output = {}
    keys = sorted(set((str(row["variant"]), int(row["step"])) for row in rows))
    for variant, step in keys:
        selected = [
            row
            for row in rows
            if str(row["variant"]) == variant and int(row["step"]) == step
        ]
        agent_slots = len(selected) * int(num_agents)
        changes = sum(len(row["changed_agents"]) for row in selected)
        output["%s/step_%d" % (variant, step)] = {
            "observations": len(selected),
            "candidate_samples": sum(int(row["target_count"]) for row in selected),
            "minimum_candidates_per_agent": min(
                int(value)
                for row in selected
                for value in row["target_count_by_agent"].values()
            ),
            "revealed_candidates": sum(int(row["revealed_count"]) for row in selected),
            "anchor_change_events": changes,
            "anchor_change_rate": (
                float(changes) / float(agent_slots) if agent_slots else float("nan")
            ),
            "full_agent_activity": all(
                int(row["revealed_count"]) == int(num_agents) for row in selected
            ),
        }
    return output


def _run_strict_rollouts(
    offline: OfflineDataset,
    graph_layers: TriGraphModel,
    policy: TriGraphPolicy,
    seeds: Sequence[int],
    config: StrictRollingMessageTrainerConfig,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    trainable: Sequence[torch.nn.Parameter] = (),
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    training = optimizer is not None
    variants = ["Real", "Zero"]
    if bool(config.include_shuffled):
        variants.append("Shuffled")
    metrics = []
    trajectory_rows = []
    for seed in seeds:
        for heldout_fold in range(int(config.folds)):
            environments = {
                variant: StrictCrossfitRollingEnvironment(
                    offline=offline,
                    seed=int(seed),
                    heldout_fold=heldout_fold,
                    folds=int(config.folds),
                    negotiation_steps=int(config.negotiation_steps),
                )
                for variant in variants
            }
            for _ in range(int(config.negotiation_steps)):
                if training:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                snapshots = {}
                # Select every branch before any optimizer update, ensuring the
                # causal branch contrast is not contaminated by update order.
                for variant in variants:
                    observation = environments[variant].observe()
                    graph_layers.eval()  # inference-identical dropout semantics
                    prepared = prepare_strict_rolling_state(
                        offline,
                        policy,
                        observation,
                        variant=variant,
                        negotiation_steps=int(config.negotiation_steps),
                        differentiable=training and not bool(config.heads_only),
                    )
                    selected = select_strict_on_policy_candidates(
                        graph_layers,
                        prepared,
                        observation,
                        config,
                        epoch,
                        variant,
                    )
                    snapshots[variant] = (observation, prepared, selected)

                losses = []
                normalizer = 0.0
                for variant in variants:
                    observation, prepared, _ = snapshots[variant]
                    grouped = _groups(observation.target_examples)
                    for (_, _agent_id), rows in sorted(grouped.items()):
                        if training:
                            loss, row = _exact_group_loss(
                                graph_layers, prepared, observation, rows, config, variant
                            )
                        else:
                            with torch.no_grad():
                                loss, row = _exact_group_loss(
                                    graph_layers,
                                    prepared,
                                    observation,
                                    rows,
                                    config,
                                    variant,
                                )
                        losses.append(loss)
                        normalizer += _branch_weight(config, variant)
                        metrics.append(row)
                if training:
                    if not losses or not normalizer > 0.0:
                        raise OfflineDataError("strict rolling step has no training loss")
                    step_loss = torch.stack(losses).sum() / float(normalizer)
                    if not bool(torch.isfinite(step_loss).item()):
                        raise FloatingPointError(
                            "non-finite strict rolling loss at epoch %d seed %d fold %d"
                            % (epoch, seed, heldout_fold)
                        )
                    step_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(config.gradient_clip)
                    )
                    optimizer.step()
                    policy._full_state_cache.clear()

                for variant in variants:
                    observation, _prepared, selected = snapshots[variant]
                    transition = environments[variant].advance(selected)
                    target_counts = {agent_id: 0 for agent_id in range(environments[variant].num_agents)}
                    for item in observation.target_examples:
                        target_counts[int(item.prequery.agent_id)] += 1
                    trajectory_rows.append(
                        {
                            "seed": int(seed),
                            "heldout_fold": heldout_fold,
                            "variant": variant,
                            "step": int(observation.step),
                            "real_message_input": bool(
                                variant == "Real" and observation.consumes_message
                            ),
                            "zero_intervention_input": bool(
                                variant == "Zero" and observation.consumes_message
                            ),
                            "anchor_hash": observation.receipt["anchor_hash"],
                            "selected": {
                                str(key): value for key, value in selected.items()
                            },
                            "changed_agents": list(transition["changed_agents"]),
                            "factor_value": float(transition["factor_value"]),
                            "target_count": len(observation.target_examples),
                            "target_count_by_agent": {
                                str(key): value for key, value in target_counts.items()
                            },
                            "revealed_count": len(
                                transition["revealed_signature_hashes"]
                            ),
                            "source_target_disjoint": bool(
                                observation.receipt["source_target_disjoint"]
                            ),
                            "candidate_objective_in_own_message": False,
                            "target_semantics": (
                                "exact_global_completion_advantage"
                                if variant == "Real"
                                or config.zero_target_semantics == "global"
                                else "exact_information_restricted_intrinsic_marginal"
                            ),
                        }
                    )
                del snapshots

    num_agents = int(offline.partitions.k)
    audit = {
        "schema_version": STRICT_TRAINING_SCHEMA,
        "trajectory_hash": _trajectory_digest(trajectory_rows),
        "trajectory_observations": len(trajectory_rows),
        "negotiation_steps": int(config.negotiation_steps),
        "variants": variants,
        "real_message_input_steps": sum(
            bool(row["real_message_input"]) for row in trajectory_rows
        ),
        "zero_intervention_input_steps": sum(
            bool(row["zero_intervention_input"]) for row in trajectory_rows
        ),
        "revealed_candidates": sum(
            int(row["revealed_count"]) for row in trajectory_rows
        ),
        "all_steps_full_agent_activity": all(
            int(row["revealed_count"]) == num_agents for row in trajectory_rows
        ),
        "all_sources_exclude_current_heldout": all(
            bool(row["source_target_disjoint"]) for row in trajectory_rows
        ),
        "candidate_objective_in_own_message": False,
        "fabricated_zero_labels": False,
        "zero_target_semantics": (
            "exact_global_completion_advantage_in_zero_branch_state"
            if config.zero_target_semantics == "global"
            else "exact_information_restricted_intrinsic_marginal; not a zero-message global optimum"
        ),
        "exact_advantage_identity": "global_advantage = local_gain + externality",
        "gradient_through_discrete_factor_transition": False,
        "per_step": _step_audit(trajectory_rows, num_agents),
    }
    return _aggregate(metrics), audit


def _load_initial_checkpoint(graph_layers: TriGraphModel, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    source = Path(path).resolve()
    if not source.is_file():
        raise OfflineDataError("initial checkpoint does not exist: %s" % source)
    payload = torch.load(str(source), map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    graph_layers.load_state_dict(state_dict, strict=True)
    return str(source)


def audit_checkpoint_depth(
    checkpoint: Mapping[str, Any],
    inference_steps: int,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    """Hard-check the recurrent depth recorded in a strict checkpoint.

    Runners can opt into a depth-transfer ablation, but the mismatch is then
    returned as an explicit audit field rather than silently accepted.
    """

    trained = checkpoint.get("trained_negotiation_steps")
    if trained is None:
        raise OfflineDataError("checkpoint has no trained_negotiation_steps")
    trained = int(trained)
    inference = int(inference_steps)
    matched = trained == inference
    if not matched and not allow_mismatch:
        raise OfflineDataError(
            "checkpoint depth mismatch: trained=%d inference=%d; pass an "
            "explicit depth-transfer override only for a labelled ablation"
            % (trained, inference)
        )
    return {
        "trained_negotiation_steps": trained,
        "inference_negotiation_steps": inference,
        "depth_match": matched,
        "depth_transfer_ablation": not matched,
    }


def train_strict_rolling_message_policy(
    offline: OfflineDataset,
    split: DatasetSplit,
    output_dir: str,
    config: Optional[StrictRollingMessageTrainerConfig] = None,
) -> StrictRollingMessageTrainingResult:
    cfg = config if config is not None else StrictRollingMessageTrainerConfig()
    if int(cfg.negotiation_steps) < 6:
        raise OfflineDataError("strict v4.5 requires at least six negotiation steps")
    if not 0.0 <= float(cfg.exploration_epsilon) <= 1.0:
        raise OfflineDataError("exploration_epsilon must lie in [0, 1]")
    if float(cfg.real_weight) <= 0.0 or float(cfg.zero_weight) < 0.0:
        raise OfflineDataError("invalid strict Real/Zero branch weights")
    if cfg.zero_target_semantics not in ("intrinsic", "global"):
        raise OfflineDataError("zero_target_semantics must be intrinsic or global")
    if bool(cfg.late_stage_weighted) and int(cfg.negotiation_steps) != 12:
        raise OfflineDataError(
            "late-stage weighting is defined only for 12 negotiation steps"
        )
    if bool(cfg.late_stage_weighted) and float(cfg.priority_pair_weight) <= 0.0:
        raise OfflineDataError("priority_pair_weight must be positive")

    destination = Path(output_dir).resolve()
    checkpoint_path = destination / "checkpoint_message_v4_5_strict.pt"
    if checkpoint_path.exists():
        raise OfflineDataError("refusing to overwrite %s" % checkpoint_path)
    destination.mkdir(parents=True, exist_ok=True)

    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model_config = TriGraphConfig()
    graph_layers = TriGraphModel(model_config)
    initial_checkpoint = _load_initial_checkpoint(graph_layers, cfg.initial_checkpoint)
    parameter_count = sum(int(value.numel()) for value in graph_layers.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise OfflineDataError(
            "strict architecture mismatch: expected %d, got %d"
            % (EXPECTED_PARAMETER_COUNT, parameter_count)
        )
    for name, parameter in graph_layers.named_parameters():
        parameter.requires_grad = (
            any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)
            if bool(cfg.heads_only)
            else any(
                name.startswith(prefix)
                for prefix in ACQUISITION_END_TO_END_PREFIXES
            )
        )
    trainable_named = [
        (name, parameter)
        for name, parameter in graph_layers.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in trainable_named]
    trainable_count = sum(int(value.numel()) for value in trainable)
    encoder_trainable_count = sum(
        int(parameter.numel())
        for name, parameter in trainable_named
        if name.startswith(ENCODER_PREFIXES)
    )
    if not bool(cfg.heads_only) and encoder_trainable_count <= 0:
        raise OfflineDataError("end-to-end mode failed to expose encoder parameters")

    policy = TriGraphPolicy(graph_layers=graph_layers, device=cfg.device, seed=0)
    graph_layers = policy.graph_layers
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    train_seeds = tuple(int(value) for value in split.manifest["train_seeds"])
    validation_seeds = tuple(
        int(value) for value in split.manifest["validation_seeds"]
    )
    test_seeds = tuple(int(value) for value in split.manifest["test_seeds"])
    if not train_seeds:
        raise OfflineDataError("strict rolling training split is empty")

    best_state = copy.deepcopy(graph_layers.state_dict())
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    history = []
    for epoch in range(1, int(cfg.epochs) + 1):
        training, training_rollout = _run_strict_rollouts(
            offline,
            graph_layers,
            policy,
            train_seeds,
            cfg,
            epoch,
            optimizer=optimizer,
            trainable=trainable,
        )
        policy.invalidate_encoded_cache()
        validation, validation_rollout = _run_strict_rollouts(
            offline, graph_layers, policy, validation_seeds, cfg, epoch=0
        )
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
                "train_rollout": training_rollout,
                "validation_rollout": validation_rollout,
                "improved": improved,
            }
        )
        print(
            json.dumps(
                {
                    "event": "strict_training_epoch",
                    "epoch": epoch,
                    "train_loss": training.get("loss"),
                    "validation_loss": validation.get("loss"),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation,
                    "stale": stale,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if stale >= int(cfg.patience):
            break

    graph_layers.load_state_dict(best_state, strict=True)
    policy.invalidate_encoded_cache()
    strict_probe = TriGraphModel(TriGraphConfig())
    strict_probe.load_state_dict(graph_layers.state_dict(), strict=True)
    checkpoint_hash = _state_dict_hash(graph_layers.state_dict())
    validation, validation_rollout = _run_strict_rollouts(
        offline, graph_layers, policy, validation_seeds, cfg, epoch=0
    )
    test, test_rollout = _run_strict_rollouts(
        offline, graph_layers, policy, test_seeds, cfg, epoch=0
    )
    checkpoint = {
        "schema_version": STRICT_TRAINING_SCHEMA,
        "state_dict": graph_layers.state_dict(),
        "model_config": asdict(model_config),
        "model_class": "snsd_v51.model.TriGraphModel",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "encoder_trainable_parameter_count": encoder_trainable_count,
        "frozen_nonacquisition_parameter_count": parameter_count - trainable_count,
        "trainable_parameter_names": [name for name, _ in trainable_named],
        "checkpoint_hash": checkpoint_hash,
        "training_mode": (
            "heads_only_strict_rolling_ablation"
            if cfg.heads_only
            else (
                "end_to_end_strict_on_policy_deep_transition_real_zero_late_weighted"
                if cfg.late_stage_weighted
                else "end_to_end_strict_on_policy_deep_transition_real_zero"
            )
        ),
        "dataset_hash": offline.receipt.get("dataset_hash"),
        "split_manifest": dict(split.manifest),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "initial_checkpoint": initial_checkpoint,
        "trained_negotiation_steps": int(cfg.negotiation_steps),
        "depth_mismatch_requires_explicit_override": True,
        "late_stage_loss_weighting": (
            {
                "enabled": True,
                "step_0_5": 1.0,
                "step_6_8": 2.0,
                "step_9_11": 3.0,
                "scope": "exact strict group loss",
            }
            if cfg.late_stage_weighted
            else {"enabled": False}
        ),
        "candidate_pair_priority": (
            {
                "enabled": True,
                "mode": "near_incumbent_and_model_hard_negative",
                "pairs_per_source_per_group": 2,
                "weight": float(cfg.priority_pair_weight),
                "pair_sign_source": "unchanged exact advantage target",
            }
            if cfg.late_stage_weighted
            else {"enabled": False}
        ),
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, str(temporary))
    os.replace(str(temporary), str(checkpoint_path))
    history_path = destination / "training_history_message_v4_5_strict.jsonl"
    _atomic_jsonl(history_path, history)
    manifest = {
        "schema_version": STRICT_TRAINING_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "strict_load_verified": True,
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "trainable_parameter_count": trainable_count,
        "encoder_trainable_parameter_count": encoder_trainable_count,
        "frozen_nonacquisition_parameter_count": parameter_count - trainable_count,
        "heads_only": bool(cfg.heads_only),
        "encoder_training": (
            "frozen-heads-only-ablation" if cfg.heads_only else "end-to-end"
        ),
        "gradient_scope": (
            "full/local/boundary/message encoders and all heads within each "
            "observation; discrete selection and exact factor transitions detached"
        ),
        "negotiation_steps": int(cfg.negotiation_steps),
        "trained_negotiation_steps": int(cfg.negotiation_steps),
        "full_agent_transitions_per_trajectory": int(cfg.negotiation_steps),
        "inference_depth_compatibility": (
            "runner must match trained_negotiation_steps or explicitly log a depth mismatch ablation"
        ),
        "candidate_objective_in_own_message": False,
        "real_zero_source_excludes_current_heldout": True,
        "fabricated_zero_labels": False,
        "zero_target": (
            "exact global held-out label in Zero on-policy branch state"
            if cfg.zero_target_semantics == "global"
            else "exact information-restricted intrinsic marginal; not claimed as zero-message global optimum"
        ),
        "real_target": "exact chain held-out label in Real on-policy branch state",
        "exact_advantage_identity": "global_advantage = local_gain + externality",
        "feature_builder": "runtime.prepare plus learned_policy_protocol-equivalent BoundaryState",
        "initial_checkpoint": initial_checkpoint,
        "trainer_config": asdict(cfg),
        "late_stage_loss_weighting": (
            {
                "enabled": True,
                "step_0_5": 1.0,
                "step_6_8": 2.0,
                "step_9_11": 3.0,
                "scope": "exact strict group loss",
                "split_or_target_semantics_changed": False,
            }
            if cfg.late_stage_weighted
            else {"enabled": False}
        ),
        "candidate_pair_priority": (
            {
                "enabled": True,
                "mode": "near_incumbent_and_model_hard_negative",
                "pairs_per_source_per_group": 2,
                "weight": float(cfg.priority_pair_weight),
                "pair_sign_source": "unchanged exact advantage target",
                "split_or_target_semantics_changed": False,
            }
            if cfg.late_stage_weighted
            else {"enabled": False}
        ),
        "split": dict(split.manifest),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "validation_metrics": validation,
        "test_metrics": test,
        "validation_rollout": validation_rollout,
        "test_rollout": test_rollout,
    }
    manifest_path = destination / "model_manifest_message_v4_5_strict.json"
    _atomic_json(manifest_path, manifest)
    return StrictRollingMessageTrainingResult(
        checkpoint_path=str(checkpoint_path),
        model_manifest_path=str(manifest_path),
        history_path=str(history_path),
        best_epoch=best_epoch,
        checkpoint_hash=checkpoint_hash,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        encoder_trainable_parameter_count=encoder_trainable_count,
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
    "STRICT_TRAINING_SCHEMA",
    "ACQUISITION_END_TO_END_PREFIXES",
    "ENCODER_PREFIXES",
    "StrictRollingMessageTrainerConfig",
    "StrictRollingMessageTrainingResult",
    "prepare_strict_rolling_state",
    "audit_checkpoint_depth",
    "select_strict_on_policy_candidates",
    "strict_online_boundary_states",
    "train_strict_rolling_message_policy",
]
