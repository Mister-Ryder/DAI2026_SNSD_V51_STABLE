"""Rolling-working-solution recurrent negotiation v4.4.

Unlike v4.3, every counterproposal is anchored at the latest joint working
solution chosen from the accumulated current macro domain.  The accepted
incumbent remains immutable until the macro-end monotone commit.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from snsd_core.candidates import CandidateGenerator
from snsd_core.data import load_arcs
from snsd_core.graph import build_conflict_graph, partition_equal_time
from snsd_core.solver import ConditionedLocalSolver
from snsd_core.verifier import verify_selected_ids

from .base_protocol import (
    ExperimentResult,
    _atomic_json,
    _deduplicate_records,
    _sha256_file,
)
from .recurrent_protocol import RecurrentNegotiationConfig
from .anchored_relay_protocol import AnchoredRelayProtocol
from .messages import ProposalEngine


ROLLING_METHODS = (
    "Full",
    "NoPeerComm",
    "SelectionOnly",
    "ShuffledComm",
    "NoGlobalContext",
    "ConflictMessageOnly",
    "BenefitMessageOnly",
    "OneShotComm",
    "FrozenAnchorComm",
)


@dataclass(frozen=True)
class RollingExperimentConfig(RecurrentNegotiationConfig):
    frozen_anchor_comm: bool = False

    @property
    def reported_method(self) -> str:
        if self.frozen_anchor_comm:
            return "FrozenAnchorComm"
        return super().reported_method


def _joint_hash(records: Mapping[int, Any]) -> str:
    payload = [
        {
            "agent_id": int(agent_id),
            "signature_hash": str(records[agent_id].signature.signature_hash),
            "selected_boundary": list(records[agent_id].signature.selected_ids),
        }
        for agent_id in sorted(records)
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class RollingNegotiationProtocol(AnchoredRelayProtocol):
    config: RollingExperimentConfig

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-rolling-recurrent-run-v4.4",
                "method": self.config.reported_method,
                "rolling_working_solution": not self.config.frozen_anchor_comm,
                "frozen_anchor_control": bool(self.config.frozen_anchor_comm),
                "accepted_incumbent_updates_only_at_macro_commit": True,
            }
        )
        return fields

    def _selected_union(self, records: Mapping[int, Any]):
        return tuple(
            sorted(
                node
                for record in records.values()
                for node in record.result.selected_ids
            )
        )

    def _independent_working(self, current):
        chosen = {}
        for agent_id in range(self.config.num_agents):
            feasible = [
                record
                for record in current[agent_id]
                if record.result.status == "feasible"
            ]
            chosen[agent_id] = max(
                feasible,
                key=lambda record: (
                    float(record.result.objective),
                    record.candidate_id,
                ),
            )
        return chosen

    def run(self) -> ExperimentResult:
        started = time.perf_counter()
        self._prepare_output()
        try:
            data_path = Path(self.config.data_path).resolve()
            self.dataset = load_arcs(str(data_path))
            self.graph = build_conflict_graph(self.dataset.arcs)
            self.partitions = partition_equal_time(
                self.dataset.arcs,
                self.graph,
                k=self.config.num_agents,
                horizon_start=self.config.horizon_start,
                horizon_end=self.config.horizon_end,
            )
            if self.partitions.has_non_adjacent_cuts:
                raise ValueError("C3 rolling protocol requires an adjacent chain")
            cut_edges_by_link = self._cut_edges_by_link(self.partitions)
            generators = [
                CandidateGenerator(
                    agent_id=i,
                    arcs=self.dataset.arcs,
                    graph=self.graph,
                    partitions=self.partitions,
                    top_h=64,
                    seed=self.config.seed,
                )
                for i in range(self.config.num_agents)
            ]
            engines = [
                ProposalEngine(i, self.config.seed, hotspot_size=64)
                for i in range(self.config.num_agents)
            ]
            solver = ConditionedLocalSolver(
                self.dataset.arcs,
                self.graph,
                self.partitions,
                moves=10 if self.config.smoke else 500,
                restarts=1 if self.config.smoke else 2,
                seed=self.config.seed,
            )
            self._event(
                "run_start",
                method=self.config.reported_method,
                seed=self.config.seed,
                smoke=self.config.smoke,
                oracle=bool(self.config.oracle),
                learned_model_used=self.policy is not None,
                strict_query_protocol=True,
                recurrent_negotiation=True,
                rolling_working_solution=not self.config.frozen_anchor_comm,
                negotiation_steps=self.config.negotiation_steps,
                num_nodes=self.graph.num_nodes,
                num_edges=self.graph.num_edges,
                cut_edges=len(self.partitions.cut_edges),
                history_archive_enabled=False,
            )

            warmup_records = {}
            safe_ids = {}
            warmup_quartets = 0 if self.config.smoke else 4
            for agent_id, generator in enumerate(generators):
                signatures, _ = generator.warmup(warmup_quartets)
                safe_ids[agent_id] = generator.empty_signature().signature_hash
                warmup_records[agent_id] = [
                    self._solve_signature(
                        solver,
                        self.graph,
                        signature,
                        phase="warmup",
                        macro_round=-1,
                        wave="warmup",
                        slot=slot,
                    )
                    for slot, signature in enumerate(signatures)
                ]
            warmup_factor, warmup_order = self._factor_pass(
                warmup_records,
                incumbent_ids={},
                safe_ids=safe_ids,
                cut_edges_by_link=cut_edges_by_link,
                label="warmup_initialization",
                count_messages=False,
                require_incumbent=False,
            )
            incumbent_records = self._chosen_records(warmup_factor, warmup_order)
            safe_records = {}
            for agent_id in range(self.config.num_agents):
                matches = [
                    record
                    for record in warmup_records[agent_id]
                    if record.candidate_id == safe_ids[agent_id]
                ]
                if len(matches) != 1 or matches[0].result.status != "feasible":
                    raise RuntimeError("missing unique feasible safe anchor")
                safe_records[agent_id] = matches[0]

            incumbent_selected = self._selected_union(incumbent_records)
            initial_check = verify_selected_ids(self.graph, incumbent_selected)
            if not initial_check.feasible:
                raise RuntimeError("initial incumbent invalid: {}".format(initial_check.message))
            self.incumbent_objective = self._objective(incumbent_selected)
            initial_objective = self.incumbent_objective

            round_metrics = []
            negotiation_metrics = []
            for macro_round in range(self.config.macro_rounds):
                macro_anchor_records = dict(incumbent_records)
                working_records = dict(incumbent_records)
                current = {
                    agent_id: _deduplicate_records(
                        [incumbent_records[agent_id], safe_records[agent_id]]
                    )
                    for agent_id in range(self.config.num_agents)
                }
                pressure = None
                last_factor = None
                last_order = None
                for step in range(self.config.negotiation_steps):
                    context_round = macro_round * self.config.negotiation_steps + step
                    anchor_records = (
                        macro_anchor_records
                        if self.config.frozen_anchor_comm
                        else working_records
                    )
                    anchor_ids = {
                        agent_id: anchor_records[agent_id].candidate_id
                        for agent_id in range(self.config.num_agents)
                    }
                    anchor_hash = _joint_hash(anchor_records)
                    step_pressure, message_mode = self._pressure_for_step(step, pressure)
                    self._event(
                        "negotiation_step_start",
                        macro_round=macro_round,
                        negotiation_step=step,
                        engine_round_context=context_round,
                        message_mode=message_mode,
                        anchor_hash=anchor_hash,
                        anchor_signature_hashes=[
                            anchor_records[i].candidate_id for i in range(6)
                        ],
                        anchor_selected_boundary_ids=[
                            list(anchor_records[i].signature.selected_ids) for i in range(6)
                        ],
                        current_domain_counts=[len(current[i]) for i in range(6)],
                    )
                    purchased = self._query_wave(
                        "A" if step == 0 else "B",
                        context_round,
                        generators,
                        engines,
                        anchor_records,
                        solver,
                        self.graph,
                        pressure_maps=step_pressure,
                    )
                    for agent_id in range(self.config.num_agents):
                        current[agent_id] = _deduplicate_records(
                            current[agent_id] + purchased[agent_id]
                        )

                    message_pass = self._count_factor_communication(step)
                    last_factor, last_order = self._factor_pass(
                        current,
                        anchor_ids,
                        safe_ids,
                        cut_edges_by_link,
                        label="macro_{}_step_{}_post_query".format(macro_round, step),
                        count_messages=message_pass,
                    )
                    pressure = self._pressure_maps(generators, last_order, last_factor)

                    if self.behavior.final_selection == "independent":
                        next_working = self._independent_working(current)
                    else:
                        next_working = self._chosen_records(last_factor, last_order)
                    working_selected = self._selected_union(next_working)
                    working_check = verify_selected_ids(self.graph, working_selected)
                    working_objective = self._objective(working_selected)
                    if (
                        self.behavior.final_selection != "independent"
                        and not working_check.feasible
                    ):
                        raise RuntimeError(
                            "joint rolling working solution invalid: {}".format(
                                working_check.message
                            )
                        )
                    changed_agents = [
                        agent_id
                        for agent_id in range(self.config.num_agents)
                        if next_working[agent_id].candidate_id != anchor_ids[agent_id]
                    ]
                    working_records = next_working
                    step_row = {
                        "macro_round": macro_round,
                        "negotiation_step": step,
                        "engine_round_context": context_round,
                        "message_mode": message_mode,
                        "message_pass": message_pass,
                        "online_calls_after_step": self.online_calls,
                        "anchor_hash": anchor_hash,
                        "anchor_signature_hashes": [anchor_ids[i] for i in range(6)],
                        "anchor_selected_boundary_ids": [
                            list(anchor_records[i].signature.selected_ids) for i in range(6)
                        ],
                        "working_hash": _joint_hash(working_records),
                        "working_signature_hashes": [
                            working_records[i].candidate_id for i in range(6)
                        ],
                        "working_selected_boundary_ids": [
                            list(working_records[i].signature.selected_ids) for i in range(6)
                        ],
                        "anchor_changed_agents": changed_agents,
                        "working_objective": working_objective,
                        "working_feasible": bool(working_check.feasible),
                        "factor_domain_objective": last_factor.best_value,
                        "factor_value": last_factor.best_value,
                        "step_objective": last_factor.best_value,
                        "current_domain_counts": [len(current[i]) for i in range(6)],
                    }
                    negotiation_metrics.append(step_row)
                    self._event("negotiation_step", **step_row)

                assert last_factor is not None and last_order is not None
                if self.behavior.final_selection == "independent":
                    chosen = self._independent_working(current)
                    final_domain_value = sum(
                        float(record.result.objective) for record in chosen.values()
                    )
                else:
                    chosen = self._chosen_records(last_factor, last_order)
                    final_domain_value = last_factor.best_value
                proposed_ids, shielded_records, shield = self._shield(chosen, generators)
                proposed_objective = float(shield["post_shield_objective"])
                accepted = proposed_objective > self.incumbent_objective + 1.0e-9
                if accepted:
                    incumbent_records = shielded_records
                    incumbent_selected = proposed_ids
                    self.incumbent_objective = proposed_objective
                if self.objective_trace:
                    self.objective_trace[-1] = float(self.incumbent_objective)
                row = {
                    "macro_round": macro_round,
                    "final_current_domain_value": final_domain_value,
                    "proposed_objective": proposed_objective,
                    "accepted": accepted,
                    "incumbent_objective": self.incumbent_objective,
                    "current_domain_counts": [len(current[i]) for i in range(6)],
                    "shield": shield,
                }
                round_metrics.append(row)
                self._event("macro_commit", **row)

            if self.online_calls != self.config.expected_online_calls:
                raise RuntimeError(
                    "online-call invariant failed: expected {}, got {}".format(
                        self.config.expected_online_calls, self.online_calls
                    )
                )
            final_check = verify_selected_ids(self.graph, incumbent_selected)
            if not final_check.feasible:
                raise RuntimeError("final schedule invalid: {}".format(final_check.message))
            gains = [value - initial_objective for value in self.objective_trace]
            summary = {
                "status": "complete",
                "method": self.config.reported_method,
                "seed": self.config.seed,
                "smoke": self.config.smoke,
                "data_path": str(data_path),
                "dataset_sha256": _sha256_file(data_path),
                "graph_hash": self.graph.graph_hash,
                "num_nodes": self.graph.num_nodes,
                "num_edges": self.graph.num_edges,
                "cut_edges": len(self.partitions.cut_edges),
                "num_agents": self.config.num_agents,
                "macro_rounds": self.config.macro_rounds,
                "waves_per_macro": self.config.negotiation_steps,
                "query_budget_per_agent_step": 1,
                "warmup_query_calls": self.warmup_calls,
                "online_query_calls": self.online_calls,
                "expected_online_query_calls": self.config.expected_online_calls,
                "initial_objective": initial_objective,
                "final_objective": self.incumbent_objective,
                "final_gain": self.incumbent_objective - initial_objective,
                "query_normalized_auc": (
                    float(sum(gains)) / float(len(gains)) if gains else 0.0
                ),
                "objective_trace": self.objective_trace,
                "final_selected_ids": list(incumbent_selected),
                "final_feasible": final_check.feasible,
                "rounds": round_metrics,
                "negotiation": negotiation_metrics,
                "message_passes": self.message_passes,
                "directed_messages": self.directed_messages,
                "message_payload_scalars": self.message_payload_scalars,
                "message_payload_bytes_float64": 8 * self.message_payload_scalars,
                "physical_worker_runtime_s": self.physical_worker_runtime_s,
                "total_runtime_s": time.perf_counter() - started,
                "history_archive_enabled": False,
                "current_domain_only": True,
            }
            summary.update(self._strict_summary_fields())
            self._event("run_complete", summary=summary)
            assert self._event_log is not None
            self._event_log.close()
            _atomic_json(self.summary_path, summary)
            receipt = {
                "schema_version": "SNSD-v1.5-rolling-receipt-v4.4",
                "status": "complete",
                "method": self.config.reported_method,
                "seed": self.config.seed,
                "artifacts": {
                    "events.jsonl": {
                        "sha256": _sha256_file(self.events_path),
                        "bytes": self.events_path.stat().st_size,
                    },
                    "summary.json": {
                        "sha256": _sha256_file(self.summary_path),
                        "bytes": self.summary_path.stat().st_size,
                    },
                },
            }
            _atomic_json(self.receipt_path, receipt)
            return ExperimentResult(
                output_dir=str(self.output_dir),
                events_path=str(self.events_path),
                summary_path=str(self.summary_path),
                receipt_path=str(self.receipt_path),
                summary=summary,
            )
        finally:
            if self._event_log is not None:
                self._event_log.close()


def run_rolling_experiment(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    allow_untrained_smoke: bool = False,
) -> ExperimentResult:
    if method not in ROLLING_METHODS:
        raise ValueError("unknown rolling method: {}".format(method))
    one_shot = method == "OneShotComm"
    frozen = method == "FrozenAnchorComm"
    base_method = "Full" if one_shot or frozen else method
    return RollingNegotiationProtocol(
        RollingExperimentConfig(
            data_path=data_path,
            output_dir=output_dir,
            method=base_method,
            seed=seed,
            smoke=smoke,
            oracle=oracle,
            checkpoint=checkpoint,
            device=device,
            allow_untrained_smoke=allow_untrained_smoke,
            one_shot_comm=one_shot,
            frozen_anchor_comm=frozen,
        )
    ).run()


__all__ = [
    "ROLLING_METHODS",
    "RollingExperimentConfig",
    "RollingNegotiationProtocol",
    "run_rolling_experiment",
]
