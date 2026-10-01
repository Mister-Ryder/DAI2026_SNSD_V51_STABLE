"""Recurrent six-step learned negotiation runner.

Each macro round purchases one proposal per agent in six sequential steps.
Step 0 uses only public/static state.  Every purchase is followed by exact
current-domain chain Max-Sum, creating an identifiable boundary message for
the next counterproposal step.  Two formal macros therefore retain the
registered 72-call budget: ``2 * 6 steps * 6 agents * Q=1``.
"""

from __future__ import annotations

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
    METHODS,
    ExperimentResult,
    _RealizedRecord,
    _atomic_json,
    _deduplicate_records,
    _sha256_file,
)
from .checkpoint_policy_protocol import CheckpointPolicyConfig, CheckpointPolicyProtocol
from .messages import ProposalEngine


RECURRENT_METHODS = tuple(METHODS) + ("OneShotComm",)


@dataclass(frozen=True)
class RecurrentNegotiationConfig(CheckpointPolicyConfig):
    one_shot_comm: bool = False

    @property
    def negotiation_steps(self) -> int:
        return 6

    @property
    def query_budget(self) -> int:
        return 1

    @property
    def expected_online_calls(self) -> int:
        return self.macro_rounds * self.negotiation_steps * self.num_agents

    @property
    def reported_method(self) -> str:
        return "OneShotComm" if self.one_shot_comm else self.method


class RecurrentNegotiationProtocol(CheckpointPolicyProtocol):
    config: RecurrentNegotiationConfig

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        fields.update(
            {
                "schema_version": "SNSD-v1.5-recurrent-learned-run-v4",
                "method": self.config.reported_method,
                "negotiation_steps_per_macro": self.config.negotiation_steps,
                "waves_per_macro": self.config.negotiation_steps,
                "query_budget_per_agent_step": 1,
                "recurrent_current_domain": True,
                "one_shot_comm": bool(self.config.one_shot_comm),
            }
        )
        return fields

    @staticmethod
    def _zero_pressure_maps(
        pressure_maps: Mapping[int, Mapping[str, Mapping[int, Any]]]
    ) -> Dict[int, Dict[str, Dict[int, Any]]]:
        result = {}
        for agent_id, channels in pressure_maps.items():
            nodes = tuple(int(node) for node in channels["full"])
            zeros = {node: 0.0 for node in nodes}
            unknown = {node: False for node in nodes}
            result[int(agent_id)] = {
                "full": dict(zeros),
                "benefit": dict(zeros),
                "conflict": dict(zeros),
                "opportunity": dict(zeros),
                "identifiable_full": dict(unknown),
                "identifiable_benefit": dict(unknown),
                "identifiable_conflict": dict(unknown),
                "identifiable_opportunity": dict(unknown),
            }
        return result

    def _pressure_for_step(self, step: int, pressure_maps):
        if int(step) == 0:
            return None, "none"
        if self.config.one_shot_comm and int(step) >= 2:
            return self._zero_pressure_maps(pressure_maps), "zero"
        if self.behavior.peer_pressure == "none":
            return self._zero_pressure_maps(pressure_maps), "zero"
        return pressure_maps, self.behavior.wave_b_intervention

    def _count_factor_communication(self, step: int) -> bool:
        if not self.behavior.wave_a_message_pass:
            return False
        if self.config.one_shot_comm:
            # The factor pass after step 0 creates the sole real message used
            # at step 1. Later factor passes remain exact internal selection.
            return int(step) == 0
        return True

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
                raise ValueError("C3 recurrent protocol requires an adjacent chain")
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
            incumbent_ids = {
                agent_id: record.candidate_id
                for agent_id, record in incumbent_records.items()
            }
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
            incumbent_selected = tuple(
                sorted(
                    node
                    for record in incumbent_records.values()
                    for node in record.result.selected_ids
                )
            )
            initial_check = verify_selected_ids(self.graph, incumbent_selected)
            if not initial_check.feasible:
                raise RuntimeError("initial incumbent invalid: {}".format(initial_check.message))
            self.incumbent_objective = self._objective(incumbent_selected)
            initial_objective = self.incumbent_objective

            round_metrics = []
            negotiation_metrics = []
            for macro_round in range(self.config.macro_rounds):
                current = {
                    i: _deduplicate_records([incumbent_records[i], safe_records[i]])
                    for i in range(self.config.num_agents)
                }
                pressure = None
                last_factor = None
                last_order = None
                for step in range(self.config.negotiation_steps):
                    context_round = macro_round * self.config.negotiation_steps + step
                    step_pressure, message_mode = self._pressure_for_step(step, pressure)
                    self._event(
                        "negotiation_step_start",
                        macro_round=macro_round,
                        negotiation_step=step,
                        engine_round_context=context_round,
                        message_mode=message_mode,
                        current_domain_counts=[len(current[i]) for i in range(6)],
                    )
                    purchased = self._query_wave(
                        "A" if step == 0 else "B",
                        context_round,
                        generators,
                        engines,
                        incumbent_records,
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
                        incumbent_ids,
                        safe_ids,
                        cut_edges_by_link,
                        label="macro_{}_step_{}_post_query".format(macro_round, step),
                        count_messages=message_pass,
                    )
                    pressure = self._pressure_maps(generators, last_order, last_factor)
                    step_row = {
                        "macro_round": macro_round,
                        "negotiation_step": step,
                        "engine_round_context": context_round,
                        "message_mode": message_mode,
                        "message_pass": message_pass,
                        "online_calls_after_step": self.online_calls,
                        "factor_value": last_factor.best_value,
                        "current_domain_counts": [len(current[i]) for i in range(6)],
                    }
                    negotiation_metrics.append(step_row)
                    self._event("negotiation_step", **step_row)

                assert last_factor is not None and last_order is not None
                if self.behavior.final_selection == "independent":
                    chosen = {}
                    for agent_id in range(self.config.num_agents):
                        feasible = [
                            item
                            for item in current[agent_id]
                            if item.result.status == "feasible"
                        ]
                        chosen[agent_id] = max(
                            feasible,
                            key=lambda item: (float(item.result.objective), item.candidate_id),
                        )
                    final_domain_value = sum(
                        float(item.result.objective) for item in chosen.values()
                    )
                else:
                    chosen = self._chosen_records(last_factor, last_order)
                    final_domain_value = last_factor.best_value

                proposed_ids, shielded_records, shield = self._shield(chosen, generators)
                proposed_objective = float(shield["post_shield_objective"])
                accepted = proposed_objective > self.incumbent_objective + 1.0e-9
                if accepted:
                    incumbent_records = shielded_records
                    incumbent_ids = {
                        i: incumbent_records[i].candidate_id
                        for i in range(self.config.num_agents)
                    }
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
                "schema_version": "SNSD-v1.5-recurrent-receipt-v4",
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


def run_recurrent_experiment(
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
    one_shot = method == "OneShotComm"
    base_method = "Full" if one_shot else method
    if method not in RECURRENT_METHODS:
        raise ValueError("unknown recurrent method: {}".format(method))
    return RecurrentNegotiationProtocol(
        RecurrentNegotiationConfig(
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
        )
    ).run()


__all__ = [
    "RECURRENT_METHODS",
    "RecurrentNegotiationConfig",
    "RecurrentNegotiationProtocol",
    "run_recurrent_experiment",
]
