"""Private-elite, advantage/conflict-gated soft anchors for SNSD v4.7.

This prototype keeps the v4.6 fixed 12-step recurrent schedule and the exact
current-domain chain Max-Sum selector.  It changes only factorial factor B.
The exact joint tuple is advice for the next proposal wave, not a compulsory
replacement for every agent's private search anchor.

For agent ``i`` after step ``t``, let ``E_i`` be its best locally feasible
candidate observed up to that step and let ``J_i`` be its component of the
current exact Max-Sum tuple.  All quantities below are computed from the
current candidate domain (there is no next-step or future information):

``A_i = F_i(J_i) - F_i(E_i)``
    advantage of the best feasible chain completion forced to contain the
    joint candidate rather than the private elite;
``L_i = max(0, u_i(E_i) - u_i(J_i))``
    private local-objective loss caused by adopting the joint candidate; and
``C_i = OC_i(J_i)``
    absolute adjacent-branch opportunity cost of the joint candidate (the
    conflict gate).

The next proposal anchor consumes ``J_i`` exactly when

``A_i > tau`` and ``A_i >= lambda_loss * L_i`` and
``C_i <= lambda_conflict * A_i``.

Otherwise it consumes ``E_i``.  Proposal anchors are deliberately private
and need not form a committed feasible tuple; only the exact Max-Sum tuple can
be selected at macro commit.  A separate, feasible factor-reference tuple is
therefore retained for factor diagnostics and pressure-message construction.

The strict 2x2 remains comparable:

* Full: A=pressure communication, B=soft-anchor feedback;
* FrozenAnchorComm: A only, with macro-start proposal anchors;
* SelectionOnly: B only; and
* FrozenAnchorSelectionOnly: neither A nor B.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
from .rolling_protocol import _joint_hash
from .long_horizon_protocol import (
    LONG_METHODS,
    LONG_NEGOTIATION_DEPTHS,
    LongNegotiationConfig,
    LongHorizonProtocol,
    _long_config,
)
from .messages import ProposalEngine


SOFT_ANCHOR_METHODS = tuple(LONG_METHODS) + ("NoPeerFinalShield",)


@dataclass(frozen=True)
class PrivateEliteConfig(LongNegotiationConfig):
    """v4.6 schedule plus scale-aware soft-anchor gate controls."""

    soft_anchor_advantage_floor: float = 1.0e-9
    soft_anchor_loss_cover: float = 5.0
    soft_anchor_conflict_slack: float = 0.10
    no_peer_final_shield: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.soft_anchor_advantage_floor) < 0.0:
            raise ValueError("soft_anchor_advantage_floor must be nonnegative")
        if float(self.soft_anchor_loss_cover) < 0.0:
            raise ValueError("soft_anchor_loss_cover must be nonnegative")
        if float(self.soft_anchor_conflict_slack) < 0.0:
            raise ValueError("soft_anchor_conflict_slack must be nonnegative")
        if self.no_peer_final_shield and self.method != "NoPeerComm":
            raise ValueError(
                "NoPeerFinalShield must use NoPeerComm as its base method"
            )

    @property
    def reported_method(self) -> str:
        if self.no_peer_final_shield:
            return "NoPeerFinalShield"
        return super().reported_method


class PrivateEliteProtocol(LongHorizonProtocol):
    """Long recurrent negotiation with persistent private-elite anchors."""

    config: PrivateEliteConfig

    def __init__(self, config: PrivateEliteConfig) -> None:
        super().__init__(config)
        self._soft_anchor_gate_rows = []  # type: List[Dict[str, Any]]
        self._soft_anchor_adoptions = 0
        self._soft_anchor_decisions = 0

    @staticmethod
    def _best_private_record(records: Iterable[Any]) -> Any:
        feasible = [
            record
            for record in records
            if record.result.status == "feasible"
        ]
        if not feasible:
            raise RuntimeError("private-elite domain has no feasible candidate")
        return max(
            feasible,
            key=lambda record: (
                float(record.result.objective),
                str(record.candidate_id),
            ),
        )

    def _update_private_elites(
        self,
        private_elites: Mapping[int, Any],
        records_by_agent: Mapping[int, Sequence[Any]],
    ) -> Dict[int, Any]:
        updated = {}
        for agent_id in range(self.config.num_agents):
            updated[agent_id] = self._best_private_record(
                [private_elites[agent_id]] + list(records_by_agent[agent_id])
            )
        return updated

    def _soft_anchor_decision(
        self,
        factor: Any,
        ordered_records: Sequence[Sequence[Any]],
        private_elites: Mapping[int, Any],
        macro_round: int,
        step: int,
    ) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
        """Apply the registered per-agent benefit and conflict gates."""

        anchors = {}
        rows = []
        eps = 1.0e-9
        if len(ordered_records) != self.config.num_agents:
            raise ValueError("soft-anchor tables must match num_agents")
        for agent_id in range(self.config.num_agents):
            records = ordered_records[agent_id]
            index_by_id = {
                record.candidate_id: index
                for index, record in enumerate(records)
            }
            elite = private_elites[agent_id]
            elite_index = index_by_id[elite.candidate_id]
            joint_index = int(factor.best_indices[agent_id])
            joint = records[joint_index]

            elite_completion = float(
                factor.forced_best_completions[agent_id][elite_index]
            )
            joint_completion = float(
                factor.forced_best_completions[agent_id][joint_index]
            )
            advantage = joint_completion - elite_completion
            local_loss = max(
                0.0,
                float(elite.result.objective) - float(joint.result.objective),
            )
            elite_cost = float(factor.opportunity_costs[agent_id][elite_index])
            joint_cost = float(factor.opportunity_costs[agent_id][joint_index])
            conflict_load = max(0.0, joint_cost)

            benefit_gate = bool(
                advantage > float(self.config.soft_anchor_advantage_floor)
                and advantage + eps
                >= float(self.config.soft_anchor_loss_cover) * local_loss
            )
            conflict_gate = bool(
                conflict_load
                <= float(self.config.soft_anchor_conflict_slack)
                * max(0.0, advantage)
                + eps
            )
            distinct = joint.candidate_id != elite.candidate_id
            adopted = bool(distinct and benefit_gate and conflict_gate)
            anchors[agent_id] = joint if adopted else elite
            rows.append(
                {
                    "macro_round": int(macro_round),
                    "source_step": int(step),
                    "target_step": int(step) + 1,
                    "agent_id": int(agent_id),
                    "elite_candidate_id": elite.candidate_id,
                    "joint_candidate_id": joint.candidate_id,
                    "selected_anchor_candidate_id": anchors[agent_id].candidate_id,
                    "forced_completion_advantage": float(advantage),
                    "private_local_loss": float(local_loss),
                    "elite_opportunity_cost": float(elite_cost),
                    "joint_opportunity_cost": float(joint_cost),
                    "conflict_load": float(conflict_load),
                    "benefit_gate": benefit_gate,
                    "conflict_gate": conflict_gate,
                    "joint_adopted": adopted,
                }
            )
        return anchors, rows

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        adoption_rate = (
            float(self._soft_anchor_adoptions)
            / float(self._soft_anchor_decisions)
            if self._soft_anchor_decisions
            else 0.0
        )
        fields.update(
            {
                "schema_version": "SNSD-v1.5-soft-anchor-run-v4.7",
                "soft_anchor_prototype": True,
                "factor_B_definition": (
                    "private_elite_plus_advantage_conflict_gated_soft_anchor"
                ),
                "private_elite_memory": "persistent_across_macro_rounds",
                "proposal_anchor_may_be_nonjoint": True,
                "macro_commit_selector": (
                    "independent_then_shield"
                    if self.behavior.final_selection == "independent"
                    and not self.config.no_peer_final_shield
                    else "exact_current_domain_chain_max_sum"
                ),
                "soft_anchor_no_future_information": True,
                "soft_anchor_gate_formula": (
                    "A=F(J)-F(E); L=max(0,u(E)-u(J)); "
                    "C=OC(J); adopt J iff "
                    "A>tau and A>=lambda_loss*L and "
                    "C<=lambda_conflict*A"
                ),
                "soft_anchor_advantage_floor": float(
                    self.config.soft_anchor_advantage_floor
                ),
                "soft_anchor_loss_cover": float(
                    self.config.soft_anchor_loss_cover
                ),
                "soft_anchor_conflict_slack": float(
                    self.config.soft_anchor_conflict_slack
                ),
                "soft_anchor_decisions": int(self._soft_anchor_decisions),
                "soft_anchor_joint_adoptions": int(
                    self._soft_anchor_adoptions
                ),
                "soft_anchor_joint_adoption_rate": float(adoption_rate),
                "soft_anchor_gate_trace": list(self._soft_anchor_gate_rows),
                "no_peer_final_shield_baseline": bool(
                    self.config.no_peer_final_shield
                ),
                "synthetic_repair_cross_macro_inheritance": bool(
                    self.behavior.final_selection == "independent"
                    and not self.config.no_peer_final_shield
                ),
            }
        )
        return fields

    def run(self) -> ExperimentResult:
        """Run v4.6 unchanged except for the proposal-anchor update rule."""

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
                raise ValueError("C3 soft-anchor protocol requires an adjacent chain")
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
                soft_anchor_feedback=True,
                private_elite_memory=True,
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
                raise RuntimeError(
                    "initial incumbent invalid: {}".format(initial_check.message)
                )
            self.incumbent_objective = self._objective(incumbent_selected)
            initial_objective = self.incumbent_objective
            private_elites = dict(incumbent_records)
            final_output_shield = None

            round_metrics = []
            negotiation_metrics = []
            for macro_round in range(self.config.macro_rounds):
                macro_anchor_records = dict(incumbent_records)
                proposal_anchor_records = dict(incumbent_records)
                factor_reference_records = dict(incumbent_records)
                current = {
                    agent_id: _deduplicate_records(
                        [
                            incumbent_records[agent_id],
                            safe_records[agent_id],
                            private_elites[agent_id],
                        ]
                    )
                    for agent_id in range(self.config.num_agents)
                }
                pressure = None
                last_factor = None
                last_order = None
                for step in range(self.config.negotiation_steps):
                    context_round = (
                        macro_round * self.config.negotiation_steps + step
                    )
                    anchor_records = (
                        macro_anchor_records
                        if self.config.frozen_anchor_comm
                        else proposal_anchor_records
                    )
                    anchor_ids = {
                        agent_id: anchor_records[agent_id].candidate_id
                        for agent_id in range(self.config.num_agents)
                    }
                    factor_reference_ids = {
                        agent_id: factor_reference_records[agent_id].candidate_id
                        for agent_id in range(self.config.num_agents)
                    }
                    anchor_hash = _joint_hash(anchor_records)
                    step_pressure, message_mode = self._pressure_for_step(
                        step, pressure
                    )
                    self._event(
                        "negotiation_step_start",
                        macro_round=macro_round,
                        negotiation_step=step,
                        engine_round_context=context_round,
                        message_mode=message_mode,
                        anchor_hash=anchor_hash,
                        anchor_signature_hashes=[
                            anchor_records[i].candidate_id
                            for i in range(self.config.num_agents)
                        ],
                        anchor_selected_boundary_ids=[
                            list(anchor_records[i].signature.selected_ids)
                            for i in range(self.config.num_agents)
                        ],
                        factor_reference_hash=_joint_hash(
                            factor_reference_records
                        ),
                        current_domain_counts=[
                            len(current[i]) for i in range(self.config.num_agents)
                        ],
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
                    private_elites = self._update_private_elites(
                        private_elites, current
                    )

                    message_pass = self._count_factor_communication(step)
                    last_factor, last_order = self._factor_pass(
                        current,
                        factor_reference_ids,
                        safe_ids,
                        cut_edges_by_link,
                        label="macro_{}_step_{}_post_query".format(
                            macro_round, step
                        ),
                        count_messages=message_pass,
                    )
                    pressure = self._pressure_maps(
                        generators, last_order, last_factor
                    )
                    exact_joint = self._chosen_records(last_factor, last_order)
                    factor_reference_records = exact_joint

                    gate_rows = []  # type: List[Dict[str, Any]]
                    if step < self.config.negotiation_steps - 1:
                        if self.behavior.final_selection == "independent":
                            next_proposal = dict(private_elites)
                        elif self.config.frozen_anchor_comm:
                            next_proposal = dict(macro_anchor_records)
                        else:
                            next_proposal, gate_rows = self._soft_anchor_decision(
                                last_factor,
                                last_order,
                                private_elites,
                                macro_round,
                                step,
                            )
                            self._soft_anchor_gate_rows.extend(gate_rows)
                            self._soft_anchor_decisions += len(gate_rows)
                            self._soft_anchor_adoptions += sum(
                                1 for row in gate_rows if row["joint_adopted"]
                            )
                    else:
                        # The terminal factor is used only for exact macro
                        # selection.  It is never emitted as next-step advice.
                        next_proposal = dict(anchor_records)

                    proposal_selected = self._selected_union(next_proposal)
                    proposal_check = verify_selected_ids(
                        self.graph, proposal_selected
                    )
                    changed_agents = [
                        agent_id
                        for agent_id in range(self.config.num_agents)
                        if next_proposal[agent_id].candidate_id
                        != anchor_ids[agent_id]
                    ]
                    proposal_anchor_records = next_proposal
                    step_row = {
                        "macro_round": macro_round,
                        "negotiation_step": step,
                        "engine_round_context": context_round,
                        "message_mode": message_mode,
                        "message_pass": message_pass,
                        "online_calls_after_step": self.online_calls,
                        "anchor_hash": anchor_hash,
                        "anchor_signature_hashes": [
                            anchor_ids[i] for i in range(self.config.num_agents)
                        ],
                        "anchor_selected_boundary_ids": [
                            list(anchor_records[i].signature.selected_ids)
                            for i in range(self.config.num_agents)
                        ],
                        "working_hash": _joint_hash(proposal_anchor_records),
                        "working_signature_hashes": [
                            proposal_anchor_records[i].candidate_id
                            for i in range(self.config.num_agents)
                        ],
                        "working_selected_boundary_ids": [
                            list(
                                proposal_anchor_records[i].signature.selected_ids
                            )
                            for i in range(self.config.num_agents)
                        ],
                        "anchor_changed_agents": changed_agents,
                        "working_objective": self._objective(proposal_selected),
                        "working_feasible": bool(proposal_check.feasible),
                        "proposal_anchor_is_committed_solution": False,
                        "factor_reference_hash": _joint_hash(
                            factor_reference_records
                        ),
                        "soft_gate_rows": gate_rows,
                        "soft_joint_adoptions": sum(
                            1 for row in gate_rows if row["joint_adopted"]
                        ),
                        "factor_domain_objective": last_factor.best_value,
                        "factor_value": last_factor.best_value,
                        "step_objective": last_factor.best_value,
                        "current_domain_counts": [
                            len(current[i]) for i in range(self.config.num_agents)
                        ],
                    }
                    negotiation_metrics.append(step_row)
                    self._event("negotiation_step", **step_row)

                assert last_factor is not None and last_order is not None
                if (
                    self.behavior.final_selection == "independent"
                    and not self.config.no_peer_final_shield
                ):
                    chosen = self._independent_working(current)
                    final_domain_value = sum(
                        float(record.result.objective)
                        for record in chosen.values()
                    )
                else:
                    chosen = self._chosen_records(last_factor, last_order)
                    final_domain_value = last_factor.best_value
                proposed_ids, shielded_records, shield = self._shield(
                    chosen, generators
                )
                proposed_objective = float(shield["post_shield_objective"])
                accepted = (
                    proposed_objective > self.incumbent_objective + 1.0e-9
                )
                if accepted:
                    incumbent_records = shielded_records
                    incumbent_selected = proposed_ids
                    self.incumbent_objective = proposed_objective
                    private_elites = self._update_private_elites(
                        private_elites,
                        {
                            i: [incumbent_records[i]]
                            for i in range(self.config.num_agents)
                        },
                    )
                if self.objective_trace:
                    self.objective_trace[-1] = float(self.incumbent_objective)
                row = {
                    "macro_round": macro_round,
                    "final_current_domain_value": final_domain_value,
                    "proposed_objective": proposed_objective,
                    "accepted": accepted,
                    "incumbent_objective": self.incumbent_objective,
                    "current_domain_counts": [
                        len(current[i]) for i in range(self.config.num_agents)
                    ],
                    "shield": shield,
                }
                round_metrics.append(row)
                self._event("macro_commit", **row)

            if self.config.no_peer_final_shield:
                # This deliberately occurs after the final macro: repaired
                # synthetic records may improve the feasible reported output,
                # but cannot enter any later proposal or factor domain.
                final_ids, _, final_output_shield = self._shield(
                    private_elites, generators
                )
                final_output_objective = float(
                    final_output_shield["post_shield_objective"]
                )
                final_output_accepted = bool(
                    final_output_objective
                    > self.incumbent_objective + 1.0e-9
                )
                if final_output_accepted:
                    incumbent_selected = final_ids
                    self.incumbent_objective = final_output_objective
                final_output_shield = dict(final_output_shield)
                final_output_shield["accepted_as_final_output"] = bool(
                    final_output_accepted
                )
                final_output_shield[
                    "synthetic_records_inherited"
                ] = False
                self._event(
                    "final_output_shield",
                    method=self.config.reported_method,
                    **final_output_shield
                )

            if self.online_calls != self.config.expected_online_calls:
                raise RuntimeError(
                    "online-call invariant failed: expected {}, got {}".format(
                        self.config.expected_online_calls, self.online_calls
                    )
                )
            final_check = verify_selected_ids(self.graph, incumbent_selected)
            if not final_check.feasible:
                raise RuntimeError(
                    "final schedule invalid: {}".format(final_check.message)
                )
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
                "message_payload_bytes_float64": 8
                * self.message_payload_scalars,
                "physical_worker_runtime_s": self.physical_worker_runtime_s,
                "total_runtime_s": time.perf_counter() - started,
                "history_archive_enabled": False,
                "current_domain_only": True,
                "final_output_shield": final_output_shield,
            }
            summary.update(self._strict_summary_fields())
            self._event("run_complete", summary=summary)
            assert self._event_log is not None
            self._event_log.close()
            _atomic_json(self.summary_path, summary)
            receipt = {
                "schema_version": "SNSD-v1.5-soft-anchor-receipt-v4.7",
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


def _soft_anchor_config(
    data_path: str,
    output_dir: str,
    method: str,
    seed: int,
    smoke: bool,
    oracle: bool,
    checkpoint: Optional[str],
    device: str,
    allow_untrained_smoke: bool,
    negotiation_depth: int,
    macro_rounds: int,
    advantage_floor: float,
    loss_cover: float,
    conflict_slack: float,
    no_peer_final_shield: bool,
) -> PrivateEliteConfig:
    requested_method = method
    if requested_method == "NoPeerFinalShield":
        method = "NoPeerComm"
    base = _long_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        smoke=smoke,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=allow_untrained_smoke,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
    )
    values = dict(base.__dict__)
    values.update(
        {
            "soft_anchor_advantage_floor": advantage_floor,
            "soft_anchor_loss_cover": loss_cover,
            "soft_anchor_conflict_slack": conflict_slack,
            "no_peer_final_shield": no_peer_final_shield,
        }
    )
    return PrivateEliteConfig(**values)


def run_soft_anchor_experiment(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    smoke: bool = False,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    allow_untrained_smoke: bool = False,
    negotiation_depth: int = 12,
    macro_rounds: int = 4,
    advantage_floor: float = 1.0e-9,
    loss_cover: float = 5.0,
    conflict_slack: float = 0.10,
) -> ExperimentResult:
    config = _soft_anchor_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        smoke=smoke,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        allow_untrained_smoke=allow_untrained_smoke,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
        advantage_floor=advantage_floor,
        loss_cover=loss_cover,
        conflict_slack=conflict_slack,
        no_peer_final_shield=(method == "NoPeerFinalShield"),
    )
    return PrivateEliteProtocol(config).run()


__all__ = [
    "LONG_NEGOTIATION_DEPTHS",
    "SOFT_ANCHOR_METHODS",
    "PrivateEliteConfig",
    "PrivateEliteProtocol",
    "run_soft_anchor_experiment",
]
