"""SNSD v5.1 paired semantic-counterfactual base_protocol protocol.

For every late coordination state, v5.1 evaluates two structured boundary
cycles from exactly the same immutable inputs:

* C0: zero semantic credit;
* CA: the registered semantic intervention (real, shuffled, alternating, or
  the first-late-step-only OneShot treatment).

The exact safeguarded comparison observes current, J, R0, Rm, C0 and CA.
Factor-B-off cells compute C0/CA for diagnosis but inject neither.  Factor-B-
on cells always preserve C0 in the next factor domain.  CA is eligible only
when it strictly beats max(J,R0,Rm,C0), and it enters the next domain only
when the authoritative exact selector accepts it as the current anchor.
Consequently an unaccepted semantic response cannot pollute later search.

Both branches are coordination-only and add no paid worker query.  The formal
protocol remains complete C3, 12 steps x 4 macro rounds, 6 agents and exactly
288 paid worker calls per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from snsd_core.solver import LocalSolveResult

from .base_protocol import ExperimentResult, _RealizedRecord, _atomic_json, _sha256_file
from .structured_joint_protocol import (
    V50_FORMAL_METHODS,
    JointNegotiationConfig,
    StructuredJointProtocol,
    _field,
    _structured_joint_config,
    preflight_v50_full_c3,
)
from .boundary_semantic import run_semantic_boundary_cycle


FORMAL_METHODS = V50_FORMAL_METHODS


class SemanticSafeguardProtocol(StructuredJointProtocol):
    """Paired C0/CA evaluation with a strict no-pollution semantic gate."""

    config: JointNegotiationConfig

    def __init__(self, config: JointNegotiationConfig) -> None:
        super().__init__(config)
        self._v51_trace = []  # type: List[Dict[str, Any]]
        self._v51_late_decisions = 0
        self._v51_c0_evaluations = 0
        self._v51_ca_evaluations = 0
        self._v51_ca_deduplicated_evaluations = 0
        self._v51_treated_evaluations = 0
        self._v51_semantic_strict_wins = 0
        self._v51_semantic_accepts = 0
        self._v51_c0_selections = 0
        self._v51_ca_minus_c0 = []  # type: List[float]
        self._v51_ca_minus_c0_positive_steps = 0
        self._v51_semantic_gain_over_baseline = 0.0
        self._v51_endpoint_difference_steps = 0
        self._v51_endpoint_symmetric_difference = 0
        self._v51_refill_difference_steps = 0
        self._v51_refill_symmetric_difference = 0
        self._v51_distinct_microresponses_c0 = 0
        self._v51_distinct_microresponses_ca = 0
        self._v51_cross_branch_distinct_steps = 0
        self._v51_cross_branch_signature_difference = 0
        self._v51_payload_bytes_c0 = 0
        self._v51_payload_bytes_ca = 0
        self._v51_counterfactual_feedback_scalars = 0
        self._v51_semantic_credit_nonzero = 0
        self._v51_c0_domain_injections = 0
        self._v51_ca_domain_injections = 0
        self._v51_unaccepted_ca_domain_injections = 0

    def _semantic_mode(self, step: int) -> str:
        """Return the v5.1 treatment for the current *late* source step.

        The explicit override keeps the registered schedule independent of an
        older deployed v5.0 module whose OneShot condition was tied to global
        step zero.  The first late decision is source step
        ``private_exploit_steps - 1`` (5 under the formal protocol).
        """

        method = self.config.reported_method
        if method in (
            "SelectionOnly",
            "FrozenAnchorSelectionOnly",
            "NoPeerFinalShield",
        ):
            return "zero"
        if method == "ShuffledComm":
            return "shuffled"
        if method == "OneShotComm":
            return (
                "real"
                if int(step) == int(self.config.private_exploit_steps) - 1
                else "zero"
            )
        if method == "AlternatingConflictBenefit":
            return "conflict_only" if int(step) % 2 else "benefit_only"
        return "real"

    def _records_from_v51_cycle(
        self,
        cycle: Any,
        macro_round: int,
        step: int,
        branch: str,
    ) -> Optional[Dict[int, _RealizedRecord]]:
        selected_by_agent = _field(cycle, "candidate_selected_ids_by_agent")
        if not selected_by_agent:
            selected_by_agent = _field(cycle, "selected_ids_by_agent")
        if not selected_by_agent or not bool(_field(cycle, "feasible", False)):
            return None
        records = {}  # type: Dict[int, _RealizedRecord]
        union = []  # type: List[int]
        for agent_id in range(self.config.num_agents):
            raw = selected_by_agent.get(agent_id)
            if raw is None:
                raw = selected_by_agent.get(str(agent_id), ())
            local_ids = tuple(sorted(int(node) for node in raw))
            if any(
                int(self.partitions.partition_id[node]) != int(agent_id)
                for node in local_ids
            ):
                raise RuntimeError(
                    "v5.1 {} cycle returned a cross-owner local record".format(
                        branch
                    )
                )
            union.extend(local_ids)
            generator = self._repair_generators[agent_id]
            boundary = set(int(node) for node in generator.boundary)
            signature = generator.make_signature(
                [node for node in local_ids if node in boundary]
            )
            result = LocalSolveResult(
                agent_id=agent_id,
                signature_hash=signature.signature_hash,
                status="feasible",
                selected_ids=local_ids,
                objective=self._objective(local_ids),
                runtime_s=0.0,
                attempted_moves=0,
                seed=self.config.seed,
                message="v5.1 paired semantic boundary response {}".format(branch),
            )
            records[agent_id] = _RealizedRecord(
                signature,
                result,
                "v51_semantic_boundary_{}".format(branch.lower()),
                int(macro_round),
                "late_step_{}".format(step),
            )
        selected = tuple(sorted(union))
        check = self._verified_objective(records, "v51_cycle_{}".format(branch))
        if check[0] != selected:
            raise RuntimeError("v5.1 cycle union changed during exact verification")
        return records

    @staticmethod
    def _cycle_set(cycle: Any, field: str) -> Set[int]:
        return set(int(value) for value in (_field(cycle, field, ()) or ()))

    @staticmethod
    def _response_signatures(cycle: Any) -> Set[str]:
        return set(
            str(value)
            for value in (_field(cycle, "response_signatures", ()) or ())
        )

    def _run_cycle(
        self,
        semantic_mode: str,
        private_elites: Mapping[int, _RealizedRecord],
        ordered_records: Sequence[Sequence[_RealizedRecord]],
        joint_records: Mapping[int, _RealizedRecord],
        channels: Optional[Mapping[int, Mapping[str, Mapping[int, float]]]],
        top_k: int,
        micro_rounds: int,
        macro_round: int,
        step: int,
    ) -> Any:
        return run_semantic_boundary_cycle(
            graph=self.graph,
            partitions=self.partitions,
            arcs=self.dataset.arcs,
            generators=self._repair_generators,
            private_elites=private_elites,
            current_records=ordered_records,
            base_records=joint_records,
            message_channels=channels,
            semantic_mode=str(semantic_mode),
            top_k=int(top_k),
            micro_rounds=int(micro_rounds),
            max_deletions=int(self.config.max_coalition_deletions),
            seed=int(self.config.seed),
            macro_round=int(macro_round),
            step=int(step),
        )

    def _soft_anchor_decision(
        self,
        factor: Any,
        ordered_records: Sequence[Sequence[_RealizedRecord]],
        private_elites: Mapping[int, _RealizedRecord],
        macro_round: int,
        step: int,
    ) -> Tuple[Dict[int, _RealizedRecord], List[Dict[str, Any]]]:
        target_step = int(step) + 1
        if target_step < int(self.config.private_exploit_steps):
            self._factor_history.setdefault(int(macro_round), []).append(
                float(factor.best_value)
            )
            self._pending_repair_records = None
            return self._early_private_response(private_elites, macro_round, step)

        joint_records = self._chosen_records(factor, ordered_records)
        reference_records = self._factor_reference_records(factor, ordered_records)
        _, reference_objective = self._verified_objective(
            reference_records, "current_reference"
        )
        _, joint_objective = self._verified_objective(joint_records, "exact_joint_J")

        value_ids, value_records, value_metrics = self._repair_private_elite_union(
            private_elites,
            macro_round,
            step,
            communication_conditioned=False,
        )
        value_objective = float(self._objective(value_ids))

        channels = None
        message_hashes = []  # type: List[str]
        message_records = None
        message_objective = None
        message_metrics = None
        if self.config.reported_method != "NoPeerFinalShield":
            channels, message_hashes = self._repair_messages(macro_round, step)
            message_ids, message_records, message_metrics = (
                self._repair_private_elite_union(
                    private_elites,
                    macro_round,
                    step,
                    communication_conditioned=True,
                )
            )
            message_objective = float(self._objective(message_ids))

        top_k, micro_rounds, plateau = self._adapt_depth(
            macro_round, float(factor.best_value)
        )
        semantic_mode = self._semantic_mode(step)

        # Paired counterfactuals use the same object references and state.
        cycle_c0 = self._run_cycle(
            "zero",
            private_elites,
            ordered_records,
            joint_records,
            channels,
            top_k,
            micro_rounds,
            macro_round,
            step,
        )
        c0_records = self._records_from_v51_cycle(
            cycle_c0, macro_round, step, "C0"
        )
        c0_objective = None
        if c0_records is not None:
            _, c0_objective = self._verified_objective(c0_records, "v51_C0")

        if semantic_mode == "zero":
            # A0 is a literal paired-control identity, not a second stochastic
            # call that could differ through an implementation side effect.
            cycle_ca = cycle_c0
            ca_records = c0_records
            ca_objective = c0_objective
            self._v51_ca_deduplicated_evaluations += 1
        else:
            cycle_ca = self._run_cycle(
                semantic_mode,
                private_elites,
                ordered_records,
                joint_records,
                channels,
                top_k,
                micro_rounds,
                macro_round,
                step,
            )
            ca_records = self._records_from_v51_cycle(
                cycle_ca, macro_round, step, "CA"
            )
            ca_objective = None
            if ca_records is not None:
                _, ca_objective = self._verified_objective(ca_records, "v51_CA")
            self._v51_treated_evaluations += 1

        self._v51_late_decisions += 1
        self._v51_c0_evaluations += 1
        self._v51_ca_evaluations += 1

        semantic_baseline_values = [float(joint_objective), float(value_objective)]
        if message_objective is not None:
            semantic_baseline_values.append(float(message_objective))
        if c0_objective is not None:
            semantic_baseline_values.append(float(c0_objective))
        semantic_baseline = max(semantic_baseline_values)
        semantic_strict_win = bool(
            semantic_mode != "zero"
            and ca_objective is not None
            and float(ca_objective) > float(semantic_baseline) + 1.0e-9
        )
        semantic_gain = (
            float(ca_objective) - float(semantic_baseline)
            if semantic_strict_win and ca_objective is not None
            else 0.0
        )
        self._v51_semantic_strict_wins += int(semantic_strict_win)
        self._v51_semantic_gain_over_baseline += float(semantic_gain)

        candidates = [
            ("current_reference", reference_records, float(reference_objective)),
            ("exact_joint_J", joint_records, float(joint_objective)),
            ("common_repair_R0", value_records, float(value_objective)),
        ]
        if message_records is not None and message_objective is not None:
            candidates.append(
                ("message_repair_Rm", message_records, float(message_objective))
            )
        if self._coalition_enabled and c0_records is not None and c0_objective is not None:
            candidates.append(
                ("zero_semantic_coalition_C0", c0_records, float(c0_objective))
            )
        if (
            self._coalition_enabled
            and semantic_strict_win
            and ca_records is not None
            and ca_objective is not None
        ):
            candidates.append(
                ("treated_semantic_coalition_CA", ca_records, float(ca_objective))
            )

        selected_kind, selected_records, selected_objective = candidates[0]
        for kind, records, objective in candidates[1:]:
            if float(objective) > float(selected_objective) + 1.0e-9:
                selected_kind, selected_records, selected_objective = (
                    kind,
                    records,
                    float(objective),
                )
        if float(selected_objective) + 1.0e-9 < float(reference_objective):
            raise RuntimeError("v5.1 exact union violated monotonic commit")

        semantic_accepted = selected_kind == "treated_semantic_coalition_CA"
        self._v51_semantic_accepts += int(semantic_accepted)
        self._v51_c0_selections += int(
            selected_kind == "zero_semantic_coalition_C0"
        )

        # R0/Rm keep the v4.9 safeguard.  B1 always injects C0.  CA is
        # injected only when it is the exact accepted anchor; unaccepted CA
        # components never enter the future domain.
        pending = {
            agent_id: [value_records[agent_id]]
            for agent_id in range(self.config.num_agents)
        }
        if message_records is not None:
            for agent_id in range(self.config.num_agents):
                pending[agent_id].append(message_records[agent_id])
        if self._coalition_enabled and c0_records is not None:
            for agent_id in range(self.config.num_agents):
                pending[agent_id].append(c0_records[agent_id])
                self._v51_c0_domain_injections += 1
        if semantic_accepted:
            if not semantic_strict_win or ca_records is None:
                raise RuntimeError("accepted CA did not pass the semantic strict-win gate")
            for agent_id in range(self.config.num_agents):
                pending[agent_id].append(ca_records[agent_id])
                self._v51_ca_domain_injections += 1
        self._pending_repair_records = pending

        c0_removed = self._cycle_set(cycle_c0, "removed_ids")
        ca_removed = self._cycle_set(cycle_ca, "removed_ids")
        c0_refill = self._cycle_set(cycle_c0, "refill_ids")
        ca_refill = self._cycle_set(cycle_ca, "refill_ids")
        endpoint_difference = sorted(c0_removed.symmetric_difference(ca_removed))
        refill_difference = sorted(c0_refill.symmetric_difference(ca_refill))
        self._v51_endpoint_difference_steps += int(bool(endpoint_difference))
        self._v51_endpoint_symmetric_difference += len(endpoint_difference)
        self._v51_refill_difference_steps += int(bool(refill_difference))
        self._v51_refill_symmetric_difference += len(refill_difference)

        c0_signatures = self._response_signatures(cycle_c0)
        ca_signatures = self._response_signatures(cycle_ca)
        signature_difference = sorted(c0_signatures.symmetric_difference(ca_signatures))
        self._v51_cross_branch_distinct_steps += int(bool(signature_difference))
        self._v51_cross_branch_signature_difference += len(signature_difference)
        distinct_c0 = int(_field(cycle_c0, "distinct_response_count", 0) or 0)
        distinct_ca = int(_field(cycle_ca, "distinct_response_count", 0) or 0)
        self._v51_distinct_microresponses_c0 += distinct_c0
        self._v51_distinct_microresponses_ca += distinct_ca

        payload_c0 = int(_field(cycle_c0, "payload_bytes", 0) or 0)
        payload_ca = int(_field(cycle_ca, "payload_bytes", 0) or 0)
        self._v51_payload_bytes_c0 += payload_c0
        self._v51_payload_bytes_ca += payload_ca
        self._v51_counterfactual_feedback_scalars += int(
            _field(cycle_c0, "counterfactual_feedback_scalars", 0) or 0
        )
        if cycle_ca is not cycle_c0:
            self._v51_counterfactual_feedback_scalars += int(
                _field(cycle_ca, "counterfactual_feedback_scalars", 0) or 0
            )
        self._v51_semantic_credit_nonzero += int(
            _field(cycle_ca, "semantic_credit_nonzero", 0) or 0
        )

        ca_minus_c0 = (
            float(ca_objective) - float(c0_objective)
            if ca_objective is not None and c0_objective is not None
            else 0.0
        )
        self._v51_ca_minus_c0.append(float(ca_minus_c0))
        self._v51_ca_minus_c0_positive_steps += int(ca_minus_c0 > 1.0e-9)

        self._repair_evaluations += 1
        self._repair_selected += int(
            selected_kind not in ("current_reference", "exact_joint_J")
        )
        self._joint_selected += int(selected_kind == "exact_joint_J")
        self._value_repair_selected += int(selected_kind == "common_repair_R0")
        self._message_repair_selected += int(selected_kind == "message_repair_Rm")
        changed = sum(
            1
            for agent_id in range(self.config.num_agents)
            if selected_records[agent_id].candidate_id
            != private_elites[agent_id].candidate_id
        )
        self._repair_feedback_changes += int(changed)

        compact = {
            "macro_round": int(macro_round),
            "source_step": int(step),
            "target_step": int(target_step),
            "stage": "late_paired_semantic_counterfactual",
            "semantic_mode": str(semantic_mode),
            "same_state_paired_evaluation": True,
            "plateau": bool(plateau),
            "packet_top_k": int(top_k),
            "microcycles": int(micro_rounds),
            "factor_B_cycle_injection_enabled": bool(self._coalition_enabled),
            "reference_objective": float(reference_objective),
            "exact_joint_objective_J": float(joint_objective),
            "common_repair_objective_R0": float(value_objective),
            "message_repair_objective_Rm": message_objective,
            "zero_semantic_objective_C0": c0_objective,
            "treated_semantic_objective_CA": ca_objective,
            "ca_minus_c0": float(ca_minus_c0),
            "semantic_baseline_max_J_R0_Rm_C0": float(semantic_baseline),
            "semantic_strict_win": bool(semantic_strict_win),
            "semantic_gain_over_baseline": float(semantic_gain),
            "semantic_accepted": bool(semantic_accepted),
            "selected_kind": str(selected_kind),
            "selected_objective": float(selected_objective),
            "c0_best_microcycle": int(_field(cycle_c0, "best_microcycle", -1)),
            "ca_best_microcycle": int(_field(cycle_ca, "best_microcycle", -1)),
            "c0_distinct_microresponses": int(distinct_c0),
            "ca_distinct_microresponses": int(distinct_ca),
            "cross_branch_distinct": bool(signature_difference),
            "cross_branch_signature_difference_count": len(signature_difference),
            "endpoint_symmetric_difference": endpoint_difference,
            "refill_symmetric_difference": refill_difference,
            "c0_removed_ids": sorted(c0_removed),
            "ca_removed_ids": sorted(ca_removed),
            "c0_refill_ids": sorted(c0_refill),
            "ca_refill_ids": sorted(ca_refill),
            "c0_payload_bytes": int(payload_c0),
            "ca_payload_bytes": int(payload_ca),
            "ca_semantic_credit_nonzero": int(
                _field(cycle_ca, "semantic_credit_nonzero", 0) or 0
            ),
            "ca_semantic_credit_linf": float(
                _field(cycle_ca, "semantic_credit_linf", 0.0) or 0.0
            ),
            "message_hashes": list(message_hashes),
            "ca_domain_injected": bool(semantic_accepted),
            "unaccepted_ca_domain_injected": False,
        }
        self._v51_trace.append(compact)
        self._repair_trace.append(dict(compact))
        self._event(
            "v51_paired_semantic_counterfactual",
            c0_micro_trace=list(_field(cycle_c0, "micro_trace", ()) or ()),
            ca_micro_trace=(
                list(_field(cycle_ca, "micro_trace", ()) or ())
                if cycle_ca is not cycle_c0
                else "deduplicated_equals_C0"
            ),
            c0_response_signatures=sorted(c0_signatures),
            ca_response_signatures=sorted(ca_signatures),
            exact_candidate_union=True,
            worker_calls_added=0,
            query_budget_unchanged=True,
            **compact
        )

        rows = []
        for agent_id in range(self.config.num_agents):
            rows.append(
                {
                    "macro_round": int(macro_round),
                    "source_step": int(step),
                    "target_step": int(target_step),
                    "agent_id": int(agent_id),
                    "stage": "late_paired_semantic_counterfactual",
                    "elite_candidate_id": private_elites[agent_id].candidate_id,
                    "joint_candidate_id": joint_records[agent_id].candidate_id,
                    "c0_candidate_id": (
                        c0_records[agent_id].candidate_id
                        if c0_records is not None
                        else None
                    ),
                    "ca_candidate_id": (
                        ca_records[agent_id].candidate_id
                        if ca_records is not None
                        else None
                    ),
                    "selected_anchor_candidate_id": selected_records[
                        agent_id
                    ].candidate_id,
                    "selected_kind": str(selected_kind),
                    "semantic_strict_win": bool(semantic_strict_win),
                    "semantic_accepted": bool(semantic_accepted),
                    "joint_adopted": bool(
                        selected_kind == "exact_joint_J"
                        and joint_records[agent_id].candidate_id
                        != private_elites[agent_id].candidate_id
                    ),
                }
            )
        return dict(selected_records), rows

    def _strict_summary_fields(self) -> Dict[str, Any]:
        fields = super()._strict_summary_fields()
        gap_mean = (
            sum(self._v51_ca_minus_c0) / float(len(self._v51_ca_minus_c0))
            if self._v51_ca_minus_c0
            else 0.0
        )
        strict_rate = (
            float(self._v51_semantic_strict_wins)
            / float(self._v51_treated_evaluations)
            if self._v51_treated_evaluations
            else 0.0
        )
        accept_rate = (
            float(self._v51_semantic_accepts)
            / float(self._v51_treated_evaluations)
            if self._v51_treated_evaluations
            else 0.0
        )
        fields.update(
            {
                "schema_version": "SNSD-v1.5-semantic-safeguard-run-v5.1",
                "paired_semantic_counterfactual": True,
                "same_state_c0_ca": True,
                "factor_A_definition": (
                    "treated semantic BoundaryPacket price-credit response CA "
                    "versus same-state zero-semantic C0"
                ),
                "factor_B_definition": (
                    "B1 injects C0 and accepted strict-win CA; B0 computes both "
                    "counterfactuals but injects neither"
                ),
                "v51_exact_candidate_union": "argmax(current,J,R0,Rm,C0,eligible_CA)",
                "v51_semantic_strict_win_rule": "CA > max(J,R0,Rm,C0)",
                "v51_ca_domain_rule": "accepted strict-win CA only",
                "v51_unaccepted_ca_domain_pollution_prohibited": True,
                "v51_late_decisions": int(self._v51_late_decisions),
                "v51_c0_evaluations": int(self._v51_c0_evaluations),
                "v51_ca_evaluations": int(self._v51_ca_evaluations),
                "v51_ca_deduplicated_evaluations": int(
                    self._v51_ca_deduplicated_evaluations
                ),
                "v51_treated_evaluations": int(self._v51_treated_evaluations),
                "v51_semantic_strict_wins": int(
                    self._v51_semantic_strict_wins
                ),
                "v51_semantic_strict_win_rate": float(strict_rate),
                "v51_semantic_accepts": int(self._v51_semantic_accepts),
                "v51_semantic_accept_rate": float(accept_rate),
                "v51_c0_selections": int(self._v51_c0_selections),
                "v51_ca_minus_c0_mean": float(gap_mean),
                "v51_ca_minus_c0_positive_steps": int(
                    self._v51_ca_minus_c0_positive_steps
                ),
                "v51_semantic_gain_over_baseline": float(
                    self._v51_semantic_gain_over_baseline
                ),
                "v51_endpoint_difference_steps": int(
                    self._v51_endpoint_difference_steps
                ),
                "v51_endpoint_symmetric_difference": int(
                    self._v51_endpoint_symmetric_difference
                ),
                "v51_refill_difference_steps": int(
                    self._v51_refill_difference_steps
                ),
                "v51_refill_symmetric_difference": int(
                    self._v51_refill_symmetric_difference
                ),
                "v51_distinct_microresponses_c0": int(
                    self._v51_distinct_microresponses_c0
                ),
                "v51_distinct_microresponses_ca": int(
                    self._v51_distinct_microresponses_ca
                ),
                "v51_cross_branch_distinct_steps": int(
                    self._v51_cross_branch_distinct_steps
                ),
                "v51_cross_branch_signature_difference": int(
                    self._v51_cross_branch_signature_difference
                ),
                "v51_payload_bytes_c0": int(self._v51_payload_bytes_c0),
                "v51_payload_bytes_ca": int(self._v51_payload_bytes_ca),
                "v51_counterfactual_feedback_scalars": int(
                    self._v51_counterfactual_feedback_scalars
                ),
                "v51_semantic_credit_nonzero": int(
                    self._v51_semantic_credit_nonzero
                ),
                "v51_c0_domain_injections": int(
                    self._v51_c0_domain_injections
                ),
                "v51_ca_domain_injections": int(
                    self._v51_ca_domain_injections
                ),
                "v51_unaccepted_ca_domain_injections": int(
                    self._v51_unaccepted_ca_domain_injections
                ),
                "v51_additional_worker_calls": 0,
                "v51_paid_worker_query_budget": 288,
                "v51_trace": list(self._v51_trace),
            }
        )
        return fields

    def run(self) -> ExperimentResult:
        result = super().run()
        receipt_path = Path(result.receipt_path)
        receipt = {
            "schema_version": "SNSD-v1.5-semantic-safeguard-receipt-v5.1",
            "status": "complete",
            "method": self.config.reported_method,
            "seed": self.config.seed,
            "artifacts": {
                "events.jsonl": {
                    "sha256": _sha256_file(Path(result.events_path)),
                    "bytes": Path(result.events_path).stat().st_size,
                },
                "summary.json": {
                    "sha256": _sha256_file(Path(result.summary_path)),
                    "bytes": Path(result.summary_path).stat().st_size,
                },
            },
        }
        _atomic_json(receipt_path, receipt)
        return result


def run_semantic_safeguard_experiment(
    data_path: str,
    output_dir: str,
    method: str = "Full",
    seed: int = 11,
    oracle: bool = False,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    negotiation_depth: int = 12,
    macro_rounds: int = 4,
    packet_top_k: int = 4,
    microcycles: int = 3,
    max_packet_top_k: int = 6,
    max_microcycles: int = 4,
) -> ExperimentResult:
    config = _structured_joint_config(
        data_path=data_path,
        output_dir=output_dir,
        method=method,
        seed=seed,
        oracle=oracle,
        checkpoint=checkpoint,
        device=device,
        negotiation_depth=negotiation_depth,
        macro_rounds=macro_rounds,
        packet_top_k=packet_top_k,
        microcycles=microcycles,
        max_packet_top_k=max_packet_top_k,
        max_microcycles=max_microcycles,
    )
    return SemanticSafeguardProtocol(config).run()


def preflight_v51_full_c3(
    data_path: str,
    method: str = "Full",
    seed: int = 11,
    packet_top_k: int = 4,
    microcycles: int = 3,
) -> Dict[str, Any]:
    result = preflight_v50_full_c3(
        data_path=data_path,
        method=method,
        seed=seed,
        packet_top_k=packet_top_k,
        microcycles=microcycles,
    )
    result.update(
        {
            "schema_version": "SNSD-v1.5-v5.1-full-c3-preflight-v1",
            "same_state_paired_c0_ca": True,
            "semantic_strict_win_rule": "CA > max(J,R0,Rm,C0)",
            "unaccepted_ca_domain_pollution_prohibited": True,
            "expected_online_calls": 288,
            "non_smoke": True,
        }
    )
    return result


__all__ = [
    "FORMAL_METHODS",
    "SemanticSafeguardProtocol",
    "run_semantic_safeguard_experiment",
    "preflight_v51_full_c3",
]
