"""Regressions for genuine six-transition and end-to-end strict training."""

from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from snsd_v51.training_cli import build_parser
from snsd_v51.learned_policy_protocol import LearnedPolicyProtocol
from snsd_v51.model import TriGraphConfig, TriGraphModel
from snsd_v51.messages import apply_message_intervention
from snsd_v51.runtime import TriGraphPolicy
from snsd_v51.training.strict_teacher import (
    StrictCrossfitRollingEnvironment,
    build_strict_crossfit_rolling_trajectories,
)
from snsd_v51.training.offline_dataset import (
    CandidateTargets,
    OfflineDataError,
    OfflineDataset,
    OfflineExample,
    PrequeryCandidate,
)
from snsd_v51.training.gnn_trainer import (
    StrictRollingMessageTrainerConfig,
    _run_strict_rollouts,
    audit_checkpoint_depth,
    _exact_group_loss,
    prepare_strict_rolling_state,
    strict_online_boundary_states,
)
from snsd_v51.training.warmstart_trainer import _groups


def _hash(text):
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _targets(objective):
    return CandidateTargets(
        solver_status="feasible",
        local_objective=float(objective),
        local_gain=0.0,
        local_gain_mask=True,
        forced_best_completion=float(objective),
        best_completion_advantage=0.0,
        advantage_mask=True,
        externality=0.0,
        externality_mask=True,
        left_opportunity_cost=0.0,
        right_opportunity_cost=0.0,
        opportunity_cost=0.0,
        completion_feasible=True,
        completion_mask=True,
        uncertainty_mask=True,
    )


def _strict_tiny_offline(perturb_hash=None, perturb=0.0, eligible=24):
    node_count = 12
    partition_id = np.repeat(np.arange(6, dtype=np.int64), 2)
    nodes = tuple(
        np.asarray((2 * agent, 2 * agent + 1), dtype=np.int64)
        for agent in range(6)
    )
    # One internal edge per agent.  Empty cut links keep compatibility simple
    # while still exercising full/local/boundary/message encoders.
    edges = np.asarray([(2 * agent, 2 * agent + 1) for agent in range(6)], dtype=np.int64)
    degree = np.ones(node_count, dtype=np.int64)
    graph = SimpleNamespace(
        num_nodes=node_count,
        num_edges=len(edges),
        edges=edges,
        edge_types=np.ones(len(edges), dtype=np.int64),
        degree=degree,
        conflict_duration=np.zeros(node_count, dtype=np.float32),
        indptr=np.arange(node_count + 1, dtype=np.int64),
        graph_hash="strict-tiny-graph",
    )
    partitions = SimpleNamespace(
        k=6,
        cut_edges=np.empty((0, 2), dtype=np.int64),
        partition_id=partition_id,
        nodes=nodes,
        horizon_start=0,
        horizon_end=100,
    )
    arcs = tuple(
        SimpleNamespace(
            weight=float(node + 1),
            priority=float(node % 3),
            link_st=float(node),
            link_et=float(node + 1),
            trace_st=float(node),
            trace_et=float(node + 1),
            ground=int(node % 2),
            satellite=int(node % 3),
        )
        for node in range(node_count)
    )
    dataset = SimpleNamespace(
        arcs=arcs,
        source_path="strict-tiny.csv",
        num_grounds=2,
        num_satellites=3,
    )
    generators = tuple(
        SimpleNamespace(agent_id=agent, boundary=(2 * agent, 2 * agent + 1))
        for agent in range(6)
    )

    examples = []
    eligible_ids = {}
    for agent_id in range(6):
        boundary = (2 * agent_id, 2 * agent_id + 1)
        safe_hash = _hash("strict-safe-%d" % agent_id)
        incumbent_hash = _hash("strict-incumbent-%d" % agent_id)
        candidates = [
            (safe_hash, (), 0.0, True),
            (incumbent_hash, (boundary[0],), 10.0 + agent_id, False),
        ]
        ids = []
        for index in range(int(eligible)):
            signature_hash = _hash("strict-target-%d-%d" % (agent_id, index))
            ids.append(signature_hash)
            selected = (boundary[1],) if index % 2 == 0 else ()
            candidates.append(
                (signature_hash, selected, 100.0 + 10.0 * index + agent_id, False)
            )
        eligible_ids[agent_id] = tuple(ids)
        for signature_hash, selected, objective, is_safe in candidates:
            if signature_hash == perturb_hash:
                objective += float(perturb)
            examples.append(
                OfflineExample(
                    prequery=PrequeryCandidate(
                        run_id="strict-tiny-101",
                        seed=101,
                        agent_id=agent_id,
                        round_id=-1,
                        signature_hash=signature_hash,
                        boundary_order_hash="strict-order-%d" % agent_id,
                        boundary_order=boundary,
                        selected_boundary=tuple(selected),
                        anchor_signature_hash=incumbent_hash,
                        anchor_selected_boundary=(boundary[0],),
                        is_safe_anchor=is_safe,
                    ),
                    targets=_targets(objective),
                )
            )
    return (
        OfflineDataset(
            examples=tuple(examples),
            arc_pressures=tuple(),
            receipt={"dataset_hash": "strict-tiny"},
            dataset=dataset,
            graph=graph,
            partitions=partitions,
            generators_by_seed={101: generators},
        ),
        eligible_ids,
    )


class OfflineTrainingV45StrictTests(unittest.TestCase):
    def test_cli_defaults_to_twelve_and_accepts_deeper_unroll(self):
        self.assertEqual(
            12, build_parser().parse_args(["--output", "unused"]).negotiation_steps
        )
        self.assertEqual(
            18,
            build_parser()
            .parse_args(["--output", "unused", "--negotiation-steps", "18"])
            .negotiation_steps,
        )

    def test_checkpoint_depth_mismatch_is_explicit(self):
        checkpoint = {"trained_negotiation_steps": 12}
        self.assertTrue(audit_checkpoint_depth(checkpoint, 12)["depth_match"])
        with self.assertRaises(OfflineDataError):
            audit_checkpoint_depth(checkpoint, 6)
        audit = audit_checkpoint_depth(checkpoint, 18, allow_mismatch=True)
        self.assertTrue(audit["depth_transfer_ablation"])

    def test_pool_exhaustion_is_a_hard_failure(self):
        offline, _ = _strict_tiny_offline(eligible=20)
        # The legacy full-table incumbent is now fold-eligible as well, so this
        # makes 21 candidates and no balanced two-fold target can reach twelve.
        with self.assertRaises(OfflineDataError):
            StrictCrossfitRollingEnvironment(
                offline, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
            )

    def test_twelve_steps_are_nonempty_full_agent_transitions(self):
        offline, _ = _strict_tiny_offline()
        observations = build_strict_crossfit_rolling_trajectories(
            offline, folds=2, negotiation_steps=12
        )
        self.assertEqual(24, len(observations))
        by_fold = {}
        for observation in observations:
            by_fold.setdefault(observation.heldout_fold, []).append(observation)
            counts = observation.receipt["remaining_target_counts"]
            self.assertEqual(6, len(counts))
            self.assertTrue(all(int(value) >= 1 for value in counts.values()))
            self.assertTrue(observation.receipt["source_target_disjoint"])
        for rows in by_fold.values():
            self.assertEqual(list(range(12)), [row.step for row in rows])
            self.assertEqual(11, sum(row.consumes_message for row in rows))

    def test_better_revealed_joint_candidate_changes_anchor(self):
        probe, _ = _strict_tiny_offline()
        probe_environment = StrictCrossfitRollingEnvironment(
            probe, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
        )
        probe_observation = probe_environment.observe()
        probe_grouped = _groups(probe_observation.target_examples)
        promoted_hash = probe_grouped[(101, 0)][0].prequery.signature_hash

        # Make one genuinely held-out candidate dominate only after it is
        # revealed.  This would be impossible if the initial source anchor had
        # leaked the full-table optimum.
        offline, _ = _strict_tiny_offline(
            perturb_hash=promoted_hash, perturb=1000000.0
        )
        environment = StrictCrossfitRollingEnvironment(
            offline, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
        )
        observation = environment.observe()
        grouped = _groups(observation.target_examples)
        selected = {
            agent_id: (
                promoted_hash
                if agent_id == 0
                else grouped[(101, agent_id)][0].prequery.signature_hash
            )
            for agent_id in range(6)
        }
        self.assertNotIn(
            promoted_hash,
            [item.prequery.signature_hash for item in environment._source[0]],
        )
        self.assertNotIn(
            promoted_hash,
            observation.receipt["initial_source_only_anchor_signature_hashes"].values(),
        )
        initial_factor_value = float(
            observation.receipt["initial_source_only_factor_value"]
        )
        transition = environment.advance(selected)
        self.assertIn(0, transition["changed_agents"])
        self.assertEqual(promoted_hash, transition["anchor_after"]["0"])
        self.assertGreater(float(transition["factor_value"]), initial_factor_value)

    def test_real_and_zero_exclude_current_heldout_objective(self):
        original, _ = _strict_tiny_offline()
        probe = StrictCrossfitRollingEnvironment(
            original, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
        )
        first = probe.observe()
        grouped = _groups(first.target_examples)
        perturb_hash = grouped[(101, 0)][-1].prequery.signature_hash
        chosen = {
            agent_id: next(
                item.prequery.signature_hash
                for item in grouped[(101, agent_id)]
                if item.prequery.signature_hash != perturb_hash
            )
            for agent_id in range(6)
        }
        changed, _ = _strict_tiny_offline(
            perturb_hash=perturb_hash, perturb=1000000.0
        )
        hashes = []
        zero_hashes = []
        for offline in (original, changed):
            environment = StrictCrossfitRollingEnvironment(
                offline, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
            )
            environment.observe()
            environment.advance(chosen)
            second = environment.observe()
            hashes.append(second.messages[0].message_hash)
            zero_hashes.append(
                apply_message_intervention(
                    second.messages[0],
                    "zero",
                    experiment_seed=101,
                    agent_id=0,
                    macro_round=1,
                ).message_hash
            )
            self.assertTrue(second.receipt["source_target_disjoint"])
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(zero_hashes[0], zero_hashes[1])

    def test_exact_advantage_decomposition_identity(self):
        offline, _ = _strict_tiny_offline()
        observation = StrictCrossfitRollingEnvironment(
            offline, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
        ).observe()
        for item in observation.target_examples:
            self.assertAlmostEqual(
                item.targets.best_completion_advantage,
                item.targets.local_gain + item.targets.externality,
                places=8,
            )

    def test_boundary_state_matches_online_experiment_field_by_field(self):
        offline, _ = _strict_tiny_offline()
        environment = StrictCrossfitRollingEnvironment(
            offline, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
        )
        observation = environment.observe()
        strict = strict_online_boundary_states(offline, observation)
        experiment_self = SimpleNamespace(partitions=offline.partitions)
        for agent_id, generator in enumerate(offline.generators_by_seed[101]):
            incumbent = SimpleNamespace(
                signature=SimpleNamespace(
                    selected_ids=observation.anchor_selected_boundary[agent_id]
                )
            )
            online = LearnedPolicyProtocol._state(
                experiment_self,
                agent_id,
                generator,
                incumbent,
                offline.dataset.arcs,
                offline.graph,
                pressure=None,
            )
            for field in (
                "boundary_order",
                "incumbent_selected",
                "raw_benefit",
                "local_marginal",
                "conflict_score",
                "prior_opportunity_cost",
                "uncertainty",
                "prior_pressure",
            ):
                self.assertEqual(getattr(online, field), getattr(strict[agent_id], field))

    def test_end_to_end_prepare_matches_inference_and_reaches_all_encoders(self):
        offline, _ = _strict_tiny_offline()
        environment = StrictCrossfitRollingEnvironment(
            offline, seed=101, heldout_fold=0, folds=2, negotiation_steps=12
        )
        first = environment.observe()
        grouped = _groups(first.target_examples)
        environment.advance(
            {
                agent_id: grouped[(101, agent_id)][0].prequery.signature_hash
                for agent_id in range(6)
            }
        )
        observation = environment.observe()
        graph_layers = TriGraphModel(TriGraphConfig())
        for parameter in graph_layers.parameters():
            parameter.requires_grad = True
        policy = TriGraphPolicy(graph_layers=graph_layers, device="cpu", seed=0)
        detached = prepare_strict_rolling_state(
            offline, policy, observation, "Real", differentiable=False
        )
        differentiable = prepare_strict_rolling_state(
            offline, policy, observation, "Real", differentiable=True
        )
        self.assertTrue(
            torch.allclose(
                detached.boundary_state.node_embeddings,
                differentiable.boundary_state.node_embeddings.detach(),
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        rows = _groups(observation.target_examples)[(101, 0)]
        loss, _ = _exact_group_loss(
            graph_layers,
            differentiable,
            observation,
            rows,
            StrictRollingMessageTrainerConfig(),
            "Real",
        )
        loss.backward()
        required = (
            "full_encoder.",
            "local_encoder.",
            "boundary_encoder.",
            "message_encoder.",
            "prequery_trunk.",
        )
        for prefix in required:
            magnitude = sum(
                float(parameter.grad.detach().abs().sum().item())
                for name, parameter in graph_layers.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            )
            self.assertGreater(magnitude, 0.0, prefix)

    def test_real_and_zero_each_execute_twelve_on_policy_transitions(self):
        offline, _ = _strict_tiny_offline()
        graph_layers = TriGraphModel(TriGraphConfig())
        policy = TriGraphPolicy(graph_layers=graph_layers, device="cpu", seed=0)
        metrics, audit = _run_strict_rollouts(
            offline,
            graph_layers,
            policy,
            seeds=(101,),
            config=StrictRollingMessageTrainerConfig(
                negotiation_steps=12, exploration_epsilon=0.0
            ),
            epoch=0,
        )
        self.assertEqual(48, audit["trajectory_observations"])
        self.assertTrue(audit["all_steps_full_agent_activity"])
        self.assertFalse(audit["fabricated_zero_labels"])
        for variant in ("Real", "Zero"):
            for step in range(12):
                row = audit["per_step"]["%s/step_%d" % (variant, step)]
                self.assertEqual(2, row["observations"])
                self.assertTrue(row["full_agent_activity"])
                self.assertGreaterEqual(row["minimum_candidates_per_agent"], 1)
        self.assertGreater(metrics["groups"], 0.0)


if __name__ == "__main__":
    unittest.main()
