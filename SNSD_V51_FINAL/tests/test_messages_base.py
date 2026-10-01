from __future__ import absolute_import

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = ROOT / "snsd_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from snsd_core.candidates import CandidateGenerator
from snsd_core.solver import ConditionedLocalSolver
from snsd_v51.messages_base import (
    BoundaryMessage,
    BoundaryState,
    MessageIntervention,
    ProposalEngine,
    apply_message_intervention,
    derived_seed,
    materialize_intent,
    measure_compliance,
    solver_compliance,
    stable_hash,
)


def _large_state():
    order = tuple(range(80))
    selected = tuple(range(12))
    return BoundaryState.from_maps(
        agent_id=2,
        boundary_order=order,
        incumbent_selected=selected,
        raw_benefit={node: float(100 + node) for node in order},
        local_marginal={node: float((node * 7) % 23) for node in order},
        conflict_score={node: float((79 - node) % 17) for node in order},
        prior_opportunity_cost={node: float((node * 3) % 13) for node in order},
        uncertainty={node: float((node * 11) % 29) / 29.0 for node in order},
        prior_pressure={node: float((node % 9) - 4) for node in order},
    )


def _messages(order):
    first = BoundaryMessage.from_maps(
        order,
        opportunity_cost={0: 500.0, 70: 80.0},
        pressure={0: -300.0, 70: 300.0},
        neighbor_recovery={0: 400.0},
        conflict_score={0: 200.0},
        uncertainty={60: 20.0},
    )
    second = BoundaryMessage.from_maps(
        order,
        opportunity_cost={11: 500.0, 79: 80.0},
        pressure={11: -300.0, 79: 300.0},
        neighbor_recovery={11: 400.0},
        conflict_score={11: 200.0},
        uncertainty={61: 20.0},
    )
    return first, second


class ProposalDeterminismTests(unittest.TestCase):
    def test_stable_hash_and_seed_ignore_mapping_insertion_order(self):
        first = {"a": 1, "b": [2.0, 3]}
        second = {"b": [2.0, 3], "a": 1}
        self.assertEqual(stable_hash(first), stable_hash(second))
        self.assertEqual(derived_seed(first), derived_seed(second))

    def test_dynamic_hotspot_is_deterministic_and_has_h64(self):
        state = _large_state()
        engine = ProposalEngine(agent_id=2, experiment_seed=17)
        first = engine.dynamic_hotspot(state, macro_round=0)
        second = engine.dynamic_hotspot(state, macro_round=0)
        self.assertEqual(first, second)
        self.assertEqual(64, first.effective_size)
        self.assertTrue(set(state.incumbent_selected).issubset(first.node_ids))
        self.assertEqual(64, len(set(first.node_ids)))

    def test_wave_a_is_reproducible_unique_and_keeps_non_hotspot_bits(self):
        state = _large_state()
        engine = ProposalEngine(agent_id=2, experiment_seed=19)
        first = engine.wave_a(state, macro_round=1)
        second = engine.wave_a(state, macro_round=1)
        self.assertEqual(
            [intent.intent_hash for intent in first],
            [intent.intent_hash for intent in second],
        )
        self.assertEqual(24, len(first))
        self.assertEqual(24, len({intent.requested_selected_boundary for intent in first}))
        boundary = set(state.boundary_order)
        anchor = set(state.incumbent_selected)
        for intent in first:
            self.assertEqual(boundary, set(intent.requested_fix_on).union(intent.requested_fix_off))
            self.assertFalse(set(intent.requested_fix_on).intersection(intent.requested_fix_off))
            self.assertEqual(intent.requested_fix_on, intent.requested_selected_boundary)
            for node_id in boundary.difference(intent.hotspot_ids):
                self.assertEqual(
                    node_id in anchor,
                    node_id in set(intent.requested_selected_boundary),
                )


class MessageInterventionTests(unittest.TestCase):
    def test_zero_masks_hidden_real_message_but_real_changes_intents(self):
        state = _large_state()
        message_a, message_b = _messages(state.boundary_order)
        engine = ProposalEngine(agent_id=2, experiment_seed=23)

        zero_a = engine.wave_b(
            state, message_a, macro_round=0, intervention=MessageIntervention.ZERO
        )
        zero_b = engine.wave_b(
            state, message_b, macro_round=0, intervention=MessageIntervention.ZERO
        )
        self.assertEqual(
            [intent.intent_hash for intent in zero_a],
            [intent.intent_hash for intent in zero_b],
        )
        self.assertEqual(
            [intent.requested_selected_boundary for intent in zero_a],
            [intent.requested_selected_boundary for intent in zero_b],
        )

        real_a = engine.wave_b(
            state, message_a, macro_round=0, intervention=MessageIntervention.REAL
        )
        real_b = engine.wave_b(
            state, message_b, macro_round=0, intervention=MessageIntervention.REAL
        )
        self.assertNotEqual(
            [intent.requested_selected_boundary for intent in real_a],
            [intent.requested_selected_boundary for intent in real_b],
        )
        self.assertIn(0, real_a[0].removed_ids)
        self.assertIn(11, real_b[0].removed_ids)
        self.assertEqual(
            {"yield", "retreat", "replace", "explore"},
            {intent.intent_type for intent in real_a},
        )
        for intent_type in ("yield", "retreat", "replace", "explore"):
            self.assertEqual(
                6, sum(intent.intent_type == intent_type for intent in real_a)
            )

    def test_shuffled_message_is_deterministic_and_preserves_channels(self):
        state = _large_state()
        message, _ = _messages(state.boundary_order)
        first = apply_message_intervention(message, "shuffled", 31, 2, 1)
        second = apply_message_intervention(message, "shuffled", 31, 2, 1)
        self.assertEqual(first, second)
        self.assertNotEqual(tuple(range(len(state.boundary_order))), first.permutation)
        for field_name in (
            "opportunity_cost",
            "pressure",
            "neighbor_recovery",
            "conflict_score",
            "uncertainty",
        ):
            self.assertEqual(
                sorted(getattr(message, field_name)),
                sorted(getattr(first, field_name)),
            )


class _Arc(object):
    def __init__(self, weight):
        self.weight = float(weight)


class _Graph(object):
    def __init__(self, size):
        self.indptr = np.zeros(size + 1, dtype=np.int64)
        self.indices = np.asarray([], dtype=np.int64)

    def neighbors(self, node_id):
        start = int(self.indptr[int(node_id)])
        end = int(self.indptr[int(node_id) + 1])
        return self.indices[start:end]


class _Partitions(object):
    def __init__(self, size):
        boundary = tuple(range(size))
        self.k = 1
        self.nodes = [boundary]
        self.boundary_left = [boundary]
        self.boundary_right = [tuple()]
        self.boundary = [boundary]
        self.cut_edges = []


class V1InteroperabilityTests(unittest.TestCase):
    def test_intent_materializes_and_runs_in_v1_worker_with_compliance(self):
        size = 8
        arcs = tuple(_Arc(10 + node_id) for node_id in range(size))
        graph = _Graph(size)
        partitions = _Partitions(size)
        generator = CandidateGenerator(
            0, arcs, graph, partitions, top_h=2, seed=7
        )
        anchor = generator.initial_anchor()
        state = BoundaryState.from_maps(
            agent_id=0,
            boundary_order=generator.boundary,
            incumbent_selected=anchor.selected_ids,
            raw_benefit={node: arcs[node].weight for node in generator.boundary},
            conflict_score={node: float(node) for node in generator.boundary},
            uncertainty={node: float(size - node) for node in generator.boundary},
        )
        intents = ProposalEngine(0, experiment_seed=37).wave_a(
            state, macro_round=0, count=4
        )
        self.assertTrue(intents)
        materialized = materialize_intent(intents[0], generator, anchor)
        self.assertTrue(materialized.projection_compliance.full_compliance)

        solver = ConditionedLocalSolver(
            arcs, graph, partitions, moves=5, restarts=2, seed=41
        )
        result = solver.solve(0, materialized.signature)
        self.assertEqual("feasible", result.status)
        compliance = solver_compliance(materialized, result)
        self.assertTrue(compliance.full_compliance)
        self.assertEqual(
            set(materialized.signature.selected_ids),
            set(compliance.realized_selected_boundary),
        )

        requested = intents[0]
        dropped = tuple(requested.requested_fix_on[1:])
        violated = measure_compliance(requested, dropped, solver_status="synthetic")
        self.assertFalse(violated.full_compliance)
        self.assertTrue(violated.violated_fix_on)


if __name__ == "__main__":
    unittest.main()
