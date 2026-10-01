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
from snsd_v51.messages import (
    BoundaryMessage,
    BoundaryState,
    MessageIntervention,
    ProposalEngine,
    apply_message_intervention,
    materialize_intent,
)


def _state():
    order = tuple(range(80))
    return BoundaryState.from_maps(
        agent_id=2,
        boundary_order=order,
        incumbent_selected=(node for node in range(12)),
        raw_benefit={node: float(100 + node) for node in order},
        local_marginal={node: float((node * 7) % 23) for node in order},
        conflict_score={node: float((79 - node) % 17) for node in order},
        prior_opportunity_cost={node: float((node * 3) % 13) for node in order},
        uncertainty={node: float((node * 11) % 29) / 29.0 for node in order},
        prior_pressure={node: float((node % 9) - 4) for node in order},
    )


class CommonRandomNumberTests(unittest.TestCase):
    def test_numeric_zero_real_and_zero_share_requests_and_worker_seeds(self):
        state = _state()
        numeric_zero = BoundaryMessage.from_maps(state.boundary_order)
        engine = ProposalEngine(agent_id=2, experiment_seed=23)

        real = engine.wave_b(
            state, numeric_zero, macro_round=1,
            intervention=MessageIntervention.REAL,
        )
        zero = engine.wave_b(
            state, numeric_zero, macro_round=1,
            intervention=MessageIntervention.ZERO,
        )

        self.assertEqual(
            [intent.requested_selected_boundary for intent in real],
            [intent.requested_selected_boundary for intent in zero],
        )
        self.assertEqual(
            [intent.random_seed for intent in real],
            [intent.random_seed for intent in zero],
        )
        self.assertNotEqual(real[0].source_message_hash, zero[0].source_message_hash)
        self.assertNotEqual(real[0].intent_hash, zero[0].intent_hash)

    def test_numeric_messages_change_intents_but_not_slot_seeds(self):
        state = _state()
        first = BoundaryMessage.from_maps(
            state.boundary_order,
            opportunity_cost={0: 500.0, 70: 80.0},
            pressure={0: -300.0, 70: 300.0},
            neighbor_recovery={0: 400.0},
            conflict_score={0: 200.0},
            uncertainty={60: 20.0},
        )
        second = BoundaryMessage.from_maps(
            state.boundary_order,
            opportunity_cost={11: 500.0, 79: 80.0},
            pressure={11: -300.0, 79: 300.0},
            neighbor_recovery={11: 400.0},
            conflict_score={11: 200.0},
            uncertainty={61: 20.0},
        )
        engine = ProposalEngine(agent_id=2, experiment_seed=29)
        intents_a = engine.wave_b(state, first, macro_round=0)
        intents_b = engine.wave_b(state, second, macro_round=0)
        self.assertNotEqual(
            [intent.requested_selected_boundary for intent in intents_a],
            [intent.requested_selected_boundary for intent in intents_b],
        )
        self.assertEqual(
            [intent.random_seed for intent in intents_a],
            [intent.random_seed for intent in intents_b],
        )

    def test_shuffle_permutation_does_not_depend_on_message_hash(self):
        state = _state()
        first = BoundaryMessage.from_maps(
            state.boundary_order, pressure={0: 1.0, 1: 2.0}
        )
        second = BoundaryMessage.from_maps(
            state.boundary_order, pressure={5: 10.0, 7: -4.0}
        )
        shuffled_a = apply_message_intervention(first, "shuffled", 31, 2, 4)
        shuffled_b = apply_message_intervention(second, "shuffled", 31, 2, 4)
        self.assertEqual(shuffled_a.permutation, shuffled_b.permutation)
        self.assertNotEqual(tuple(range(80)), shuffled_a.permutation)


class _Arc(object):
    def __init__(self, weight):
        self.weight = float(weight)


class _Graph(object):
    def __init__(self):
        # One internal boundary conflict: 0 -- 1.
        self._neighbors = {
            0: np.asarray([1], dtype=np.int64),
            1: np.asarray([0], dtype=np.int64),
            2: np.asarray([], dtype=np.int64),
            3: np.asarray([], dtype=np.int64),
        }
        self.indptr = np.asarray([0, 1, 2, 2, 2], dtype=np.int64)
        self.indices = np.asarray([1, 0], dtype=np.int64)

    def neighbors(self, node_id):
        return self._neighbors[int(node_id)]


class _Partitions(object):
    def __init__(self):
        boundary = (0, 1, 2, 3)
        self.k = 1
        self.nodes = [boundary]
        self.boundary_left = [boundary]
        self.boundary_right = [tuple()]
        self.boundary = [boundary]
        self.cut_edges = []


class StrictMaterializationTests(unittest.TestCase):
    def _fixture(self):
        arcs = tuple(_Arc(weight) for weight in (20, 10, 8, 6))
        generator = CandidateGenerator(
            0, arcs, _Graph(), _Partitions(), top_h=4, seed=7
        )
        anchor = generator.empty_signature()
        state = BoundaryState.from_maps(
            agent_id=0,
            boundary_order=generator.boundary,
            incumbent_selected=(node for node in anchor.selected_ids),
            raw_benefit={node: arcs[node].weight for node in generator.boundary},
        )
        engine = ProposalEngine(0, experiment_seed=37, hotspot_size=4)
        hotspot = engine.dynamic_hotspot(state, macro_round=0)
        intent = engine._make_intent(
            state=state,
            hotspot=hotspot,
            macro_round=0,
            proposal_wave="A",
            intent_type="synthetic-conflict",
            slot=0,
            requested_selected=(0, 1),
            source_message_hash="test-provenance",
            message_intervention="none",
            top_pressure_ids=(),
            top_pressure_values=(),
        )
        return generator, anchor, state, intent

    def test_iterator_incumbent_is_not_consumed(self):
        state = BoundaryState.from_maps(
            agent_id=0,
            boundary_order=(0, 1, 2, 3),
            incumbent_selected=(node for node in (0, 2)),
        )
        self.assertEqual((0, 2), state.incumbent_selected)

    def test_strict_projection_noncompliance_is_ineligible(self):
        generator, anchor, _, intent = self._fixture()
        materialized = materialize_intent(intent, generator, anchor)
        self.assertTrue(materialized.projection_changed)
        self.assertFalse(materialized.projection_compliance.full_compliance)
        self.assertFalse(materialized.eligible_for_query)
        self.assertEqual(
            ("projection_noncompliance",), materialized.rejection_reasons
        )

        permissive = materialize_intent(
            intent, generator, anchor, strict_compliance=False
        )
        self.assertTrue(permissive.projection_changed)
        self.assertTrue(permissive.eligible_for_query)
        self.assertEqual((), permissive.rejection_reasons)

    def test_duplicate_is_diagnosed_without_mutating_seen_collection(self):
        generator, anchor, _, intent = self._fixture()
        first = materialize_intent(
            intent, generator, anchor, strict_compliance=False
        )
        seen = {first.signature.signature_hash}
        before = set(seen)
        duplicate = materialize_intent(
            intent,
            generator,
            anchor,
            strict_compliance=False,
            seen_signature_hashes=seen,
        )
        self.assertEqual(before, seen)
        self.assertTrue(duplicate.duplicate_signature)
        self.assertFalse(duplicate.eligible_for_query)
        self.assertEqual(("duplicate_signature",), duplicate.rejection_reasons)

    def test_generator_boundary_order_must_match_exactly(self):
        generator, anchor, _, intent = self._fixture()
        object.__setattr__(intent, "boundary_order", tuple(reversed(intent.boundary_order)))
        with self.assertRaisesRegex(ValueError, "boundary order"):
            materialize_intent(intent, generator, anchor)


if __name__ == "__main__":
    unittest.main()
