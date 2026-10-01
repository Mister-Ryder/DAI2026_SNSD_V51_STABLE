"""End-to-end tensor-adapter tests for the leakage-safe TriGraph runtime."""

from __future__ import annotations

import math
import unittest

import torch

from snsd_core.candidates import CandidateGenerator
from snsd_core.data import Arc, ArcDataset
from snsd_core.graph import build_conflict_graph, partition_equal_time
from snsd_v51.model import FeatureMode, TriGraphConfig, TriGraphModel
from snsd_v51.messages import BoundaryMessage, BoundaryState, ProposalIntent
from snsd_v51.runtime import TriGraphPolicy


def _problem():
    arcs = (
        Arc(0, 0, 0, "g0", "s0", 10, 20, 5, 25),
        Arc(1, 1, 1, "g1", "s1", 40, 49, 35, 54),
        Arc(2, 1, 2, "g1", "s2", 50, 61, 45, 66),
        Arc(3, 2, 0, "g2", "s0", 80, 92, 75, 97),
    )
    dataset = ArcDataset(
        arcs=arcs,
        ground_names=("g0", "g1", "g2"),
        satellite_names=("s0", "s1", "s2"),
        source_path="tiny.csv",
        encoding="utf-8",
        header=tuple(),
    )
    graph = build_conflict_graph(arcs)
    partitions = partition_equal_time(arcs, graph, k=2, horizon_start=0, horizon_end=100)
    generators = [
        CandidateGenerator(i, arcs, graph, partitions, top_h=8, seed=17)
        for i in range(2)
    ]
    states = {}
    messages = {}
    for agent_id, generator in enumerate(generators):
        order = generator.boundary
        selected = order[:1]
        states[agent_id] = BoundaryState.from_maps(
            agent_id=agent_id,
            boundary_order=order,
            incumbent_selected=selected,
            raw_benefit={node: arcs[node].weight for node in order},
            # This prohibited field is deliberately conspicuous.  The runtime
            # must not consume it for pre-query acquisition.
            local_marginal={node: 999999.0 + node for node in order},
            conflict_score={node: float(graph.degree[node]) for node in order},
            uncertainty={node: 0.1 * (node + 1) for node in order},
        )
        messages[agent_id] = BoundaryMessage.from_maps(
            order,
            opportunity_cost={node: 2.0 * (node + 1) for node in order},
            pressure={node: (-1.0) ** node * (node + 1) for node in order},
            neighbor_recovery={node: 0.5 * (node + 1) for node in order},
            conflict_score={node: float(graph.degree[node]) for node in order},
            uncertainty={node: 0.2 for node in order},
        )
    return dataset, graph, partitions, generators, states, messages


def _intent(agent_id, state, slot, selected):
    order = tuple(state.boundary_order)
    anchor = set(state.incumbent_selected)
    selected_set = set(selected)
    return ProposalIntent(
        agent_id=agent_id,
        macro_round=0,
        proposal_wave="A",
        intent_type="test",
        slot=slot,
        boundary_order=order,
        anchor_selected_boundary=tuple(node for node in order if node in anchor),
        hotspot_ids=order,
        requested_fix_on=tuple(node for node in order if node in selected_set),
        requested_fix_off=tuple(node for node in order if node not in selected_set),
        requested_selected_boundary=tuple(node for node in order if node in selected_set),
        added_ids=tuple(node for node in order if node in selected_set - anchor),
        removed_ids=tuple(node for node in order if node in anchor - selected_set),
        source_message_hash="prequery",
        message_intervention="real",
        top_pressure_ids=tuple(),
        top_pressure_values=tuple(),
        boundary_order_hash=state.boundary_order_hash,
        state_hash=state.state_hash,
        hotspot_hash="hotspot",
        random_seed=slot,
        intent_hash="intent-%d-%d" % (agent_id, slot),
    )


def _config():
    return TriGraphConfig(
        num_agents=2,
        hidden_dim=8,
        full_layers=1,
        local_layers=1,
        boundary_layers=1,
        dropout=0.0,
        candidate_struct_dim=2,
        candidate_benefit_dim=2,
        candidate_conflict_dim=2,
    )


class _ObjectiveBomb(object):
    def __init__(self, signature):
        self.signature = signature

    @property
    def objective(self):  # pragma: no cover - accessing it is the failure
        raise AssertionError("worker objective leaked into pre-query runtime")


class RuntimeV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.problem = _problem()
        self.policy = TriGraphPolicy(
            graph_layers=TriGraphModel(_config()), device="cpu", seed=123
        )

    def test_prepare_and_rank_every_registered_ablation(self):
        dataset, graph, partitions, generators, states, messages = self.problem
        for mode in FeatureMode:
            prepared = self.policy.prepare(
                dataset,
                graph,
                partitions,
                generators,
                states,
                messages,
                mode=mode,
            )
            for agent_id in range(2):
                order = states[agent_id].boundary_order
                intents = [
                    _intent(agent_id, states[agent_id], 0, tuple()),
                    _intent(agent_id, states[agent_id], 1, order),
                ]
                ranked = self.policy.rank_intents(prepared, agent_id, intents)
                self.assertEqual(2, len(ranked))
                self.assertEqual([1, 2], [row.rank for row in ranked])
                self.assertTrue(all(math.isfinite(row.selection_logit) for row in ranked))
                self.assertGreaterEqual(ranked[0].selection_logit, ranked[1].selection_logit)
                self.assertEqual({item.intent_hash for item in intents}, {row.intent.intent_hash for row in ranked})

    def test_full_encoding_is_cached_and_metadata_is_auditable(self):
        dataset, graph, partitions, generators, states, messages = self.problem
        first = self.policy.prepare(
            dataset, graph, partitions, generators, states, messages, mode="Full"
        )
        second = self.policy.prepare(
            dataset, graph, partitions, generators, states, messages, mode="Full"
        )
        self.assertIs(first.full_state, second.full_state)
        self.assertEqual(first.cache_key, second.cache_key)
        metadata = self.policy.metadata.to_dict()
        self.assertEqual(64, len(metadata["model_hash"]))
        self.assertGreater(metadata["parameter_count"], 0)
        self.assertFalse(metadata["checkpoint_loaded"])

    def test_objective_and_exact_local_marginal_are_not_read(self):
        dataset, graph, partitions, generators, states, messages = self.problem
        incumbent = {
            agent_id: _ObjectiveBomb(generator.make_signature(states[agent_id].incumbent_selected))
            for agent_id, generator in enumerate(generators)
        }
        prepared_a = self.policy.prepare(
            dataset,
            graph,
            partitions,
            generators,
            states,
            messages,
            incumbent=incumbent,
            mode="Full",
        )
        changed_states = {
            agent_id: BoundaryState.from_maps(
                agent_id=agent_id,
                boundary_order=state.boundary_order,
                incumbent_selected=state.incumbent_selected,
                raw_benefit=state.value_map("raw_benefit"),
                local_marginal={node: -1.0e20 for node in state.boundary_order},
                conflict_score=state.value_map("conflict_score"),
                uncertainty=state.value_map("uncertainty"),
            )
            for agent_id, state in states.items()
        }
        prepared_b = self.policy.prepare(
            dataset,
            graph,
            partitions,
            generators,
            changed_states,
            messages,
            incumbent=incumbent,
            mode="Full",
        )
        agent_id = 0
        intents_a = [
            _intent(agent_id, states[agent_id], 0, tuple()),
            _intent(agent_id, states[agent_id], 1, states[agent_id].boundary_order),
        ]
        intents_b = [
            _intent(agent_id, changed_states[agent_id], 0, tuple()),
            _intent(
                agent_id,
                changed_states[agent_id],
                1,
                changed_states[agent_id].boundary_order,
            ),
        ]
        scores_a = {
            row.intent.slot: row.selection_logit
            for row in self.policy.rank_intents(prepared_a, agent_id, intents_a)
        }
        scores_b = {
            row.intent.slot: row.selection_logit
            for row in self.policy.rank_intents(prepared_b, agent_id, intents_b)
        }
        for slot in scores_a:
            self.assertAlmostEqual(scores_a[slot], scores_b[slot], places=7)

    def test_rejects_cross_agent_or_misaligned_intents(self):
        dataset, graph, partitions, generators, states, messages = self.problem
        prepared = self.policy.prepare(
            dataset, graph, partitions, generators, states, messages
        )
        wrong = _intent(1, states[1], 0, tuple())
        with self.assertRaises(ValueError):
            self.policy.rank_intents(prepared, 0, [wrong])


if __name__ == "__main__":
    unittest.main()
