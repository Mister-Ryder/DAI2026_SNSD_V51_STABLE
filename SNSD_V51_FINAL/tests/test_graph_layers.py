"""CPU tests for the dependency-light TriGraph v1.5 graph_layers."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - the formal environment requires torch
    torch = None

if torch is not None:
    from snsd_v51.graph_layers import (
        FeatureMode,
        RelationMeanConv,
        TriGraphConfig,
        TriGraphModel,
    )


def _config():
    return TriGraphConfig(
        full_struct_dim=3,
        full_benefit_dim=2,
        full_conflict_dim=2,
        local_struct_dim=2,
        local_benefit_dim=1,
        local_conflict_dim=2,
        boundary_struct_dim=1,
        boundary_benefit_dim=1,
        boundary_conflict_dim=1,
        message_benefit_dim=2,
        message_conflict_dim=2,
        candidate_struct_dim=1,
        candidate_benefit_dim=1,
        candidate_conflict_dim=1,
        full_edge_dim=1,
        local_edge_dim=1,
        boundary_edge_dim=1,
        hidden_dim=8,
        num_agents=2,
        full_relations=2,
        local_relations=2,
        boundary_relations=2,
        full_layers=2,
        local_layers=1,
        boundary_layers=1,
        dropout=0.0,
        candidate_pool_chunk=2,
    )


def _full_inputs(requires_grad=False):
    structural = torch.linspace(-1.0, 1.0, 18).reshape(6, 3)
    benefit = torch.tensor(
        [[0.1, 0.4], [0.2, 0.5], [0.3, 0.7], [0.8, 0.1], [0.9, 0.2], [1.0, 0.3]],
        dtype=torch.float32,
    )
    conflict = torch.tensor(
        [[0.0, 0.2], [0.2, 0.0], [0.3, 0.4], [0.4, 0.3], [0.0, 0.7], [0.8, 0.1]],
        dtype=torch.float32,
    )
    if requires_grad:
        structural.requires_grad_()
        benefit.requires_grad_()
        conflict.requires_grad_()
    # One canonical edge per undirected conflict; no doubled edge tensor.
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], dtype=torch.long
    )
    edge_type = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    edge_features = torch.linspace(0.1, 0.6, 6).reshape(6, 1)
    return structural, benefit, conflict, edge_index, edge_type, edge_features


def _local_state(graph_layers, full_state, agent_id, mode):
    if agent_id == 0:
        ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        core = torch.tensor([True, True, True, False])
        halo = torch.tensor([False, False, False, True])
        boundary = torch.tensor([False, True, True, False])
    else:
        ids = torch.tensor([2, 3, 4, 5], dtype=torch.long)
        core = torch.tensor([False, True, True, True])
        halo = torch.tensor([True, False, False, False])
        boundary = torch.tensor([False, True, True, False])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    edge_features = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float32)
    structural = torch.tensor(
        [[0.0, 1.0], [0.2, 0.8], [0.7, 0.3], [1.0, 0.0]],
        dtype=torch.float32,
    )
    benefit = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.float32)
    conflict = torch.tensor(
        [[0.1, 0.0], [0.2, 0.3], [0.4, 0.5], [0.7, 0.2]],
        dtype=torch.float32,
    )
    message_conflict = torch.zeros(4, 2)
    message_benefit = torch.zeros(4, 2)
    return graph_layers.encode_local(
        agent_id=agent_id,
        full_state=full_state,
        global_node_ids=ids,
        structural_features=structural,
        benefit_features=benefit,
        conflict_features=conflict,
        edge_index=edge_index,
        core_mask=core,
        halo_mask=halo,
        boundary_mask=boundary,
        edge_type=edge_type,
        edge_features=edge_features,
        message_conflict=message_conflict,
        message_benefit=message_benefit,
        mode=mode,
    )


def _encode_problem(
    graph_layers,
    mode=FeatureMode.FULL,
    boundary_message_conflict=None,
    boundary_message_benefit=None,
):
    full_inputs = _full_inputs()
    full_state = graph_layers.encode_full(*full_inputs, mode=mode)
    local_zero = _local_state(graph_layers, full_state, 0, mode)
    local_one = _local_state(graph_layers, full_state, 1, mode)

    # Boundary order: global nodes 1, 2 owned by agent 0 and nodes 3, 4
    # owned by agent 1.  The local embeddings are stitched without allowing
    # either agent to control its neighbour's halo.
    local_boundary_embeddings = torch.stack(
        (
            local_zero.node_embeddings[1],
            local_zero.node_embeddings[2],
            local_one.node_embeddings[1],
            local_one.node_embeddings[2],
        ),
        dim=0,
    )
    agent_contexts = torch.stack(
        (local_zero.agent_context, local_one.agent_context), dim=0
    )
    if boundary_message_conflict is None:
        boundary_message_conflict = torch.tensor(
            [[0.2, 0.1], [0.4, 0.7], [0.8, 0.3], [0.5, 0.9]],
            dtype=torch.float32,
        )
    if boundary_message_benefit is None:
        boundary_message_benefit = torch.tensor(
            [[0.5, 0.2], [0.1, 0.8], [0.6, 0.4], [0.9, 0.3]],
            dtype=torch.float32,
        )
    boundary_state = graph_layers.encode_boundary(
        full_state=full_state,
        global_node_ids=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        agent_ids=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        local_node_embeddings=local_boundary_embeddings,
        agent_contexts=agent_contexts,
        structural_features=torch.tensor([[0.0], [0.3], [0.7], [1.0]]),
        benefit_features=torch.tensor([[0.2], [0.5], [0.6], [0.9]]),
        conflict_features=torch.tensor([[0.8], [0.6], [0.4], [0.2]]),
        cut_edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        cut_edge_type=torch.tensor([0, 1], dtype=torch.long),
        cut_edge_features=torch.tensor([[0.25], [0.75]], dtype=torch.float32),
        message_conflict=boundary_message_conflict,
        message_benefit=boundary_message_benefit,
        mode=mode,
    )
    return full_state, local_zero, local_one, boundary_state


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RelationMeanConvTests(unittest.TestCase):
    def test_canonical_edge_propagates_in_both_directions(self):
        layer = RelationMeanConv(1, 1, num_relations=1, edge_feature_dim=0)
        with torch.no_grad():
            layer.self_linear.weight.zero_()
            layer.self_linear.bias.zero_()
            layer.relation_linears[0].weight.fill_(1.0)
        values = torch.tensor([[1.0], [3.0]])
        canonical_edge = torch.tensor([[0], [1]], dtype=torch.long)
        result = layer(values, canonical_edge, undirected=True)
        torch.testing.assert_close(result, torch.tensor([[3.0], [1.0]]))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TriGraphEncodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.graph_layers = TriGraphModel(_config()).cpu()
        self.graph_layers.eval()

    def test_small_cpu_graph_shapes_and_exact_benefit_identity(self):
        full_state, local_zero, _, boundary_state = _encode_problem(self.graph_layers)
        self.assertEqual(tuple(full_state.node_embeddings.shape), (6, 8))
        self.assertEqual(tuple(full_state.graph_embedding.shape), (16,))
        self.assertEqual(tuple(local_zero.node_embeddings.shape), (4, 8))
        self.assertEqual(tuple(local_zero.core_pool.shape), (16,))
        self.assertEqual(tuple(local_zero.halo_pool.shape), (16,))
        self.assertEqual(tuple(local_zero.boundary_pool.shape), (16,))
        self.assertEqual(tuple(local_zero.agent_context.shape), (8,))
        self.assertEqual(tuple(boundary_state.node_embeddings.shape), (4, 8))
        self.assertEqual(tuple(boundary_state.agent_embeddings.shape), (2, 8))

        arc_scores = self.graph_layers.score_arcs(boundary_state)
        self.assertEqual(tuple(arc_scores.add_logit.shape), (4,))
        self.assertEqual(tuple(arc_scores.delete_logit.shape), (4,))

        exact = torch.tensor([0.0, 1.25, -0.75], requires_grad=True)
        candidate_scores = self.graph_layers.score_candidates(
            local_state=local_zero,
            boundary_state=boundary_state,
            boundary_indices=torch.tensor([0, 1], dtype=torch.long),
            candidate_bits=torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.bool),
            exact_local_benefit=exact,
            structural_features=torch.tensor([[0.0], [0.5], [1.0]]),
            benefit_features=torch.tensor([[0.2], [0.7], [0.4]]),
            conflict_features=torch.tensor([[0.8], [0.1], [0.6]]),
        )
        torch.testing.assert_close(
            candidate_scores.exact_local_benefit,
            torch.tensor([0.0, 1.25, -0.75]),
        )
        torch.testing.assert_close(
            candidate_scores.predicted_best_completion_advantage,
            candidate_scores.exact_local_benefit
            + candidate_scores.learned_externality,
        )
        self.assertTrue(torch.isfinite(candidate_scores.selection_logit).all().item())
        self.assertTrue((candidate_scores.uncertainty > 0.0).all().item())
        self.assertTrue(
            (
                (candidate_scores.completion_probability >= 0.0)
                & (candidate_scores.completion_probability <= 1.0)
            ).all().item()
        )

        candidate_scores.selection_logit.sum().backward()
        # Worker-produced local benefit is a known value, never a learned head.
        self.assertIsNone(exact.grad)
        self.assertGreater(self.graph_layers.parameter_count(), 0)

    def test_feature_modes_mask_only_the_declared_information(self):
        conflict = torch.tensor(
            [[0.1, 0.3], [0.2, 0.4], [0.5, 0.7], [0.6, 0.9]],
            dtype=torch.float32,
        )
        benefit = torch.tensor(
            [[0.9, 0.7], [0.8, 0.6], [0.4, 0.2], [0.3, 0.1]],
            dtype=torch.float32,
        )
        full = self.graph_layers.encode_messages(conflict, benefit, FeatureMode.FULL)
        zero = self.graph_layers.encode_messages(conflict, benefit, FeatureMode.ZERO)
        self.assertFalse(torch.allclose(full, zero))
        torch.testing.assert_close(zero, torch.zeros_like(zero))

        conflict_only_a = self.graph_layers.encode_messages(
            conflict, benefit, FeatureMode.CONFLICT_ONLY
        )
        conflict_only_b = self.graph_layers.encode_messages(
            conflict, benefit + 100.0, FeatureMode.CONFLICT_ONLY
        )
        torch.testing.assert_close(conflict_only_a, conflict_only_b)
        benefit_only_a = self.graph_layers.encode_messages(
            conflict, benefit, FeatureMode.BENEFIT_ONLY
        )
        benefit_only_b = self.graph_layers.encode_messages(
            conflict + 100.0, benefit, FeatureMode.BENEFIT_ONLY
        )
        torch.testing.assert_close(benefit_only_a, benefit_only_b)

        permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        shuffled = self.graph_layers.encode_messages(
            conflict,
            benefit,
            FeatureMode.SHUFFLED,
            shuffle_index=permutation,
        )
        expected = self.graph_layers.encode_messages(
            conflict.index_select(0, permutation),
            benefit.index_select(0, permutation),
            FeatureMode.FULL,
        )
        torch.testing.assert_close(shuffled, expected)

        structural, raw_benefit, raw_conflict, edges, types, edge_features = _full_inputs()
        no_global = self.graph_layers.encode_full(
            structural,
            raw_benefit,
            raw_conflict,
            edges,
            types,
            edge_features,
            mode=FeatureMode.NO_GLOBAL,
        )
        torch.testing.assert_close(
            no_global.node_embeddings, torch.zeros_like(no_global.node_embeddings)
        )
        torch.testing.assert_close(
            no_global.graph_embedding, torch.zeros_like(no_global.graph_embedding)
        )

        conflict_graph_a = self.graph_layers.encode_full(
            structural,
            raw_benefit,
            raw_conflict,
            edges,
            types,
            edge_features,
            mode=FeatureMode.CONFLICT_ONLY,
        )
        conflict_graph_b = self.graph_layers.encode_full(
            structural,
            raw_benefit + 10.0,
            raw_conflict,
            edges,
            types,
            edge_features,
            mode=FeatureMode.CONFLICT_ONLY,
        )
        torch.testing.assert_close(
            conflict_graph_a.node_embeddings, conflict_graph_b.node_embeddings
        )

        benefit_graph_a = self.graph_layers.encode_full(
            structural,
            raw_benefit,
            raw_conflict,
            edges,
            types,
            edge_features,
            mode=FeatureMode.BENEFIT_ONLY,
        )
        benefit_graph_b = self.graph_layers.encode_full(
            structural,
            raw_benefit,
            raw_conflict + 10.0,
            edges,
            types,
            edge_features,
            mode=FeatureMode.BENEFIT_ONLY,
        )
        torch.testing.assert_close(
            benefit_graph_a.node_embeddings, benefit_graph_b.node_embeddings
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MessageGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31415)
        self.graph_layers = TriGraphModel(_config()).cpu()
        self.graph_layers.eval()

    def _message_gradient(self, mode):
        conflict = torch.tensor(
            [[0.2, 0.1], [0.4, 0.7], [0.8, 0.3], [0.5, 0.9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        benefit = torch.tensor(
            [[0.5, 0.2], [0.1, 0.8], [0.6, 0.4], [0.9, 0.3]],
            dtype=torch.float32,
            requires_grad=True,
        )
        encoded = self.graph_layers.encode_messages(conflict, benefit, mode=mode)
        (encoded.square().sum() + encoded.sum()).backward()
        return conflict.grad, benefit.grad

    def test_message_gradients_respect_full_zero_and_channel_masks(self):
        full_conflict, full_benefit = self._message_gradient(FeatureMode.FULL)
        self.assertGreater(float(full_conflict.abs().sum().item()), 0.0)
        self.assertGreater(float(full_benefit.abs().sum().item()), 0.0)

        zero_conflict, zero_benefit = self._message_gradient(FeatureMode.ZERO)
        torch.testing.assert_close(zero_conflict, torch.zeros_like(zero_conflict))
        torch.testing.assert_close(zero_benefit, torch.zeros_like(zero_benefit))

        conflict_only_conflict, conflict_only_benefit = self._message_gradient(
            FeatureMode.CONFLICT_ONLY
        )
        self.assertGreater(float(conflict_only_conflict.abs().sum().item()), 0.0)
        torch.testing.assert_close(
            conflict_only_benefit, torch.zeros_like(conflict_only_benefit)
        )

        benefit_only_conflict, benefit_only_benefit = self._message_gradient(
            FeatureMode.BENEFIT_ONLY
        )
        torch.testing.assert_close(
            benefit_only_conflict, torch.zeros_like(benefit_only_conflict)
        )
        self.assertGreater(float(benefit_only_benefit.abs().sum().item()), 0.0)

    def test_candidate_heads_receive_boundary_message_gradient(self):
        conflict = torch.tensor(
            [[0.2, 0.1], [0.4, 0.7], [0.8, 0.3], [0.5, 0.9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        benefit = torch.tensor(
            [[0.5, 0.2], [0.1, 0.8], [0.6, 0.4], [0.9, 0.3]],
            dtype=torch.float32,
            requires_grad=True,
        )
        _, local_zero, _, boundary_state = _encode_problem(
            self.graph_layers,
            boundary_message_conflict=conflict,
            boundary_message_benefit=benefit,
        )
        scores = self.graph_layers.score_candidates(
            local_state=local_zero,
            boundary_state=boundary_state,
            boundary_indices=torch.tensor([0, 1], dtype=torch.long),
            candidate_bits=torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.bool),
            exact_local_benefit=torch.tensor([0.0, 0.5, -0.2]),
            structural_features=torch.tensor([[0.0], [0.5], [1.0]]),
            benefit_features=torch.tensor([[0.2], [0.7], [0.4]]),
            conflict_features=torch.tensor([[0.8], [0.1], [0.6]]),
        )
        arc_scores = self.graph_layers.score_arcs(boundary_state)
        loss = (
            scores.learned_externality.square().sum()
            + scores.completion_logit.square().sum()
            + scores.selection_logit.square().sum()
            + arc_scores.add_logit.square().sum()
            + arc_scores.delete_logit.square().sum()
        )
        loss.backward()
        self.assertGreater(float(conflict.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(benefit.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
