"""Tests for leakage-safe v1.5 pre-query intent scoring."""

from __future__ import annotations

import inspect
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from test_graph_layers import _config, _encode_problem
    from snsd_v51.model import FeatureMode, TriGraphModel


def _score(graph_layers, local_state, boundary_state, mode=FeatureMode.FULL, **overrides):
    arguments = {
        "local_state": local_state,
        "boundary_state": boundary_state,
        "boundary_indices": torch.tensor([0, 1], dtype=torch.long),
        "requested_bits": torch.tensor(
            [[1, 0], [0, 1], [1, 1]], dtype=torch.bool
        ),
        "incumbent_bits": torch.tensor([1, 0], dtype=torch.bool),
        "raw_arc_benefit": torch.tensor([10.0, 20.0]),
        "benefit_scale": 10.0,
        "structural_features": torch.tensor([[0.0], [0.5], [1.0]]),
        "benefit_features": torch.tensor([[0.2], [0.7], [0.4]]),
        "conflict_features": torch.tensor([[0.8], [0.1], [0.6]]),
        "mode": mode,
    }
    arguments.update(overrides)
    return graph_layers.score_intents_prequery(**arguments)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PrequeryScoringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27182)
        self.graph_layers = TriGraphModel(_config()).cpu()
        self.graph_layers.eval()

    def test_prequery_api_has_no_worker_gain_and_keeps_proxy_separate(self):
        signature = inspect.signature(self.graph_layers.score_intents_prequery)
        self.assertNotIn("exact_local_benefit", signature.parameters)
        _, local_zero, _, boundary_state = _encode_problem(self.graph_layers)
        raw_weights = torch.tensor([10.0, 20.0], requires_grad=True)
        scores = _score(
            self.graph_layers,
            local_zero,
            boundary_state,
            raw_arc_benefit=raw_weights,
        )
        # Relative to incumbent [1, 0]: keep -> 0, replace 10 by 20 -> 10,
        # and keep 10 plus add 20 -> 20.
        torch.testing.assert_close(
            scores.raw_arc_benefit_proxy, torch.tensor([0.0, 10.0, 20.0])
        )
        torch.testing.assert_close(
            scores.normalized_arc_benefit_proxy, torch.tensor([0.0, 1.0, 2.0])
        )
        torch.testing.assert_close(
            scores.predicted_local_benefit,
            scores.normalized_arc_benefit_proxy
            + scores.learned_local_benefit_residual,
        )
        torch.testing.assert_close(
            scores.predicted_best_completion_advantage,
            scores.predicted_local_benefit + scores.learned_externality,
        )
        self.assertTrue(torch.isfinite(scores.selection_logit).all().item())
        self.assertTrue((scores.uncertainty > 0.0).all().item())
        scores.selection_logit.sum().backward()
        # Public weights are an exact input proxy, not a trainable/worker label.
        self.assertIsNone(raw_weights.grad)

    def test_prequery_ranking_receives_real_boundary_message_gradient(self):
        message_conflict = torch.tensor(
            [[0.2, 0.1], [0.4, 0.7], [0.8, 0.3], [0.5, 0.9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        message_benefit = torch.tensor(
            [[0.5, 0.2], [0.1, 0.8], [0.6, 0.4], [0.9, 0.3]],
            dtype=torch.float32,
            requires_grad=True,
        )
        _, local_zero, _, boundary_state = _encode_problem(
            self.graph_layers,
            boundary_message_conflict=message_conflict,
            boundary_message_benefit=message_benefit,
        )
        scores = _score(self.graph_layers, local_zero, boundary_state)
        loss = (
            scores.selection_logit.square().sum()
            + scores.completion_logit.square().sum()
        )
        loss.backward()
        self.assertGreater(float(message_conflict.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(message_benefit.grad.abs().sum().item()), 0.0)

    def test_prequery_ablation_mode_must_match_encoded_states(self):
        _, local_zero, _, boundary_state = _encode_problem(self.graph_layers)
        with self.assertRaises(ValueError):
            _score(
                self.graph_layers,
                local_zero,
                boundary_state,
                mode=FeatureMode.ZERO,
            )

        _, conflict_local, _, conflict_boundary = _encode_problem(
            self.graph_layers, mode=FeatureMode.CONFLICT_ONLY
        )
        conflict_a = _score(
            self.graph_layers,
            conflict_local,
            conflict_boundary,
            mode=FeatureMode.CONFLICT_ONLY,
        )
        conflict_b = _score(
            self.graph_layers,
            conflict_local,
            conflict_boundary,
            mode=FeatureMode.CONFLICT_ONLY,
            benefit_features=torch.tensor([[100.0], [200.0], [300.0]]),
        )
        torch.testing.assert_close(
            conflict_a.selection_logit, conflict_b.selection_logit
        )

        _, benefit_local, _, benefit_boundary = _encode_problem(
            self.graph_layers, mode=FeatureMode.BENEFIT_ONLY
        )
        benefit_a = _score(
            self.graph_layers,
            benefit_local,
            benefit_boundary,
            mode=FeatureMode.BENEFIT_ONLY,
        )
        benefit_b = _score(
            self.graph_layers,
            benefit_local,
            benefit_boundary,
            mode=FeatureMode.BENEFIT_ONLY,
            conflict_features=torch.tensor([[100.0], [200.0], [300.0]]),
        )
        torch.testing.assert_close(
            benefit_a.selection_logit, benefit_b.selection_logit
        )


if __name__ == "__main__":
    unittest.main()
