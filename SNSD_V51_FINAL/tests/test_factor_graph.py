"""Tests for the exact candidate-level chain factor graph."""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from snsd_v51.factor_graph import (
    CandidateAction,
    ChainFactorGraph,
    InvalidSafeAnchorError,
    build_compatibility_matrix,
    solve_chain_max_sum,
)


def action(
    agent_id,
    candidate_id,
    value,
    selected=(),
    incumbent=False,
    safe=False,
    feasible=True,
):
    return CandidateAction(
        agent_id=agent_id,
        candidate_id=candidate_id,
        local_objective=value,
        selected_boundary=frozenset(selected),
        is_incumbent=incumbent,
        is_safe=safe,
        is_feasible=feasible,
    )


def brute_force(tables, matrices, forced=None):
    best_value = float("-inf")
    best_tuple = None
    for indices in itertools.product(*[range(len(table)) for table in tables]):
        if forced is not None and indices[forced[0]] != forced[1]:
            continue
        if any(not tables[i][indices[i]].is_feasible for i in range(len(tables))):
            continue
        if any(
            not matrices[i][indices[i], indices[i + 1]]
            for i in range(len(matrices))
        ):
            continue
        value = sum(tables[i][indices[i]].local_objective for i in range(len(tables)))
        if value > best_value:
            best_value = value
            best_tuple = indices
    return best_value, best_tuple


class CandidateContractTests(unittest.TestCase):
    def test_candidate_normalizes_boundary_and_rejects_nonfinite_value(self):
        candidate = action(0, "c", 7, selected=[3, 2, 3])
        self.assertEqual(candidate.selected_boundary, frozenset([2, 3]))
        self.assertEqual(candidate.local_objective, 7.0)
        with self.assertRaises(ValueError):
            action(0, "bad", float("nan"))

    def test_exact_adjacent_compatibility_matrix(self):
        left = [
            action(0, "l0", 0, incumbent=True, safe=True),
            action(0, "l1", 5, selected=[10]),
            action(0, "l_bad", 100, selected=[11], feasible=False),
        ]
        right = [
            action(1, "r0", 0, incumbent=True, safe=True),
            action(1, "r1", 6, selected=[20]),
            action(1, "r2", 7, selected=[21]),
        ]
        matrix = build_compatibility_matrix(left, right, [(10, 20), (11, 21)])
        expected = np.asarray(
            [
                [True, True, True],
                [True, False, True],
                [False, False, False],
            ],
            dtype=np.bool_,
        )
        np.testing.assert_array_equal(matrix, expected)
        self.assertFalse(matrix.flags.writeable)


class ExactChainTests(unittest.TestCase):
    def test_backpointer_recovers_known_optimum(self):
        tables = [
            [
                action(0, "a0", 0, incumbent=True, safe=True),
                action(0, "a1", 9, selected=[1]),
            ],
            [
                action(1, "b0", 0, incumbent=True, safe=True),
                action(1, "b1", 8, selected=[2, 3]),
                action(1, "b2", 4, selected=[4]),
            ],
            [
                action(2, "c0", 0, incumbent=True, safe=True),
                action(2, "c1", 10, selected=[5]),
            ],
        ]
        edges = [[(1, 2)], [(3, 5)]]
        result = solve_chain_max_sum(tables, edges)
        # a1 + b2 + c1 is compatible and dominates all alternatives.
        self.assertEqual(result.best_indices, (1, 2, 1))
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in result.best_actions),
            ("a1", "b2", "c1"),
        )
        self.assertEqual(result.best_value, 23.0)

    def test_random_chains_match_brute_force_for_optimum_and_forced_values(self):
        rng = np.random.RandomState(24680)
        for trial in range(40):
            n_agents = int(rng.randint(2, 7))
            tables = []
            for agent_id in range(n_agents):
                table = [
                    action(
                        agent_id,
                        "a%d_inc" % agent_id,
                        int(rng.randint(0, 8)),
                        incumbent=True,
                        safe=True,
                    )
                ]
                for candidate_index in range(1, int(rng.randint(2, 5))):
                    selected = [
                        agent_id * 10 + bit
                        for bit in range(3)
                        if rng.rand() < 0.45
                    ]
                    table.append(
                        action(
                            agent_id,
                            "a%d_%d" % (agent_id, candidate_index),
                            int(rng.randint(-4, 20)),
                            selected=selected,
                            feasible=bool(rng.rand() > 0.12),
                        )
                    )
                tables.append(table)

            cut_edges = []
            for agent_id in range(n_agents - 1):
                edges = []
                for left_bit in range(3):
                    for right_bit in range(3):
                        if rng.rand() < 0.30:
                            edges.append(
                                (
                                    agent_id * 10 + left_bit,
                                    (agent_id + 1) * 10 + right_bit,
                                )
                            )
                cut_edges.append(edges)

            result = solve_chain_max_sum(tables, cut_edges)
            brute_value, _ = brute_force(
                tables, result.compatibility_matrices
            )
            self.assertAlmostEqual(result.best_value, brute_value, places=10)
            chosen_value = sum(item.local_objective for item in result.best_actions)
            self.assertAlmostEqual(chosen_value, result.best_value, places=10)

            for agent_position, table in enumerate(tables):
                for candidate_index in range(len(table)):
                    forced_value, _ = brute_force(
                        tables,
                        result.compatibility_matrices,
                        forced=(agent_position, candidate_index),
                    )
                    actual = result.forced_best_completions[agent_position][
                        candidate_index
                    ]
                    if math.isinf(forced_value):
                        self.assertTrue(np.isneginf(actual))
                    else:
                        self.assertAlmostEqual(actual, forced_value, places=10)

    def test_locally_infeasible_candidate_is_audited_but_never_selected(self):
        tables = [
            [action(0, "a0", 1, incumbent=True, safe=True)],
            [
                action(1, "b0", 2, incumbent=True, safe=True),
                action(1, "impossible", 1000, feasible=False),
            ],
        ]
        result = solve_chain_max_sum(tables, [[]])
        self.assertEqual(result.best_indices, (0, 0))
        self.assertEqual(result.best_value, 3.0)
        self.assertTrue(np.isneginf(result.forced_best_completions[1][1]))
        self.assertTrue(np.isneginf(result.best_completion_advantages[1][1]))
        self.assertFalse(result.completion_feasible[1][1])
        self.assertTrue(np.isinf(result.opportunity_costs[1][1]))

    def test_candidate_without_any_completion_gets_negative_infinity(self):
        # Safe-anchor validation is intentionally disabled in this diagnostic
        # graph so a forced, locally feasible candidate can have no completion.
        tables = [
            [action(0, "a_inc", 3, selected=[10], incumbent=True)],
            [
                action(1, "b_inc", 2, incumbent=True),
                action(1, "blocked", 50, selected=[20]),
            ],
        ]
        graph = ChainFactorGraph(
            tables,
            [[(10, 20)]],
            require_safe_anchor=False,
        )
        result = graph.solve()
        self.assertEqual(result.best_indices, (0, 0))
        self.assertTrue(np.isneginf(result.forced_best_completions[1][1]))
        self.assertFalse(result.completion_feasible[1][1])


class SafeAnchorAndOpportunityCostTests(unittest.TestCase):
    def test_safe_anchor_must_complete_every_feasible_neighbor_action(self):
        tables = [
            [
                action(0, "a_inc", 1, incumbent=True),
                action(0, "a_safe_bad", 0, selected=[10], safe=True),
            ],
            [
                action(1, "b_inc", 1, incumbent=True),
                action(1, "b_safe", 0, safe=True),
                action(1, "b_other", 2, selected=[20]),
            ],
        ]
        with self.assertRaises(InvalidSafeAnchorError):
            ChainFactorGraph(tables, [[(10, 20)]])

    def test_opportunity_cost_identity_and_externality(self):
        tables = [
            [
                action(0, "a0", 4, incumbent=True, safe=True),
                action(0, "a1", 12, selected=[10]),
            ],
            [
                action(1, "b0", 5, incumbent=True, safe=True),
                action(1, "b1", 11, selected=[20]),
                action(1, "b2", 8, selected=[21]),
            ],
            [
                action(2, "c0", 3, incumbent=True, safe=True),
                action(2, "c1", 13, selected=[30]),
            ],
        ]
        result = solve_chain_max_sum(
            tables,
            [[(10, 20)], [(21, 30)]],
        )

        for i, table in enumerate(tables):
            incumbent_index = result.incumbent_indices[i]
            incumbent_cost = result.opportunity_costs[i][incumbent_index]
            for j, candidate in enumerate(table):
                if not result.completion_feasible[i][j]:
                    continue
                reconstructed = (
                    candidate.local_objective
                    + result.free_left_branch_values[i]
                    + result.free_right_branch_values[i]
                    - result.left_opportunity_costs[i][j]
                    - result.right_opportunity_costs[i][j]
                )
                self.assertAlmostEqual(
                    reconstructed,
                    result.forced_best_completions[i][j],
                    places=10,
                )
                expected_externality = -(
                    result.opportunity_costs[i][j] - incumbent_cost
                )
                self.assertAlmostEqual(
                    result.externalities[i][j], expected_externality, places=10
                )
                self.assertAlmostEqual(
                    result.best_completion_advantages[i][j],
                    result.local_gains[i][j] + result.externalities[i][j],
                    places=10,
                )


if __name__ == "__main__":
    unittest.main()
