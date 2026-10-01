"""Signature-constrained local solver.

This is a fixed, non-learning MWIS heuristic: deterministic greedy construction
followed by bounded one-for-many exchanges.  Every boundary bit is enforced at
the central add/remove guards, so learning can only choose which oracle query to
buy; it cannot alter the local search policy.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .candidates import BoundarySignature


@dataclass(frozen=True)
class LocalSolveResult:
    agent_id: int
    signature_hash: str
    status: str
    selected_ids: Tuple[int, ...]
    objective: float
    runtime_s: float
    attempted_moves: int
    seed: int
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "signature_hash": self.signature_hash,
            "status": self.status,
            "selected_ids": list(self.selected_ids),
            "objective": self.objective,
            "runtime_s": self.runtime_s,
            "attempted_moves": self.attempted_moves,
            "seed": self.seed,
            "message": self.message,
        }


class ConditionedLocalSolver:
    """A reusable local oracle for all partitions and signatures."""

    def __init__(
        self,
        arcs: Sequence[object],
        graph: object,
        partitions: object,
        moves: int = 500,
        restarts: int = 1,
        seed: int = 0,
    ) -> None:
        self.arcs = arcs
        self.graph = graph
        self.partitions = partitions
        self.moves = max(0, int(moves))
        self.restarts = max(1, int(restarts))
        self.seed = int(seed)
        self.num_nodes = len(arcs)
        self.weights = np.asarray(
            [float(getattr(arc, "weight")) for arc in arcs], dtype=np.float64
        )
        self._partition_nodes: List[np.ndarray] = [
            np.asarray(nodes, dtype=np.int64) for nodes in partitions.nodes
        ]
        self._boundary: List[Set[int]] = []
        self._priority: List[List[int]] = []
        degree = np.diff(graph.indptr).astype(np.float64)
        for agent_id, nodes in enumerate(self._partition_nodes):
            boundary = set(int(x) for x in partitions.boundary_left[agent_id])
            boundary.update(int(x) for x in partitions.boundary_right[agent_id])
            self._boundary.append(boundary)
            interior = [int(x) for x in nodes if int(x) not in boundary]
            interior.sort(
                key=lambda node_id: (
                    -self.weights[node_id] / float(degree[node_id] + 1.0),
                    -self.weights[node_id],
                    node_id,
                )
            )
            self._priority.append(interior)

    def _derived_seed(self, signature: BoundarySignature, restart: int) -> int:
        digest = hashlib.sha256(
            ("%d|%s|%d" % (self.seed, signature.signature_hash, restart)).encode("ascii")
        ).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    def _neighbors(self, node_id: int) -> np.ndarray:
        start = int(self.graph.indptr[node_id])
        end = int(self.graph.indptr[node_id + 1])
        return self.graph.indices[start:end]

    def _check_signature(self, agent_id: int, signature: BoundarySignature) -> None:
        if signature.agent_id != agent_id:
            raise ValueError("signature belongs to a different agent")
        if set(signature.boundary_order) != self._boundary[agent_id]:
            raise ValueError("signature does not cover the complete local boundary")
        selected = set(signature.selected_ids)
        if not selected.issubset(self._boundary[agent_id]):
            raise ValueError("signature selects a non-boundary node")

    def _greedy_order(self, agent_id: int, rng: random.Random, restart: int) -> List[int]:
        base = self._priority[agent_id]
        if restart == 0 or len(base) < 2:
            return base
        # Jitter only inside small priority blocks; this creates reproducible
        # restarts without discarding the weight/conflict ordering.
        order: List[int] = []
        block_size = 32
        for begin in range(0, len(base), block_size):
            block = list(base[begin : begin + block_size])
            rng.shuffle(block)
            order.extend(block)
        return order

    def _run_once(
        self,
        agent_id: int,
        signature: BoundarySignature,
        restart: int,
    ) -> Tuple[Optional[np.ndarray], int, str]:
        partition_nodes = self._partition_nodes[agent_id]
        partition_mask = np.zeros(self.num_nodes, dtype=np.bool_)
        partition_mask[partition_nodes] = True
        boundary = self._boundary[agent_id]
        fixed_one = set(int(x) for x in signature.selected_ids)
        fixed_zero = boundary.difference(fixed_one)

        selected = np.zeros(self.num_nodes, dtype=np.bool_)
        conflict_count = np.zeros(self.num_nodes, dtype=np.int32)

        def add(node_id: int) -> None:
            if selected[node_id]:
                return
            selected[node_id] = True
            neighbors = self._neighbors(node_id)
            conflict_count[neighbors] += 1

        def remove(node_id: int) -> None:
            if not selected[node_id]:
                return
            selected[node_id] = False
            neighbors = self._neighbors(node_id)
            conflict_count[neighbors] -= 1

        # Fixed-one boundary bits are installed first.  Any internal conflict
        # means the signature itself is infeasible and no search is attempted.
        for node_id in sorted(fixed_one):
            if conflict_count[node_id] != 0:
                return None, 0, "fixed-one boundary vertices conflict"
            add(node_id)

        rng_seed = self._derived_seed(signature, restart)
        rng = random.Random(rng_seed)
        order = self._greedy_order(agent_id, rng, restart)
        priority_rank = {node_id: rank for rank, node_id in enumerate(order)}

        for node_id in order:
            if self.weights[node_id] > 0.0 and conflict_count[node_id] == 0:
                add(node_id)

        attempted = 0
        if order:
            scan_order = list(order)
            rng.shuffle(scan_order)
            cursor = 0
            for _ in range(self.moves):
                attempted += 1
                if cursor >= len(scan_order):
                    rng.shuffle(scan_order)
                    cursor = 0
                candidate = scan_order[cursor]
                cursor += 1
                if selected[candidate]:
                    continue

                neighbors = self._neighbors(candidate)
                conflicting = [int(x) for x in neighbors if selected[int(x)]]
                if any(node_id in fixed_one for node_id in conflicting):
                    continue
                removed_value = float(sum(self.weights[node_id] for node_id in conflicting))
                if self.weights[candidate] <= removed_value + 1e-9:
                    continue

                refill_pool: Set[int] = set()
                for node_id in conflicting:
                    refill_pool.update(
                        int(x)
                        for x in self._neighbors(node_id)
                        if partition_mask[int(x)]
                    )
                    remove(node_id)
                if conflict_count[candidate] != 0:
                    # Defensive rollback; this should only occur for malformed
                    # asymmetric adjacency arrays.
                    for node_id in conflicting:
                        if conflict_count[node_id] == 0:
                            add(node_id)
                    continue
                add(candidate)

                # Restore local maximality only in the region exposed by the
                # exchange.  Boundary-zero bits never enter this pool.
                refill = [
                    node_id
                    for node_id in refill_pool
                    if node_id not in boundary
                    and node_id != candidate
                    and not selected[node_id]
                    and self.weights[node_id] > 0.0
                ]
                refill.sort(key=lambda n: (priority_rank.get(n, len(order)), n))
                for node_id in refill:
                    if conflict_count[node_id] == 0:
                        add(node_id)

        selected_ids = np.flatnonzero(selected)
        if any(int(x) in fixed_zero for x in selected_ids):
            return None, attempted, "fixed-zero boundary vertex was selected"
        actual_boundary = set(int(x) for x in selected_ids if int(x) in boundary)
        if actual_boundary != fixed_one:
            return None, attempted, "returned boundary differs from requested signature"
        return selected_ids, attempted, ""

    def solve(
        self,
        agent_id: int,
        signature: BoundarySignature,
    ) -> LocalSolveResult:
        started = time.perf_counter()
        try:
            self._check_signature(agent_id, signature)
        except Exception as exc:
            return LocalSolveResult(
                agent_id=agent_id,
                signature_hash=signature.signature_hash,
                status="error",
                selected_ids=(),
                objective=0.0,
                runtime_s=time.perf_counter() - started,
                attempted_moves=0,
                seed=self.seed,
                message=str(exc),
            )

        best_ids: Optional[np.ndarray] = None
        best_value = float("-inf")
        total_moves = 0
        last_message = ""
        for restart in range(self.restarts):
            ids, attempted, message = self._run_once(agent_id, signature, restart)
            total_moves += attempted
            last_message = message
            if ids is None:
                continue
            value = float(self.weights[ids].sum())
            if value > best_value + 1e-9:
                best_ids = ids
                best_value = value

        runtime = time.perf_counter() - started
        if best_ids is None:
            return LocalSolveResult(
                agent_id=agent_id,
                signature_hash=signature.signature_hash,
                status="infeasible",
                selected_ids=(),
                objective=0.0,
                runtime_s=runtime,
                attempted_moves=total_moves,
                seed=self.seed,
                message=last_message,
            )

        return LocalSolveResult(
            agent_id=agent_id,
            signature_hash=signature.signature_hash,
            status="feasible",
            selected_ids=tuple(int(x) for x in best_ids),
            objective=best_value,
            runtime_s=runtime,
            attempted_moves=total_moves,
            seed=self.seed,
        )
