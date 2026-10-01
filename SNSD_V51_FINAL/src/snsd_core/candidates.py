"""Full binary separator signatures and deterministic proposal generation.

The important invariant in this module is that a signature fixes *every* local
boundary vertex.  ``selected_ids`` stores the sparse one bits; all other
vertices in ``boundary_order`` are therefore fixed to zero.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class BoundarySignature:
    agent_id: int
    left_selected: Tuple[int, ...]
    right_selected: Tuple[int, ...]
    boundary_order: Tuple[int, ...]
    boundary_order_hash: str
    signature_hash: str

    @property
    def selected_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(set(self.left_selected).union(self.right_selected)))

    def assignment(self) -> Dict[int, int]:
        selected = set(self.selected_ids)
        return {node_id: int(node_id in selected) for node_id in self.boundary_order}

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "left_selected": list(self.left_selected),
            "right_selected": list(self.right_selected),
            "boundary_order": list(self.boundary_order),
            "boundary_order_hash": self.boundary_order_hash,
            "signature_hash": self.signature_hash,
        }


@dataclass(frozen=True)
class Quartet:
    quartet_id: str
    q00: str
    q10: str
    q01: str
    q11: str


class CandidateGenerator:
    """Generate legal full boundary signatures around a verified anchor."""

    def __init__(
        self,
        agent_id: int,
        arcs: Sequence[object],
        graph: object,
        partitions: object,
        top_h: int = 8,
        seed: int = 0,
    ) -> None:
        self.agent_id = int(agent_id)
        self.arcs = arcs
        self.graph = graph
        self.partitions = partitions
        self.top_h = max(1, int(top_h))
        self.seed = int(seed)

        self.left = tuple(sorted(int(x) for x in partitions.boundary_left[self.agent_id]))
        self.right = tuple(sorted(int(x) for x in partitions.boundary_right[self.agent_id]))
        self.boundary = tuple(sorted(set(self.left).union(self.right)))
        self._boundary_set = set(self.boundary)
        self._left_set = set(self.left)
        self._right_set = set(self.right)
        raw_order = json.dumps(
            {"agent": self.agent_id, "boundary": self.boundary},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.boundary_order_hash = hashlib.sha256(raw_order).hexdigest()

        self._cut_degree = {node_id: 0 for node_id in self.boundary}
        for edge in partitions.cut_edges:
            u, v = int(edge[0]), int(edge[1])
            if u in self._cut_degree:
                self._cut_degree[u] += 1
            if v in self._cut_degree:
                self._cut_degree[v] += 1

        self._score = {node_id: self._influence(node_id) for node_id in self.boundary}
        self.ranked_left = tuple(sorted(self.left, key=lambda n: (-self._score[n], n)))
        self.ranked_right = tuple(sorted(self.right, key=lambda n: (-self._score[n], n)))
        self.mutable_left = self.ranked_left[: self.top_h]
        self.mutable_right = self.ranked_right[: self.top_h]
        self.mutable = tuple(sorted(set(self.mutable_left).union(self.mutable_right)))

    def influence(self, node_id: int) -> float:
        return float(self._score[int(node_id)])

    def mutable_for_anchor(
        self, anchor: BoundarySignature
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Freeze the round's Top-H rule, prioritising incumbent one bits."""

        selected = set(anchor.selected_ids)

        def choose(ranked: Sequence[int]) -> Tuple[int, ...]:
            ordered = [node_id for node_id in ranked if node_id in selected]
            ordered.extend(node_id for node_id in ranked if node_id not in selected)
            return tuple(ordered[: self.top_h])

        left = choose(self.ranked_left)
        right = choose(self.ranked_right)
        return left, right, tuple(sorted(set(left).union(right)))

    def _weight(self, node_id: int) -> float:
        return float(getattr(self.arcs[node_id], "weight"))

    def _degree(self, node_id: int) -> int:
        indptr = getattr(self.graph, "indptr")
        return int(indptr[node_id + 1] - indptr[node_id])

    def _influence(self, node_id: int) -> float:
        degree = self._degree(node_id)
        cut_degree = self._cut_degree.get(node_id, 0)
        weight = self._weight(node_id)
        return weight / float(degree + 1) + 0.05 * weight * float(cut_degree)

    def _neighbors(self, node_id: int) -> Iterable[int]:
        return self.graph.neighbors(node_id)

    def _project(
        self,
        requested_selected: Iterable[int],
        locked_selected: Iterable[int] = (),
    ) -> Tuple[int, ...]:
        requested = set(int(x) for x in requested_selected).intersection(self._boundary_set)
        locked = set(int(x) for x in locked_selected).intersection(requested)

        # Bits outside Top-H are inherited from the anchor and cannot be silently
        # removed by projection.  A legal anchor makes the locked prefix feasible.
        chosen: Set[int] = set()
        blocked: Set[int] = set()
        for node_id in sorted(locked, key=lambda n: (-self._score[n], n)):
            if node_id in blocked:
                raise ValueError("locked boundary bits are internally incompatible")
            chosen.add(node_id)
            blocked.update(int(x) for x in self._neighbors(node_id))

        remaining = requested.difference(locked)
        for node_id in sorted(remaining, key=lambda n: (-self._score[n], n)):
            if node_id not in blocked:
                chosen.add(node_id)
                blocked.update(int(x) for x in self._neighbors(node_id))
        return tuple(sorted(chosen))

    def make_signature(
        self,
        selected_ids: Iterable[int],
        anchor: Optional[BoundarySignature] = None,
        mutable_override: Optional[Iterable[int]] = None,
    ) -> BoundarySignature:
        locked: Set[int] = set()
        if anchor is not None:
            mutable = set(
                self.mutable if mutable_override is None else mutable_override
            )
            locked = set(anchor.selected_ids).difference(mutable)
        selected = self._project(selected_ids, locked_selected=locked)
        selected_set = set(selected)
        left_selected = tuple(node_id for node_id in self.left if node_id in selected_set)
        right_selected = tuple(node_id for node_id in self.right if node_id in selected_set)
        bits = "".join("1" if node_id in selected_set else "0" for node_id in self.boundary)
        payload = (
            "%d|%s|%s" % (self.agent_id, self.boundary_order_hash, bits)
        ).encode("ascii")
        signature_hash = hashlib.sha256(payload).hexdigest()
        return BoundarySignature(
            agent_id=self.agent_id,
            left_selected=left_selected,
            right_selected=right_selected,
            boundary_order=self.boundary,
            boundary_order_hash=self.boundary_order_hash,
            signature_hash=signature_hash,
        )

    def empty_signature(self) -> BoundarySignature:
        return self.make_signature(())

    def initial_anchor(self) -> BoundarySignature:
        # A deterministic boundary-only MWIS seed.  It is not claimed to be
        # globally optimal; the conditioned local solver fills the interior.
        selected: List[int] = []
        blocked: Set[int] = set()
        for node_id in sorted(self.boundary, key=lambda n: (-self._score[n], n)):
            if node_id not in blocked:
                selected.append(node_id)
                blocked.update(int(x) for x in self._neighbors(node_id))
        return self.make_signature(selected)

    def _toggle(
        self,
        anchor: BoundarySignature,
        toggled: Iterable[int],
        mutable_override: Optional[Iterable[int]] = None,
    ) -> BoundarySignature:
        selected = set(anchor.selected_ids)
        for node_id in toggled:
            if node_id in selected:
                selected.remove(node_id)
            else:
                selected.add(node_id)
        return self.make_signature(
            selected, anchor=anchor, mutable_override=mutable_override
        )

    def compatible_family(
        self,
        anchor: BoundarySignature,
        safe_superset: Iterable[int],
        count: int,
    ) -> List[BoundarySignature]:
        """Create distinct signatures contained in one global safe superset."""

        _, _, mutable = self.mutable_for_anchor(anchor)
        safe = set(int(x) for x in safe_superset)
        anchor_selected = set(anchor.selected_ids)
        flippable = [
            node_id
            for node_id in mutable
            if node_id in anchor_selected or node_id in safe
        ]
        result: Dict[str, BoundarySignature] = {}
        for width in range(1, min(3, len(flippable)) + 1):
            for mask in itertools.combinations(flippable, width):
                signature = self._toggle(
                    anchor, mask, mutable_override=mutable
                )
                if set(signature.selected_ids).issubset(safe):
                    result.setdefault(signature.signature_hash, signature)
                if len(result) >= int(count):
                    return list(result.values())
        return list(result.values())

    def warmup(self, num_quartets: int) -> Tuple[List[BoundarySignature], List[Quartet]]:
        """Return safe + counterfactual quartet conditions, deduplicated.

        Quartet hashes remain valid when an endpoint has only one separator side;
        such agents simply do not contribute a coupling label.
        """

        requested = max(0, int(num_quartets))
        safe = self.empty_signature()
        if requested == 0:
            return [safe], []

        base_bits_needed = int(math.ceil(math.log(float(requested), 2.0))) if requested > 1 else 0

        # Preferred construction: perturb a strong boundary-MWIS anchor only
        # by turning one bits off.  This preserves feasibility, exact quartet
        # uniqueness and the useful warm-up incumbent quality.
        strong_anchor = self.initial_anchor()
        anchor_selected = set(strong_anchor.selected_ids)
        dynamic_left, dynamic_right, dynamic_mutable = self.mutable_for_anchor(
            strong_anchor
        )
        selected_left = [node_id for node_id in dynamic_left if node_id in anchor_selected]
        selected_right = [node_id for node_id in dynamic_right if node_id in anchor_selected]
        if selected_left and selected_right:
            removal_pairs = [
                (left, right)
                for left in selected_left
                for right in selected_right
                if left != right
            ]
        else:
            selected_active = [
                node_id for node_id in dynamic_mutable if node_id in anchor_selected
            ]
            removal_pairs = list(itertools.combinations(selected_active, 2))

        for delta_left, delta_right in removal_pairs:
            code_candidates = [
                node_id
                for node_id in dynamic_mutable
                if node_id in anchor_selected
                and node_id not in (delta_left, delta_right)
            ]
            if len(code_candidates) < base_bits_needed:
                continue
            code_bits = tuple(code_candidates[:base_bits_needed])
            untouched = anchor_selected.difference(
                set((delta_left, delta_right)).union(code_bits)
            )
            if not untouched:
                continue
            signatures = {safe.signature_hash: safe}
            quartets = []
            for q in range(requested):
                base = set(anchor_selected)
                for bit_index, node_id in enumerate(code_bits):
                    if (q >> bit_index) & 1:
                        base.remove(node_id)
                q00 = self.make_signature(
                    base, anchor=strong_anchor, mutable_override=dynamic_mutable
                )
                q10 = self.make_signature(
                    base.difference([delta_left]),
                    anchor=strong_anchor,
                    mutable_override=dynamic_mutable,
                )
                q01 = self.make_signature(
                    base.difference([delta_right]),
                    anchor=strong_anchor,
                    mutable_override=dynamic_mutable,
                )
                q11 = self.make_signature(
                    base.difference([delta_left, delta_right]),
                    anchor=strong_anchor,
                    mutable_override=dynamic_mutable,
                )
                for member in (q00, q10, q01, q11):
                    signatures[member.signature_hash] = member
                quartets.append(
                    Quartet(
                        quartet_id="a%d-q%d" % (self.agent_id, q),
                        q00=q00.signature_hash,
                        q10=q10.signature_hash,
                        q01=q01.signature_hash,
                        q11=q11.signature_hash,
                    )
                )
            expected = 1 + 4 * requested
            if len(signatures) == expected:
                return list(signatures.values()), quartets

        # Build an independent basis so the Cartesian quartet construction is
        # guaranteed not to collapse under boundary projection.  For four
        # quartets this uses two delta bits, one always-on anchor bit and two
        # base-code bits: 1 + 4*4 exactly unique signatures.
        extras_needed = 1 + base_bits_needed
        if self.mutable_left and self.mutable_right:
            delta_pairs = [
                (left, right)
                for left in self.mutable_left
                for right in self.mutable_right
                if left != right
            ]
        else:
            delta_pairs = list(itertools.combinations(self.mutable, 2))
        delta_pairs.sort(
            key=lambda pair: (
                -(self._score[pair[0]] + self._score[pair[1]]),
                pair[0],
                pair[1],
            )
        )

        basis = None
        for delta_left, delta_right in delta_pairs:
            if delta_right in set(int(x) for x in self._neighbors(delta_left)):
                continue
            remaining = [
                node_id
                for node_id in self.mutable
                if node_id not in (delta_left, delta_right)
            ]
            remaining.sort(key=lambda n: (-self._score[n], n))
            for extras in itertools.combinations(remaining, extras_needed):
                chosen = (delta_left, delta_right) + tuple(extras)
                independent = True
                for first, second in itertools.combinations(chosen, 2):
                    if second in set(int(x) for x in self._neighbors(first)):
                        independent = False
                        break
                if independent:
                    basis = chosen
                    break
            if basis is not None:
                break
        if basis is None:
            raise ValueError(
                "agent {} cannot construct {} unique Top-H warm-up quartets; "
                "increase boundary_top_h".format(self.agent_id, requested)
            )

        delta_left, delta_right = basis[0], basis[1]
        anchor_bit = basis[2]
        code_bits = basis[3:]
        signatures: Dict[str, BoundarySignature] = {safe.signature_hash: safe}
        quartets: List[Quartet] = []
        for q in range(requested):
            base = {anchor_bit}
            for bit_index, node_id in enumerate(code_bits):
                if (q >> bit_index) & 1:
                    base.add(node_id)
            q00 = self.make_signature(base)
            q10 = self.make_signature(base.union([delta_left]))
            q01 = self.make_signature(base.union([delta_right]))
            q11 = self.make_signature(base.union([delta_left, delta_right]))
            members = (q00, q10, q01, q11)
            for member in members:
                signatures[member.signature_hash] = member
            quartets.append(
                Quartet(
                    quartet_id="a%d-q%d" % (self.agent_id, q),
                    q00=q00.signature_hash,
                    q10=q10.signature_hash,
                    q01=q01.signature_hash,
                    q11=q11.signature_hash,
                )
            )
        expected = 1 + 4 * requested
        if len(signatures) != expected:
            raise RuntimeError(
                "warm-up uniqueness invariant failed: expected {}, got {}".format(
                    expected, len(signatures)
                )
            )
        return list(signatures.values()), quartets

    def messages_base(
        self,
        anchor: BoundarySignature,
        count: int,
        round_id: int,
        exclude_hashes: Iterable[str] = (),
    ) -> List[BoundarySignature]:
        """Enumerate then sample deterministic Top-H perturbations."""

        target = max(0, int(count))
        excluded = set(exclude_hashes)
        excluded.add(anchor.signature_hash)
        result: Dict[str, BoundarySignature] = {}
        _, _, mutable = self.mutable_for_anchor(anchor)

        # Constructive budget-feasibility guard: turning one bits off can never
        # introduce a new cut conflict with the neighboring incumbent tuple.
        # These conservative messages_base ensure CCQM has compatible coverage
        # options before value-driven add/drop perturbations are considered.
        selected_mutable = [
            node_id for node_id in mutable if node_id in set(anchor.selected_ids)
        ]
        conservative_masks: List[Tuple[int, ...]] = []
        conservative_masks.extend((node_id,) for node_id in selected_mutable)
        conservative_masks.extend(itertools.combinations(selected_mutable, 2))
        for mask in conservative_masks:
            signature = self._toggle(anchor, mask, mutable_override=mutable)
            if signature.signature_hash not in excluded:
                result.setdefault(signature.signature_hash, signature)
            if len(result) >= target:
                return list(result.values())[:target]

        masks: List[Tuple[int, ...]] = []
        for width in (1, 2, 3):
            masks.extend(itertools.combinations(mutable, width))

        rng = random.Random(
            self.seed + 104729 * (self.agent_id + 1) + 1009 * int(round_id)
        )
        masks = list(masks)
        rng.shuffle(masks)
        max_masks = max(target * 20, 256)
        for mask in masks[:max_masks]:
            signature = self._toggle(anchor, mask, mutable_override=mutable)
            if signature.signature_hash not in excluded:
                result.setdefault(signature.signature_hash, signature)
            if len(result) >= target:
                break

        # Fill from random binary masks when projection caused many duplicates.
        attempts = 0
        while len(result) < target and mutable and attempts < max_masks:
            attempts += 1
            width = rng.randint(1, len(mutable))
            mask = rng.sample(list(mutable), width)
            signature = self._toggle(anchor, mask, mutable_override=mutable)
            if signature.signature_hash not in excluded:
                result.setdefault(signature.signature_hash, signature)
        return list(result.values())[:target]


def signature_assignment(signature: BoundarySignature) -> Dict[int, int]:
    """Explicit adapter used by coordination messages."""

    return signature.assignment()
