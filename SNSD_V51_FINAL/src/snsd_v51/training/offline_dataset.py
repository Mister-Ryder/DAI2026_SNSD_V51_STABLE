"""Leakage-safe warm-up teacher data for TriGraph v1.5.

The primary supervision unit is one complete six-agent warm-up candidate
factor graph, not an isolated local worker result.  For every seed this module
reconstructs the deterministic ``6 x 17`` signature table, joins worker values
by ``(seed, agent_id, signature_hash)``, and runs exact chain Max-Sum over the
complete cut-edge set.  The resulting labels include local gain, forced best
completion, best-completion advantage, externality and opportunity cost.

Feature and label payloads are deliberately different dataclasses and are
written to different JSONL files.  ``PrequeryCandidate.model_input_dict`` has
no objective-valued field.  Worker values are reachable only through
``CandidateTargets`` after the signature join.

Arc pressure is a masked label.  A boundary arc is identifiable only when the
realized candidate domain contains at least one finite best completion with
the arc on and at least one with it off.  Missing sides are never encoded as a
zero pressure; zero is valid only when both finite maxima exist and are equal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from snsd_core.candidates import BoundarySignature, CandidateGenerator
from snsd_core.data import load_arcs
from snsd_core.graph import (
    ConflictParameters,
    build_conflict_graph,
    partition_equal_time,
)
from snsd_core.solver import ConditionedLocalSolver
from snsd_core.verifier import verify_selected_ids
from snsd_v51.factor_graph import CandidateAction, solve_chain_max_sum


SCHEMA_VERSION = 2
DEFAULT_TRAIN_SEEDS = (44, 55, 66, 77, 88)
DEFAULT_VALIDATION_SEEDS = (33,)
DEFAULT_TEST_SEEDS = (11, 22)
FORBIDDEN_MODEL_INPUT_TOKENS = (
    "objective",
    "realized",
    "worker_result",
    "teacher_label",
    "local_gain",
    "externality",
    "completion_advantage",
    "opportunity_cost",
)


class OfflineDataError(ValueError):
    """Raised when an offline artifact violates its auditable data contract."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as exc:
        raise OfflineDataError("cannot read JSON artifact %s: %s" % (path, exc))
    if not isinstance(payload, dict):
        raise OfflineDataError("JSON artifact must contain an object: %s" % path)
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    os.replace(str(temporary), str(path))


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise OfflineDataError("%s must be numeric" % name)
    if not math.isfinite(result):
        raise OfflineDataError("%s must be finite" % name)
    return result


def _seed_from_run(run_dir: Path, warmup: Mapping[str, Any]) -> int:
    candidates = []  # type: List[Any]
    candidates.append(warmup.get("seed"))
    for filename in (
        "shared_manifest.json",
        "comparison_config.resolved.json",
        "config.resolved.json",
        "teacher_config.json",
    ):
        path = run_dir / filename
        if path.exists():
            payload = _load_json(path)
            candidates.append(payload.get("seed"))
    match = re.search(r"seed[_-]?(\d+)", run_dir.name, flags=re.IGNORECASE)
    if match:
        candidates.append(match.group(1))
    parsed = []
    for value in candidates:
        if value is None:
            continue
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            raise OfflineDataError("invalid seed in %s: %r" % (run_dir, value))
    if not parsed:
        raise OfflineDataError("cannot determine seed for %s" % run_dir)
    if len(set(parsed)) != 1:
        raise OfflineDataError("conflicting seed metadata in %s: %s" % (run_dir, parsed))
    return parsed[0]


def _config_for_run(run_dir: Path) -> Dict[str, Any]:
    for filename in (
        "comparison_config.resolved.json",
        "config.resolved.json",
        "teacher_config.json",
    ):
        path = run_dir / filename
        if path.exists():
            return _load_json(path)
    return {}


def _resolve_data_path(
    run_dir: Path, config: Mapping[str, Any], explicit: Optional[str]
) -> Path:
    raw = explicit if explicit is not None else config.get("data_path")
    if raw is None:
        raise OfflineDataError(
            "data_path is required because exact cut-edge teacher labels cannot be "
            "derived from warmup_manifest.json alone"
        )
    path = Path(str(raw))
    candidates = []  # type: List[Path]
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            (
                Path.cwd() / path,
                run_dir / path,
                run_dir.parent / path,
                run_dir.parent.parent / path,
                Path(__file__).resolve().parents[2] / path,
            )
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise OfflineDataError(
        "cannot resolve data_path %r; checked %s"
        % (raw, ", ".join(str(item.resolve()) for item in candidates))
    )


def _problem_from_config(data_path: Path, config: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    dataset = load_arcs(str(data_path))
    graph_config = dict(config.get("graph", {}))
    parameters = ConflictParameters(
        ground_trans_time=int(graph_config.get("ground_transition", 340)),
        satellite_change_time=int(graph_config.get("satellite_change", 150)),
        satellite_trans_time=int(graph_config.get("satellite_transition", 300)),
    )
    graph = build_conflict_graph(dataset.arcs, parameters)
    partitions = partition_equal_time(
        dataset.arcs,
        graph,
        k=int(graph_config.get("num_agents", 6)),
        horizon_start=int(graph_config.get("horizon_start", 0)),
        horizon_end=int(graph_config.get("horizon_end", 259200)),
    )
    if partitions.has_non_adjacent_cuts:
        raise OfflineDataError("exact chain teacher cannot accept non-adjacent cut edges")
    return dataset, graph, partitions


def cut_edges_by_link(partitions: Any) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    links = [[] for _ in range(int(partitions.k) - 1)]  # type: List[List[Tuple[int, int]]]
    for raw_u, raw_v in partitions.cut_edges:
        u, v = int(raw_u), int(raw_v)
        pu = int(partitions.partition_id[u])
        pv = int(partitions.partition_id[v])
        if abs(pu - pv) != 1:
            raise OfflineDataError("non-adjacent cut edge (%d, %d)" % (u, v))
        if pu < pv:
            links[pu].append((u, v))
        else:
            links[pv].append((v, u))
    return tuple(tuple(sorted(values)) for values in links)


def _signature_from_dict(payload: Mapping[str, Any], expected_agent: int) -> BoundarySignature:
    try:
        agent_id = int(payload["agent_id"])
        left = tuple(int(value) for value in payload.get("left_selected", ()))
        right = tuple(int(value) for value in payload.get("right_selected", ()))
        boundary = tuple(int(value) for value in payload["boundary_order"])
        order_hash = str(payload["boundary_order_hash"])
        signature_hash = str(payload["signature_hash"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OfflineDataError("invalid boundary signature: %s" % exc)
    if agent_id != int(expected_agent):
        raise OfflineDataError("signature agent does not match enclosing table")
    if len(boundary) != len(set(boundary)) or tuple(sorted(boundary)) != boundary:
        raise OfflineDataError("boundary_order must be sorted and unique")
    selected = set(left).union(right)
    if not selected.issubset(set(boundary)):
        raise OfflineDataError("signature selects a node outside boundary_order")
    expected_order_hash = hashlib.sha256(
        json.dumps(
            {"agent": agent_id, "boundary": boundary},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if order_hash != expected_order_hash:
        raise OfflineDataError("boundary_order_hash mismatch for agent %d" % agent_id)
    bits = "".join("1" if node in selected else "0" for node in boundary)
    expected_signature_hash = hashlib.sha256(
        ("%d|%s|%s" % (agent_id, order_hash, bits)).encode("ascii")
    ).hexdigest()
    if signature_hash != expected_signature_hash:
        raise OfflineDataError("signature_hash mismatch for agent %d" % agent_id)
    return BoundarySignature(
        agent_id=agent_id,
        left_selected=tuple(sorted(left)),
        right_selected=tuple(sorted(right)),
        boundary_order=boundary,
        boundary_order_hash=order_hash,
        signature_hash=signature_hash,
    )


@dataclass(frozen=True)
class PrequeryCandidate:
    """Public candidate context available after warm-up and before a new query."""

    run_id: str
    seed: int
    agent_id: int
    round_id: int
    signature_hash: str
    boundary_order_hash: str
    boundary_order: Tuple[int, ...]
    selected_boundary: Tuple[int, ...]
    anchor_signature_hash: str
    anchor_selected_boundary: Tuple[int, ...]
    source: str = "warmup_teacher"
    is_safe_anchor: bool = False

    def join_key(self) -> Tuple[int, int, int, str]:
        return (self.seed, self.round_id, self.agent_id, self.signature_hash)

    def model_input_dict(self) -> Dict[str, Any]:
        """Return the only payload allowed to enter pre-query graph_layers scoring."""

        return {
            "agent_id": self.agent_id,
            "round_id": self.round_id,
            "signature_hash": self.signature_hash,
            "boundary_order_hash": self.boundary_order_hash,
            "boundary_order": list(self.boundary_order),
            "requested_selected_boundary": list(self.selected_boundary),
            "anchor_signature_hash": self.anchor_signature_hash,
            "anchor_selected_boundary": list(self.anchor_selected_boundary),
            "source": self.source,
            "is_safe_anchor": self.is_safe_anchor,
        }

    def provenance_dict(self) -> Dict[str, Any]:
        result = self.model_input_dict()
        result.update({"run_id": self.run_id, "seed": self.seed})
        return result


@dataclass(frozen=True)
class CandidateTargets:
    """Post-join teacher quantities; none may enter ``model_input_dict``."""

    solver_status: str
    local_objective: float
    local_gain: Optional[float]
    local_gain_mask: bool
    forced_best_completion: Optional[float]
    best_completion_advantage: Optional[float]
    advantage_mask: bool
    externality: Optional[float]
    externality_mask: bool
    left_opportunity_cost: Optional[float]
    right_opportunity_cost: Optional[float]
    opportunity_cost: Optional[float]
    completion_feasible: bool
    completion_mask: bool
    uncertainty_mask: bool
    teacher_source: str = "warmup_exact_chain_maxsum"

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class OfflineExample:
    prequery: PrequeryCandidate
    targets: CandidateTargets

    def join_key(self) -> Tuple[int, int, int, str]:
        return self.prequery.join_key()


@dataclass(frozen=True)
class ArcPressureLabel:
    seed: int
    agent_id: int
    round_id: int
    node_id: int
    boundary_order_hash: str
    incumbent_bit: int
    finite_on_count: int
    finite_off_count: int
    best_on: Optional[float]
    best_off: Optional[float]
    pressure: Optional[float]
    pressure_identifiable: bool
    teacher_source: str = "warmup_exact_chain_maxsum"

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self.__dict__)


def audit_model_input(payload: Mapping[str, Any]) -> None:
    """Reject label-like keys recursively before a training batch is built."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key).strip().lower()
                if any(token in key for token in FORBIDDEN_MODEL_INPUT_TOKENS):
                    raise OfflineDataError("forbidden graph_layers-input key at %s.%s" % (path, key))
                visit(item, "%s.%s" % (path, key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, "%s[%d]" % (path, index))

    visit(payload, "model_input")


def _pressure_labels(
    seed: int,
    agent_id: int,
    boundary_order: Sequence[int],
    boundary_order_hash: str,
    incumbent_selected: Set[int],
    signatures: Sequence[BoundarySignature],
    forced_values: Sequence[float],
) -> Tuple[ArcPressureLabel, ...]:
    rows = []
    for node_id in boundary_order:
        on = [
            float(value)
            for signature, value in zip(signatures, forced_values)
            if node_id in signature.selected_ids and math.isfinite(float(value))
        ]
        off = [
            float(value)
            for signature, value in zip(signatures, forced_values)
            if node_id not in signature.selected_ids and math.isfinite(float(value))
        ]
        identifiable = bool(on and off)
        best_on = max(on) if on else None
        best_off = max(off) if off else None
        pressure = (best_on - best_off) if identifiable else None
        rows.append(
            ArcPressureLabel(
                seed=int(seed),
                agent_id=int(agent_id),
                round_id=-1,
                node_id=int(node_id),
                boundary_order_hash=str(boundary_order_hash),
                incumbent_bit=int(node_id in incumbent_selected),
                finite_on_count=len(on),
                finite_off_count=len(off),
                best_on=best_on,
                best_off=best_off,
                pressure=pressure,
                pressure_identifiable=identifiable,
            )
        )
    return tuple(rows)


def _audit_online_manifests(run_dir: Path) -> Dict[str, Any]:
    paths = {
        name: run_dir / (name + ".json")
        for name in ("proposal_manifest", "prediction_manifest", "reveal_ledger")
    }
    if not all(path.exists() for path in paths.values()):
        return {"available": False}
    proposal = _load_json(paths["proposal_manifest"])
    prediction = _load_json(paths["prediction_manifest"])
    reveal = _load_json(paths["reveal_ledger"])
    if proposal.get("generated_before_online_labels") is not True:
        raise OfflineDataError("proposal stream was not registered before online labels")
    proposal_keys = set()
    for round_row in proposal.get("rounds", ()):
        round_id = int(round_row["round_id"])
        for raw_agent, rows in round_row.get("proposals_by_agent", {}).items():
            agent_id = int(raw_agent)
            for row in rows:
                key = (round_id, agent_id, str(row["signature_hash"]))
                if key in proposal_keys:
                    raise OfflineDataError("duplicate proposal key in %s" % run_dir)
                proposal_keys.add(key)
    prediction_keys = set()
    for round_row in prediction.get("rounds", ()):
        if round_row.get("written_before_any_round_query_label") is not True:
            raise OfflineDataError("prediction was not written before round reveal")
        round_id = int(round_row["round_id"])
        for raw_agent, rows in round_row.get("predictions_by_agent", {}).items():
            agent_id = int(raw_agent)
            for row in rows:
                prediction_keys.add((round_id, agent_id, str(row["signature_hash"])))
    if prediction_keys != proposal_keys:
        raise OfflineDataError("proposal/prediction key sets differ in %s" % run_dir)
    reveal_values = {}  # type: Dict[Tuple[int, int, str], Tuple[str, float]]
    raw_reveals = 0
    for round_row in reveal.get("rounds", ()):
        if round_row.get("selection_frozen_before_materialization") is not True:
            raise OfflineDataError("reveal selection was not frozen before materialization")
        round_id = int(round_row["round_id"])
        for rows in round_row.get("methods", {}).values():
            for row in rows:
                raw_reveals += 1
                key = (round_id, int(row["agent_id"]), str(row["signature_hash"]))
                value = (str(row["status"]), _finite("reveal objective", row["objective"]))
                if key in reveal_values and reveal_values[key] != value:
                    raise OfflineDataError("conflicting duplicate reveal for %s" % (key,))
                reveal_values[key] = value
    orphan = set(reveal_values).difference(proposal_keys)
    if orphan:
        raise OfflineDataError("reveal ledger contains %d orphan candidates" % len(orphan))
    return {
        "available": True,
        "proposal_count": len(proposal_keys),
        "prediction_count": len(prediction_keys),
        "raw_reveal_rows": raw_reveals,
        "unique_reveal_labels": len(reveal_values),
        "duplicate_logical_reveals": raw_reveals - len(reveal_values),
        "proposal_reveal_coverage": (
            float(len(reveal_values)) / float(len(proposal_keys)) if proposal_keys else 0.0
        ),
        "source_hashes": {name: _file_digest(path) for name, path in paths.items()},
    }


@dataclass(frozen=True)
class OfflineDataset:
    examples: Tuple[OfflineExample, ...]
    arc_pressures: Tuple[ArcPressureLabel, ...]
    receipt: Mapping[str, Any]
    dataset: Any = None
    graph: Any = None
    partitions: Any = None
    generators_by_seed: Optional[Mapping[int, Tuple[CandidateGenerator, ...]]] = None

    @property
    def seeds(self) -> Tuple[int, ...]:
        return tuple(sorted(set(item.prequery.seed for item in self.examples)))

    def examples_for_seeds(self, seeds: Iterable[int]) -> Tuple[OfflineExample, ...]:
        wanted = set(int(value) for value in seeds)
        return tuple(item for item in self.examples if item.prequery.seed in wanted)

    def pressures_for_seeds(self, seeds: Iterable[int]) -> Tuple[ArcPressureLabel, ...]:
        wanted = set(int(value) for value in seeds)
        return tuple(item for item in self.arc_pressures if item.seed in wanted)

    def split_by_seed(
        self,
        train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
        validation_seeds: Sequence[int] = DEFAULT_VALIDATION_SEEDS,
        test_seeds: Sequence[int] = DEFAULT_TEST_SEEDS,
        require_all: bool = True,
    ) -> "DatasetSplit":
        train = tuple(int(value) for value in train_seeds)
        validation = tuple(int(value) for value in validation_seeds)
        test = tuple(int(value) for value in test_seeds)
        if set(train).intersection(validation) or set(train).intersection(test) or set(validation).intersection(test):
            raise OfflineDataError("train/validation/test seed groups must be disjoint")
        available = set(self.seeds)
        requested = set(train).union(validation).union(test)
        missing = sorted(requested.difference(available))
        if require_all and missing:
            raise OfflineDataError("missing requested seed groups: %s" % missing)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "split_unit": "seed",
            "train_seeds": list(train),
            "validation_seeds": list(validation),
            "test_seeds": list(test),
            "available_seeds": list(self.seeds),
            "missing_requested_seeds": missing,
            "seed_overlap": False,
            "same_instance_scope": (
                "trajectory/worker-seed robustness on one C3 instance; not cross-instance generalization"
            ),
        }
        return DatasetSplit(
            train=self.examples_for_seeds(train),
            validation=self.examples_for_seeds(validation),
            test=self.examples_for_seeds(test),
            train_pressures=self.pressures_for_seeds(train),
            validation_pressures=self.pressures_for_seeds(validation),
            test_pressures=self.pressures_for_seeds(test),
            manifest=manifest,
        )

    def write_artifacts(self, output_dir: str) -> Dict[str, str]:
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        prequery_path = destination / "prequery_candidates.jsonl"
        labels_path = destination / "teacher_labels.jsonl"
        pressure_path = destination / "arc_pressures.jsonl"
        receipt_path = destination / "dataset_receipt.json"
        _atomic_jsonl(prequery_path, (item.prequery.provenance_dict() for item in self.examples))
        _atomic_jsonl(
            labels_path,
            (
                {
                    "seed": item.prequery.seed,
                    "round_id": item.prequery.round_id,
                    "agent_id": item.prequery.agent_id,
                    "signature_hash": item.prequery.signature_hash,
                    "targets": item.targets.to_dict(),
                }
                for item in self.examples
            ),
        )
        _atomic_jsonl(pressure_path, (item.to_dict() for item in self.arc_pressures))
        _atomic_json(receipt_path, self.receipt)
        return {
            "prequery_candidates": str(prequery_path),
            "teacher_labels": str(labels_path),
            "arc_pressures": str(pressure_path),
            "dataset_receipt": str(receipt_path),
        }


@dataclass(frozen=True)
class DatasetSplit:
    train: Tuple[OfflineExample, ...]
    validation: Tuple[OfflineExample, ...]
    test: Tuple[OfflineExample, ...]
    train_pressures: Tuple[ArcPressureLabel, ...]
    validation_pressures: Tuple[ArcPressureLabel, ...]
    test_pressures: Tuple[ArcPressureLabel, ...]
    manifest: Mapping[str, Any]

    def write_manifest(self, path: str) -> None:
        payload = dict(self.manifest)
        payload["counts"] = {
            "train_candidates": len(self.train),
            "validation_candidates": len(self.validation),
            "test_candidates": len(self.test),
            "train_identifiable_pressures": sum(
                item.pressure_identifiable for item in self.train_pressures
            ),
            "validation_identifiable_pressures": sum(
                item.pressure_identifiable for item in self.validation_pressures
            ),
            "test_identifiable_pressures": sum(
                item.pressure_identifiable for item in self.test_pressures
            ),
        }
        _atomic_json(Path(path), payload)


def _warmup_table(
    run_dir: Path,
    seed: int,
    warmup: Mapping[str, Any],
    dataset: Any,
    graph: Any,
    partitions: Any,
    top_h: int,
    quartets: int,
) -> Tuple[
    Tuple[OfflineExample, ...],
    Tuple[ArcPressureLabel, ...],
    Tuple[CandidateGenerator, ...],
    Dict[str, Any],
]:
    generators = tuple(
        CandidateGenerator(
            agent_id=agent_id,
            arcs=dataset.arcs,
            graph=graph,
            partitions=partitions,
            top_h=top_h,
            seed=seed,
        )
        for agent_id in range(int(partitions.k))
    )
    manifest_signatures = warmup.get("signatures_by_agent")
    if not isinstance(manifest_signatures, Mapping):
        raise OfflineDataError("warmup manifest has no signatures_by_agent")
    generated_by_agent = {}  # type: Dict[int, Tuple[BoundarySignature, ...]]
    quartet_hashes = {}
    for agent_id, generator in enumerate(generators):
        generated, generated_quartets = generator.warmup(quartets)
        generated_by_agent[agent_id] = tuple(generated)
        raw_rows = manifest_signatures.get(str(agent_id), manifest_signatures.get(agent_id))
        if raw_rows is None:
            raise OfflineDataError("warmup manifest is missing agent %d signatures" % agent_id)
        parsed = tuple(_signature_from_dict(row, agent_id) for row in raw_rows)
        generated_hashes = [item.signature_hash for item in generated]
        parsed_hashes = [item.signature_hash for item in parsed]
        if set(generated_hashes) != set(parsed_hashes):
            raise OfflineDataError(
                "CandidateGenerator.warmup hash reconstruction failed for seed %d agent %d"
                % (seed, agent_id)
            )
        if len(parsed_hashes) != len(set(parsed_hashes)):
            raise OfflineDataError("duplicate warmup signature for agent %d" % agent_id)
        quartet_hashes[str(agent_id)] = _stable_digest(
            [
                {
                    "quartet_id": item.quartet_id,
                    "q00": item.q00,
                    "q10": item.q10,
                    "q01": item.q01,
                    "q11": item.q11,
                }
                for item in generated_quartets
            ]
        )
    results = {}  # type: Dict[Tuple[int, str], Dict[str, Any]]
    for row in warmup.get("results", ()):
        key = (int(row["agent_id"]), str(row["signature_hash"]))
        value = {
            "status": str(row["status"]),
            "objective": _finite("warmup objective", row["objective"]),
        }
        if key in results and results[key] != value:
            raise OfflineDataError("conflicting warmup result for %s" % (key,))
        results[key] = value
    expected_keys = {
        (agent_id, signature.signature_hash)
        for agent_id, signatures in generated_by_agent.items()
        for signature in signatures
    }
    if set(results) != expected_keys:
        raise OfflineDataError(
            "warmup signature/result key mismatch: expected %d, observed %d"
            % (len(expected_keys), len(results))
        )
    raw_incumbents = warmup.get("initial_selected_signatures")
    if not isinstance(raw_incumbents, Mapping):
        raise OfflineDataError("warmup manifest has no initial_selected_signatures")
    incumbent_ids = {
        agent_id: str(raw_incumbents.get(str(agent_id), raw_incumbents.get(agent_id)))
        for agent_id in range(int(partitions.k))
    }
    actions = []
    for agent_id in range(int(partitions.k)):
        table = []
        empty_id = generators[agent_id].empty_signature().signature_hash
        for signature in generated_by_agent[agent_id]:
            row = results[(agent_id, signature.signature_hash)]
            table.append(
                CandidateAction(
                    agent_id=agent_id,
                    candidate_id=signature.signature_hash,
                    local_objective=row["objective"],
                    selected_boundary=frozenset(signature.selected_ids),
                    is_incumbent=signature.signature_hash == incumbent_ids[agent_id],
                    is_safe=signature.signature_hash == empty_id,
                    is_feasible=row["status"] == "feasible",
                )
            )
        actions.append(table)
    factor = solve_chain_max_sum(
        actions,
        cut_edges_by_link(partitions),
        require_incumbent=True,
        require_safe_anchor=True,
    )
    manifest_initial = _finite("initial_objective", warmup["initial_objective"])
    if abs(float(factor.best_value) - manifest_initial) > 1.0e-6:
        raise OfflineDataError(
            "exact warmup value %.9f differs from manifest %.9f for seed %d"
            % (factor.best_value, manifest_initial, seed)
        )
    examples = []
    pressures = []
    completion_positive = 0
    for agent_id in range(int(partitions.k)):
        signature_by_hash = {
            item.signature_hash: item for item in generated_by_agent[agent_id]
        }
        anchor = signature_by_hash[incumbent_ids[agent_id]]
        ordered_signatures = generated_by_agent[agent_id]
        for candidate_index, signature in enumerate(ordered_signatures):
            diagnostics = factor.diagnostics(agent_id, candidate_index)
            finite_advantage = math.isfinite(diagnostics.best_completion_advantage)
            finite_externality = math.isfinite(diagnostics.externality)
            finite_forced = math.isfinite(diagnostics.forced_best_completion)
            completion_positive += int(diagnostics.completion_feasible)
            result_row = results[(agent_id, signature.signature_hash)]
            prequery = PrequeryCandidate(
                run_id=run_dir.name,
                seed=seed,
                agent_id=agent_id,
                round_id=-1,
                signature_hash=signature.signature_hash,
                boundary_order_hash=signature.boundary_order_hash,
                boundary_order=tuple(signature.boundary_order),
                selected_boundary=tuple(signature.selected_ids),
                anchor_signature_hash=anchor.signature_hash,
                anchor_selected_boundary=tuple(anchor.selected_ids),
                is_safe_anchor=not bool(signature.selected_ids),
            )
            audit_model_input(prequery.model_input_dict())
            examples.append(
                OfflineExample(
                    prequery=prequery,
                    targets=CandidateTargets(
                        solver_status=result_row["status"],
                        local_objective=result_row["objective"],
                        local_gain=(
                            diagnostics.local_gain
                            if math.isfinite(diagnostics.local_gain)
                            else None
                        ),
                        local_gain_mask=(
                            result_row["status"] == "feasible"
                            and math.isfinite(diagnostics.local_gain)
                        ),
                        forced_best_completion=(
                            diagnostics.forced_best_completion if finite_forced else None
                        ),
                        best_completion_advantage=(
                            diagnostics.best_completion_advantage
                            if finite_advantage
                            else None
                        ),
                        advantage_mask=finite_advantage,
                        externality=(
                            diagnostics.externality if finite_externality else None
                        ),
                        externality_mask=finite_externality,
                        left_opportunity_cost=(
                            diagnostics.left_opportunity_cost
                            if math.isfinite(diagnostics.left_opportunity_cost)
                            else None
                        ),
                        right_opportunity_cost=(
                            diagnostics.right_opportunity_cost
                            if math.isfinite(diagnostics.right_opportunity_cost)
                            else None
                        ),
                        opportunity_cost=(
                            diagnostics.opportunity_cost
                            if math.isfinite(diagnostics.opportunity_cost)
                            else None
                        ),
                        completion_feasible=diagnostics.completion_feasible,
                        completion_mask=True,
                        uncertainty_mask=finite_advantage,
                    ),
                )
            )
        pressures.extend(
            _pressure_labels(
                seed=seed,
                agent_id=agent_id,
                boundary_order=ordered_signatures[0].boundary_order,
                boundary_order_hash=ordered_signatures[0].boundary_order_hash,
                incumbent_selected=set(anchor.selected_ids),
                signatures=ordered_signatures,
                forced_values=factor.forced_best_completions[agent_id],
            )
        )
    run_receipt = {
        "seed": seed,
        "run_dir": str(run_dir),
        "warmup_manifest_hash": _file_digest(run_dir / "warmup_manifest.json"),
        "candidate_count": len(examples),
        "candidate_count_by_agent": [len(item) for item in generated_by_agent.values()],
        "quartet_hashes": quartet_hashes,
        "exact_restricted_value": factor.best_value,
        "completion_positive": completion_positive,
        "completion_negative": len(examples) - completion_positive,
        "pressure_total": len(pressures),
        "pressure_identifiable": sum(item.pressure_identifiable for item in pressures),
        "pressure_coverage": (
            float(sum(item.pressure_identifiable for item in pressures))
            / float(max(1, len(pressures)))
        ),
        "online_artifact_audit": _audit_online_manifests(run_dir),
    }
    return tuple(examples), tuple(pressures), generators, run_receipt


def load_offline_dataset(
    run_dirs: Sequence[str],
    data_path: Optional[str] = None,
    warmup_quartets: int = 4,
    strict: bool = True,
    prepared_problem: Optional[Tuple[Any, Any, Any]] = None,
) -> OfflineDataset:
    """Load warm-up manifests and derive exact joint teacher labels.

    ``strict`` is retained in the public API for future schema migrations.  The
    v2 teacher always enforces hash, join, factor-graph and leakage invariants;
    setting it to ``False`` only permits extra unrecognized JSON fields.
    """

    del strict
    paths = tuple(Path(value).resolve() for value in run_dirs)
    if not paths:
        raise OfflineDataError("at least one run directory is required")
    warmups = []
    seeds = []
    configs = []
    for run_dir in paths:
        warmup_path = run_dir / "warmup_manifest.json"
        if not warmup_path.exists():
            raise OfflineDataError("missing warmup manifest: %s" % warmup_path)
        warmup = _load_json(warmup_path)
        warmups.append(warmup)
        seeds.append(_seed_from_run(run_dir, warmup))
        configs.append(_config_for_run(run_dir))
    if len(seeds) != len(set(seeds)):
        raise OfflineDataError("run_dirs contain duplicate seed groups: %s" % seeds)
    if prepared_problem is None:
        resolved_data = _resolve_data_path(paths[0], configs[0], data_path)
        dataset, graph, partitions = _problem_from_config(resolved_data, configs[0])
    else:
        dataset, graph, partitions = prepared_problem
        resolved_data = Path(str(getattr(dataset, "source_path", data_path or "prepared")))
    if int(partitions.k) != 6:
        raise OfflineDataError("v1.5 C3 teacher expects exactly six agents")
    all_examples = []
    all_pressures = []
    run_receipts = []
    generators_by_seed = {}
    for run_dir, seed, warmup, config in zip(paths, seeds, warmups, configs):
        graph_config = dict(config.get("graph", {}))
        if graph_config and int(graph_config.get("num_agents", partitions.k)) != int(partitions.k):
            raise OfflineDataError("inconsistent agent count in %s" % run_dir)
        candidate_config = dict(config.get("candidates", {}))
        top_h = int(candidate_config.get("boundary_top_h", 8))
        examples, pressures, generators, receipt = _warmup_table(
            run_dir=run_dir,
            seed=seed,
            warmup=warmup,
            dataset=dataset,
            graph=graph,
            partitions=partitions,
            top_h=top_h,
            quartets=int(warmup_quartets),
        )
        all_examples.extend(examples)
        all_pressures.extend(pressures)
        generators_by_seed[seed] = generators
        run_receipts.append(receipt)
    join_keys = [item.join_key() for item in all_examples]
    if len(join_keys) != len(set(join_keys)):
        raise OfflineDataError("offline examples contain duplicate seed-scoped join keys")
    completion_positive = sum(item.targets.completion_feasible for item in all_examples)
    identifiable = sum(item.pressure_identifiable for item in all_pressures)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "teacher": "warmup_exact_chain_maxsum",
        "grain": "one row per (seed, round=-1, agent_id, signature_hash)",
        "join_key": ["seed", "round_id", "agent_id", "signature_hash"],
        "model_input_contract": "prequery_candidates.jsonl contains no objective-valued field",
        "data_path": str(resolved_data),
        "data_hash": _file_digest(resolved_data) if resolved_data.exists() else None,
        "graph_nodes": int(graph.num_nodes),
        "graph_edges": int(graph.num_edges),
        "cut_edges": int(len(partitions.cut_edges)),
        "seeds": sorted(seeds),
        "candidate_count": len(all_examples),
        "local_gain_label_count": sum(item.targets.local_gain_mask for item in all_examples),
        "advantage_label_count": sum(item.targets.advantage_mask for item in all_examples),
        "externality_label_count": sum(item.targets.externality_mask for item in all_examples),
        "completion_label_count": sum(item.targets.completion_mask for item in all_examples),
        "completion_positive": completion_positive,
        "completion_negative": len(all_examples) - completion_positive,
        "completion_is_degenerate": completion_positive in (0, len(all_examples)),
        "pressure_total": len(all_pressures),
        "pressure_identifiable": identifiable,
        "pressure_coverage": float(identifiable) / float(max(1, len(all_pressures))),
        "pressure_missing_policy": "mask_false; never impute zero",
        "runs": run_receipts,
    }
    receipt["dataset_hash"] = _stable_digest(
        {
            "prequery": [item.prequery.provenance_dict() for item in all_examples],
            "labels": [item.targets.to_dict() for item in all_examples],
            "pressures": [item.to_dict() for item in all_pressures],
        }
    )
    return OfflineDataset(
        examples=tuple(all_examples),
        arc_pressures=tuple(all_pressures),
        receipt=receipt,
        dataset=dataset,
        graph=graph,
        partitions=partitions,
        generators_by_seed=generators_by_seed,
    )


def materialize_warmup_manifest(
    seed: int,
    data_path: str,
    output_dir: str,
    moves: int = 500,
    restarts: int = 2,
    boundary_top_h: int = 8,
    warmup_quartets: int = 4,
    num_agents: int = 6,
    horizon_start: int = 0,
    horizon_end: int = 259200,
) -> Path:
    """Generate one training-only warm-up manifest with the canonical v1 worker.

    This function intentionally refuses to overwrite an existing manifest.
    It performs only the 17 warm-up queries per agent and does not run online
    proposal experiments.
    """

    destination = Path(output_dir).resolve()
    manifest_path = destination / "warmup_manifest.json"
    if manifest_path.exists():
        raise OfflineDataError("refusing to overwrite %s" % manifest_path)
    config = {
        "schema_version": SCHEMA_VERSION,
        "scenario": "C3",
        "seed": int(seed),
        "data_path": str(Path(data_path).resolve()),
        "graph": {
            "num_agents": int(num_agents),
            "horizon_start": int(horizon_start),
            "horizon_end": int(horizon_end),
            "ground_transition": 340,
            "satellite_change": 150,
            "satellite_transition": 300,
        },
        "candidates": {
            "boundary_top_h": int(boundary_top_h),
            "warmup_quartets": int(warmup_quartets),
        },
        "local_solver": {"moves": int(moves), "restarts": int(restarts)},
        "purpose": "v1.5 offline teacher only; no online evaluation",
    }
    dataset, graph, partitions = _problem_from_config(Path(data_path).resolve(), config)
    if int(partitions.k) != int(num_agents):
        raise OfflineDataError("partition agent count mismatch")
    generators = [
        CandidateGenerator(
            agent_id=agent_id,
            arcs=dataset.arcs,
            graph=graph,
            partitions=partitions,
            top_h=int(boundary_top_h),
            seed=int(seed),
        )
        for agent_id in range(int(num_agents))
    ]
    solver = ConditionedLocalSolver(
        dataset.arcs,
        graph,
        partitions,
        moves=int(moves),
        restarts=int(restarts),
        seed=int(seed),
    )
    signatures_by_agent = {}
    quartets_by_agent = {}
    result_objects = {}
    action_tables = []
    for agent_id, generator in enumerate(generators):
        signatures, quartets_for_agent = generator.warmup(int(warmup_quartets))
        signatures_by_agent[agent_id] = signatures
        quartets_by_agent[agent_id] = quartets_for_agent
        actions = []
        empty_id = generator.empty_signature().signature_hash
        for signature in signatures:
            result = solver.solve(agent_id, signature)
            result_objects[(agent_id, signature.signature_hash)] = result
            actions.append(
                CandidateAction(
                    agent_id=agent_id,
                    candidate_id=signature.signature_hash,
                    local_objective=float(result.objective),
                    selected_boundary=frozenset(signature.selected_ids),
                    is_safe=signature.signature_hash == empty_id,
                    is_feasible=result.status == "feasible",
                )
            )
        action_tables.append(actions)
    initial_factor = solve_chain_max_sum(
        action_tables,
        cut_edges_by_link(partitions),
        require_incumbent=False,
        require_safe_anchor=True,
    )
    incumbent_ids = {
        agent_id: initial_factor.best_actions[agent_id].candidate_id
        for agent_id in range(int(num_agents))
    }
    selected_ids = tuple(
        sorted(
            node
            for agent_id, candidate_id in incumbent_ids.items()
            for node in result_objects[(agent_id, candidate_id)].selected_ids
        )
    )
    verification = verify_selected_ids(graph, selected_ids)
    if not verification.feasible:
        raise OfflineDataError("generated warmup incumbent is invalid: %s" % verification.message)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "logical_query_calls": len(result_objects),
        "signatures_by_agent": {
            agent_id: [signature.to_dict() for signature in signatures]
            for agent_id, signatures in signatures_by_agent.items()
        },
        "quartets_by_agent": {
            agent_id: [
                {
                    "quartet_id": item.quartet_id,
                    "q00": item.q00,
                    "q10": item.q10,
                    "q01": item.q01,
                    "q11": item.q11,
                }
                for item in rows
            ]
            for agent_id, rows in quartets_by_agent.items()
        },
        "results": [
            {
                "agent_id": key[0],
                "signature_hash": key[1],
                "status": result.status,
                "objective": float(result.objective),
                "selected_ids_hash": _stable_digest(result.selected_ids),
                "selected_count": len(result.selected_ids),
                "solver_seed": int(result.seed),
                "runtime_s": float(result.runtime_s),
            }
            for key, result in sorted(result_objects.items())
        ],
        "initial_objective": float(initial_factor.best_value),
        "initial_selected_ids_hash": _stable_digest(selected_ids),
        "initial_selected_signatures": incumbent_ids,
        "verification": verification.message,
        "teacher_generation": {
            "worker": "snsd_v1.ConditionedLocalSolver",
            "moves": int(moves),
            "restarts": int(restarts),
            "online_queries": 0,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "teacher_config.json", config)
    _atomic_json(manifest_path, manifest)
    return manifest_path


__all__ = [
    "ArcPressureLabel",
    "CandidateTargets",
    "DatasetSplit",
    "DEFAULT_TEST_SEEDS",
    "DEFAULT_TRAIN_SEEDS",
    "DEFAULT_VALIDATION_SEEDS",
    "OfflineDataError",
    "OfflineDataset",
    "OfflineExample",
    "PrequeryCandidate",
    "audit_model_input",
    "cut_edges_by_link",
    "load_offline_dataset",
    "materialize_warmup_manifest",
]
