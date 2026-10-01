"""Input data graph_layers and robust CSV loading for the satellite-link instances.

The legacy data files are GBK encoded and some historical code contains
mojibake versions of the Chinese column names.  This loader therefore uses
semantic column names when they are recognisable and otherwise falls back to
the stable six-column prefix used by C3/W3.
"""

from __future__ import absolute_import

import csv
import io
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Arc:
    """One feasible satellite-to-ground communication opportunity.

    ``id`` is a dense zero-based identifier.  The optimisation objective in
    the inherited KP-NLNS code is total link duration, exposed as both
    :attr:`link_time` and :attr:`weight`.
    """

    id: int
    ground: int
    satellite: int
    ground_name: str
    satellite_name: str
    link_st: int
    link_et: int
    trace_st: int
    trace_et: int
    priority: float = 0.0

    @property
    def link_time(self):  # type: () -> int
        return self.link_et - self.link_st

    @property
    def weight(self):  # type: () -> float
        return float(self.link_time)


@dataclass(frozen=True)
class ArcDataset:
    """Immutable parsed instance plus stable resource-index dictionaries."""

    arcs: Tuple[Arc, ...]
    ground_names: Tuple[str, ...]
    satellite_names: Tuple[str, ...]
    source_path: str
    encoding: str
    header: Tuple[str, ...]

    @property
    def num_arcs(self):  # type: () -> int
        return len(self.arcs)

    @property
    def num_grounds(self):  # type: () -> int
        return len(self.ground_names)

    @property
    def num_satellites(self):  # type: () -> int
        return len(self.satellite_names)


_HEADER_ALIASES = {
    "ground": (
        "地面站标识",
        "地面站标志",
        "ground",
        "groundid",
        "groundstation",
        "groundstationid",
        "station",
        "stationid",
    ),
    "satellite": (
        "卫星标识",
        "卫星标志",
        "satellite",
        "satelliteid",
        "sat",
        "satid",
    ),
    "link_st": (
        "开始建链时间",
        "建链开始时间",
        "linkstart",
        "linkstarttime",
        "linkst",
        "start",
    ),
    "link_et": (
        "结束建链时间",
        "建链结束时间",
        "linkend",
        "linkendtime",
        "linket",
        "end",
    ),
    "trace_st": (
        "开始跟踪时间",
        "跟踪开始时间",
        "tracestart",
        "tracestarttime",
        "tracest",
    ),
    "trace_et": (
        "结束跟踪时间",
        "跟踪结束时间",
        "traceend",
        "traceendtime",
        "traceet",
    ),
    "priority": ("优先级", "priority"),
}


def _clean_token(value):  # type: (str) -> str
    value = value.strip().lstrip("\ufeff")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def _normalise_header(value):  # type: (str) -> str
    value = _clean_token(value).lower()
    return re.sub(r"[\s_\-:/\\()（）\[\]{}]+", "", value)


_NORMALISED_ALIASES = {
    field: set(_normalise_header(alias) for alias in aliases)
    for field, aliases in _HEADER_ALIASES.items()
}


def _decode_csv(path, encoding):  # type: (str, Optional[str]) -> Tuple[str, str]
    with open(path, "rb") as stream:
        raw = stream.read()

    candidates = [encoding] if encoding else ["utf-8-sig", "gb18030", "gbk"]
    failures = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return raw.decode(candidate, errors="strict"), candidate
        except (LookupError, UnicodeDecodeError) as exc:
            failures.append("{}: {}".format(candidate, exc))
    raise UnicodeError(
        "Cannot decode CSV {!r}; attempted {}".format(path, "; ".join(failures))
    )


def _parse_int(value, row_number, field):  # type: (str, int, str) -> int
    cleaned = _clean_token(value)
    try:
        return int(cleaned)
    except ValueError:
        try:
            numeric = float(cleaned)
        except ValueError:
            raise ValueError(
                "Row {} has invalid {} value {!r}".format(row_number, field, value)
            )
        if not numeric.is_integer():
            raise ValueError(
                "Row {} has non-integral {} value {!r}".format(
                    row_number, field, value
                )
            )
        return int(numeric)


def _parse_priority(value):  # type: (str) -> float
    cleaned = _clean_token(value)
    if not cleaned or cleaned == "*":
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _looks_like_data_row(row):  # type: (Sequence[str]) -> bool
    if len(row) < 6:
        return False
    try:
        for index in (2, 3, 4, 5):
            int(_clean_token(row[index]))
        return True
    except (ValueError, TypeError):
        return False


def _column_map(header):  # type: (Sequence[str]) -> Dict[str, int]
    normalised = [_normalise_header(item) for item in header]
    result = {}  # type: Dict[str, int]
    for field, aliases in _NORMALISED_ALIASES.items():
        for index, token in enumerate(normalised):
            if token in aliases:
                result[field] = index
                break

    required = ("ground", "satellite", "link_st", "link_et", "trace_st", "trace_et")
    if not all(field in result for field in required):
        # The old source refers to mojibake header literals.  Their semantics
        # cannot be decoded reliably, but C3/W3 have a fixed positional prefix.
        result.update(
            {
                "ground": 0,
                "satellite": 1,
                "link_st": 2,
                "link_et": 3,
                "trace_st": 4,
                "trace_et": 5,
            }
        )
    if "priority" not in result and len(header) > 11:
        result["priority"] = 11
    return result


def load_arcs(path, encoding=None):  # type: (str, Optional[str]) -> ArcDataset
    """Load a C3/W3-style CSV using dense, first-occurrence resource IDs.

    Parameters
    ----------
    path:
        Input CSV path.
    encoding:
        Optional forced encoding.  By default strict UTF-8 is attempted first,
        followed by GB18030/GBK.
    """

    source_path = os.path.abspath(os.fspath(path))
    text, detected_encoding = _decode_csv(source_path, encoding)
    rows = [
        list(row)
        for row in csv.reader(io.StringIO(text, newline=""))
        if row and any(_clean_token(value) for value in row)
    ]
    if not rows:
        raise ValueError("Input CSV {!r} is empty".format(source_path))

    has_header = not _looks_like_data_row(rows[0])
    header = tuple(_clean_token(value) for value in rows[0]) if has_header else tuple()
    columns = _column_map(header if has_header else ("",) * max(12, len(rows[0])))
    data_rows = rows[1:] if has_header else rows
    first_data_line = 2 if has_header else 1

    ground_to_id = {}  # type: Dict[str, int]
    satellite_to_id = {}  # type: Dict[str, int]
    ground_names = []  # type: List[str]
    satellite_names = []  # type: List[str]
    arcs = []  # type: List[Arc]

    required_max = max(columns[field] for field in (
        "ground", "satellite", "link_st", "link_et", "trace_st", "trace_et"
    ))
    for offset, row in enumerate(data_rows):
        row_number = first_data_line + offset
        if len(row) <= required_max:
            raise ValueError(
                "Row {} has {} columns; at least {} are required".format(
                    row_number, len(row), required_max + 1
                )
            )
        ground_name = _clean_token(row[columns["ground"]])
        satellite_name = _clean_token(row[columns["satellite"]])
        if not ground_name or not satellite_name:
            raise ValueError("Row {} has an empty resource identifier".format(row_number))

        if ground_name not in ground_to_id:
            ground_to_id[ground_name] = len(ground_names)
            ground_names.append(ground_name)
        if satellite_name not in satellite_to_id:
            satellite_to_id[satellite_name] = len(satellite_names)
            satellite_names.append(satellite_name)

        link_st = _parse_int(row[columns["link_st"]], row_number, "link_st")
        link_et = _parse_int(row[columns["link_et"]], row_number, "link_et")
        trace_st = _parse_int(row[columns["trace_st"]], row_number, "trace_st")
        trace_et = _parse_int(row[columns["trace_et"]], row_number, "trace_et")
        if link_et < link_st:
            raise ValueError(
                "Row {} has link_et < link_st ({} < {})".format(
                    row_number, link_et, link_st
                )
            )

        priority_index = columns.get("priority")
        priority = (
            _parse_priority(row[priority_index])
            if priority_index is not None and priority_index < len(row)
            else 0.0
        )
        arcs.append(
            Arc(
                id=len(arcs),
                ground=ground_to_id[ground_name],
                satellite=satellite_to_id[satellite_name],
                ground_name=ground_name,
                satellite_name=satellite_name,
                link_st=link_st,
                link_et=link_et,
                trace_st=trace_st,
                trace_et=trace_et,
                priority=priority,
            )
        )

    return ArcDataset(
        arcs=tuple(arcs),
        ground_names=tuple(ground_names),
        satellite_names=tuple(satellite_names),
        source_path=source_path,
        encoding=detected_encoding,
        header=header,
    )


# A readable alias for callers that treat loading as parsing rather than I/O.
read_arcs = load_arcs

