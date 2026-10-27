# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Load a canonical action/state layout and scatter native groups into it.

Slot widths come from the layout YAML. Robot YAML files say which native
group fills which slice. Unwritten slots stay zero and are masked out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


@dataclass(frozen=True)
class FieldSpan:
    """One named slot and its absolute range in the canonical vector."""

    name: str
    start: int
    dim: int


@dataclass(frozen=True)
class Placement:
    """Copy ``length`` dims of a native group into ``dest`` of the vector."""

    source: str
    dest: int
    length: int


@dataclass(frozen=True)
class CanonicalLayout:
    dim: int
    fields: dict[str, FieldSpan]


@dataclass(frozen=True)
class RobotCanonicalMap:
    robot_type: str
    action: tuple[Placement, ...]
    state: tuple[Placement, ...]

    def sources(self, kind: str) -> list[str]:
        placements = self.action if kind == "action" else self.state
        ordered: list[str] = []
        for placement in placements:
            if placement.source not in ordered:
                ordered.append(placement.source)
        return ordered


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path from the cwd, then from the repo root."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    repo_candidate = Path(__file__).resolve().parents[2] / candidate
    if repo_candidate.is_file():
        return repo_candidate
    raise FileNotFoundError(f"Canonical config does not exist: {path}")


def _load_mapping(path: str | Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(resolve_config_path(path)), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Canonical config must be a mapping: {path}")
    return loaded


def _as_block_list(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return list(value)
    raise ValueError(f"{label} must be a mapping or a list of mappings")


def parse_layout(data: dict[str, Any]) -> CanonicalLayout:
    """Build absolute field spans. Raises if the widths do not sum to ``dim``."""
    dim = int(data["dim"])
    arms = [str(arm) for arm in data["arms"]]
    arm_fields = _as_block_list(data["arm_fields"], "arm_fields")
    shared_fields = _as_block_list(data["shared"], "shared")
    fields: dict[str, FieldSpan] = {}
    cursor = 0

    def _add(name: str, width: int) -> None:
        nonlocal cursor
        if name in fields:
            raise ValueError(f"Duplicate canonical field {name!r}")
        if width <= 0:
            raise ValueError(f"Canonical field {name!r} must have a positive dim")
        fields[name] = FieldSpan(name=name, start=cursor, dim=width)
        cursor += width

    for arm in arms:
        for field in arm_fields:
            _add(f"{arm}.{field['name']}", int(field["dim"]))
    for field in shared_fields:
        _add(str(field["name"]), int(field["dim"]))
    if cursor != dim:
        raise ValueError(f"Canonical fields sum to {cursor}, layout dim is {dim}")
    return CanonicalLayout(dim=dim, fields=fields)


def load_layout(path: str | Path) -> CanonicalLayout:
    return parse_layout(_load_mapping(path))


def _placements(layout: CanonicalLayout, entries: list[dict[str, Any]], kind: str) -> tuple[Placement, ...]:
    occupied = np.zeros(layout.dim, dtype=bool)
    placements: list[Placement] = []
    for entry in entries:
        field_name = str(entry["field"])
        if field_name not in layout.fields:
            raise ValueError(f"Unknown canonical field {field_name!r} in {kind} map")
        field = layout.fields[field_name]
        start = int(entry.get("start", 0))
        if "length" in entry and entry["length"] is not None:
            length = int(entry["length"])
        else:
            length = field.dim - start
        if start < 0 or length <= 0 or start + length > field.dim:
            raise ValueError(
                f"{kind} field {field_name!r} slice start={start} length={length} "
                f"does not fit in dim {field.dim}"
            )
        dest = field.start + start
        if bool(occupied[dest : dest + length].any()):
            raise ValueError(
                f"{kind} writes overlap canonical dims {dest}:{dest + length} "
                f"via source {entry['source']!r}"
            )
        occupied[dest : dest + length] = True
        placements.append(Placement(source=str(entry["source"]), dest=dest, length=length))
    return tuple(placements)


def parse_robot_map(data: dict[str, Any], layout: CanonicalLayout) -> RobotCanonicalMap:
    robot_type = str(data["robot_type"])
    action_entries = list(data.get("action") or [])
    state_entries = list(data.get("state") or [])
    return RobotCanonicalMap(
        robot_type=robot_type,
        action=_placements(layout, action_entries, "action"),
        state=_placements(layout, state_entries, "state"),
    )


def load_robot_maps(path: str | Path, layout: CanonicalLayout) -> dict[str, RobotCanonicalMap]:
    """Load every YAML robot map in a file or directory, keyed by robot_type."""
    raw = Path(path)
    if raw.exists():
        resolved = raw
    else:
        resolved = Path(__file__).resolve().parents[2] / raw
    if not resolved.exists():
        raise FileNotFoundError(f"Canonical robot map path does not exist: {path}")
    files = [resolved] if resolved.is_file() else sorted(resolved.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"No robot map YAML files in {resolved}")
    maps: dict[str, RobotCanonicalMap] = {}
    for file_path in files:
        robot_map = parse_robot_map(_load_mapping(file_path), layout)
        if robot_map.robot_type in maps:
            raise ValueError(f"Duplicate canonical robot_type {robot_map.robot_type!r}")
        maps[robot_map.robot_type] = robot_map
    return maps


def scatter(
    groups: dict[str, np.ndarray],
    placements: tuple[Placement, ...] | list[Placement],
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter native groups into a ``(T, dim)`` vector and a matching mask."""
    if not placements:
        raise ValueError("Canonical scatter received no placements")
    first = np.asarray(groups[placements[0].source], dtype=np.float32)
    time_steps = 1 if first.ndim == 1 else int(first.shape[0])
    vector = np.zeros((time_steps, dim), dtype=np.float32)
    mask = np.zeros((dim,), dtype=np.float32)
    for placement in placements:
        if placement.source not in groups:
            raise KeyError(
                f"Canonical source {placement.source!r} is missing from {sorted(groups)}"
            )
        value = np.asarray(groups[placement.source], dtype=np.float32)
        if value.ndim == 1:
            value = value[None, :]
        if value.shape[0] != time_steps:
            raise ValueError(
                f"Source {placement.source!r} has {value.shape[0]} steps, expected {time_steps}"
            )
        if value.shape[-1] < placement.length:
            raise ValueError(
                f"Source {placement.source!r} has width {value.shape[-1]}, "
                f"placement needs {placement.length}"
            )
        vector[:, placement.dest : placement.dest + placement.length] = value[..., : placement.length]
        mask[placement.dest : placement.dest + placement.length] = 1.0
    tiled = np.broadcast_to(mask, vector.shape).copy()
    return vector, tiled


def gather(vector: np.ndarray, placements: tuple[Placement, ...] | list[Placement]) -> dict[str, np.ndarray]:
    """Slice a canonical vector back into native groups. Padded slots are dropped."""
    array = np.asarray(vector)
    grouped: dict[str, np.ndarray] = {}
    for placement in placements:
        if placement.source in grouped:
            raise ValueError(f"Canonical source {placement.source!r} is written more than once")
        grouped[placement.source] = array[..., placement.dest : placement.dest + placement.length]
    return grouped
