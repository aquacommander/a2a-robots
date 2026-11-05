"""The allowlist: which ROS 2 actions, services and topics the bridge exposes.

Each entry maps one A2A skill to one ROS interface. Nothing outside the list
can be reached through the bridge.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Kind(StrEnum):
    ACTION = "action"
    SERVICE = "service"
    TOPIC = "topic"  # read the latest message


_TYPE_DIR = {Kind.ACTION: "action", Kind.SERVICE: "srv", Kind.TOPIC: "msg"}
_DEFAULT_TIMEOUT_S = {Kind.ACTION: 60.0, Kind.SERVICE: 10.0, Kind.TOPIC: 10.0}
_FIELDS = {"id", "kind", "ros_name", "type", "description", "timeout_s", "exclusive"}

_ID = re.compile(r"[A-Za-z0-9_.-]+")
# Absolute names only, so the result never depends on the node's namespace.
_ROS_NAME = re.compile(r"(/[A-Za-z_][A-Za-z0-9_]*)+")
_TYPE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)/(action|srv|msg)/([A-Z][A-Za-z0-9]*)")


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """One allowlisted ROS interface. Built by ``parse_skills``.

    ``timeout_s`` bounds the whole call, including waiting for the server.
    ``exclusive`` refuses a new call to this skill while one is still running.
    """

    id: str
    kind: Kind
    ros_name: str
    type: str
    timeout_s: float
    description: str = ""
    exclusive: bool = False


def parse_skills(entries: Iterable[Mapping[str, Any]]) -> tuple[SkillSpec, ...]:
    """Validate the allowlist. Raises ``ValueError`` naming the bad field."""
    skills: list[SkillSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        spec = _parse_entry(entry, f"skills[{i}]")
        if spec.id in seen:
            raise ValueError(f"skills[{i}].id: duplicate skill id {spec.id!r}")
        seen.add(spec.id)
        skills.append(spec)
    if not skills:
        raise ValueError("skills: list at least one ROS interface to expose")
    return tuple(skills)


def _parse_entry(entry: Mapping[str, Any], where: str) -> SkillSpec:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: must be an object")
    unknown = sorted(set(entry) - _FIELDS)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {', '.join(unknown)}")

    skill_id = entry.get("id")
    if not isinstance(skill_id, str) or not _ID.fullmatch(skill_id):
        raise ValueError(f"{where}.id: letters, digits, '_', '.' or '-' only")

    try:
        kind = Kind(entry.get("kind"))
    except (TypeError, ValueError):
        raise ValueError(f"{where}.kind: one of action, service, topic") from None

    ros_name = entry.get("ros_name")
    if not isinstance(ros_name, str) or not _ROS_NAME.fullmatch(ros_name):
        raise ValueError(f"{where}.ros_name: an absolute ROS name such as /robot/arm/move")

    type_name = entry.get("type")
    match = _TYPE.fullmatch(type_name) if isinstance(type_name, str) else None
    if match is None or match.group(2) != _TYPE_DIR[kind]:
        raise ValueError(f"{where}.type: a {kind} type such as pkg/{_TYPE_DIR[kind]}/Name")

    description = entry.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"{where}.description: must be a string")

    timeout_s = entry.get("timeout_s", _DEFAULT_TIMEOUT_S[kind])
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, int | float)
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError(f"{where}.timeout_s: a positive number of seconds")

    exclusive = entry.get("exclusive", False)
    if not isinstance(exclusive, bool):
        raise ValueError(f"{where}.exclusive: true or false")

    return SkillSpec(skill_id, kind, ros_name, type_name, float(timeout_s), description, exclusive)
