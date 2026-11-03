"""The planning layer between a skill and joint-level control.

A planner turns a pose target into a joint trajectory: one position target per
control step for the joints that carry the hand (a floating base in simulation,
an arm on a real robot). The backend streams those targets to the robot and
checks the stop flag between them. Collision checking belongs to the planner,
because only it knows the scene.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .skills import FailureCode, Pose


@dataclass(frozen=True, slots=True)
class JointTrajectory:
    """Position targets for ``joints``, one per control step of ``dt`` seconds.

    The last point is the goal.
    """

    joints: tuple[str, ...]
    points: tuple[tuple[float, ...], ...]
    dt: float


class PlanError(Exception):
    """No safe trajectory exists. Becomes a completed skill with ``success: false``."""

    def __init__(self, failure_code: FailureCode, detail: str) -> None:
        super().__init__(detail)
        self.failure_code = failure_code
        self.detail = detail


class Planner(ABC):
    @abstractmethod
    def plan(
        self, hand: str, start: Mapping[str, float], target: Pose, dt: float
    ) -> JointTrajectory:
        """Plan from the joint positions ``start`` to a configuration that puts
        ``hand`` at ``target``. Raise ``PlanError`` when there is none."""


def interpolate(
    start: Sequence[float], goal: Sequence[float], max_speeds: Sequence[float], dt: float
) -> list[tuple[float, ...]]:
    """Points from ``start`` (excluded) to ``goal`` (included), one per ``dt``.

    Uses a smooth (minimum-jerk) time scaling that starts and ends at rest; the
    duration is chosen so no joint exceeds its entry in ``max_speeds``.
    """
    # Minimum-jerk peak speed is 1.875 times the average speed.
    duration = max(
        (1.875 * abs(b - a) / v for a, b, v in zip(start, goal, max_speeds, strict=True)),
        default=0.0,
    )
    steps = max(1, math.ceil(duration / dt - 1e-9))
    points = []
    for i in range(1, steps + 1):
        s = i / steps
        blend = s**3 * (10 - 15 * s + 6 * s**2)
        points.append(tuple(a + (b - a) * blend for a, b in zip(start, goal, strict=True)))
    return points
