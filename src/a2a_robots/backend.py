"""The interface a robot implements to be driven by ``RobotAdapter``.

Backend methods are synchronous and run in a worker thread, because simulator
steps and vendor SDK calls block. Cancelling the asyncio task cannot interrupt
them, so every motion method must poll ``stop`` at each control step and return
promptly once it is set, leaving the hand where it is.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .skills import FailureCode, Pose


@dataclass(slots=True)
class HandState:
    pose: Pose
    contact: dict[str, float] = field(default_factory=dict)
    held: bool = False


@dataclass(frozen=True, slots=True)
class MotionOutcome:
    """What a motion call did. ``failure_code`` is None on success."""

    failure_code: FailureCode | None = None
    detail: str | None = None


class RobotBackend(ABC):
    @property
    @abstractmethod
    def hands(self) -> frozenset[str]: ...

    @abstractmethod
    def observe(self) -> dict[str, HandState]:
        """Current state of every hand. Must be safe to call during motion."""

    @abstractmethod
    def move_to(self, hand: str, target: Pose, stop: threading.Event) -> MotionOutcome: ...

    @abstractmethod
    def open(self, hand: str, stop: threading.Event) -> MotionOutcome: ...

    @abstractmethod
    def close(self, hand: str, stop: threading.Event) -> MotionOutcome: ...

    @abstractmethod
    def hold(self) -> None:
        """Freeze every hand at its current pose."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release the robot or simulator. Called once, after motion has stopped."""
