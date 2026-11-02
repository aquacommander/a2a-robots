"""In-memory robot for tests and examples. Moves in timed steps so stop,
cancel and timeouts can be exercised without hardware."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from .backend import HandState, MotionOutcome, RobotBackend
from .skills import FailureCode, Pose

_IDENTITY = (1.0, 0.0, 0.0, 0.0)


class FakeBackend(RobotBackend):
    def __init__(
        self,
        *,
        hands: Iterable[str] = ("left", "right"),
        frame: str = "world",
        steps: int = 10,
        step_time_s: float = 0.01,
        reachable_objects: Iterable[str] = (),
        slipping: Iterable[str] = (),
    ) -> None:
        """``reachable_objects`` names hands with an object to grasp;
        ``slipping`` names hands whose grasp slips."""
        names = tuple(hands)
        self._hands = frozenset(names)
        self._steps = steps
        self._step_time_s = step_time_s
        self._reachable = frozenset(reachable_objects)
        self._slipping = frozenset(slipping)
        self._lock = threading.Lock()
        self._state = {
            hand: HandState(Pose(frame, (0.3, 0.2 * (i - 0.5), 0.1), _IDENTITY))
            for i, hand in enumerate(names)
        }
        self.hold_calls = 0
        self.shut_down = False

    @property
    def hands(self) -> frozenset[str]:
        return self._hands

    def observe(self) -> dict[str, HandState]:
        with self._lock:
            return {h: HandState(s.pose, dict(s.contact), s.held) for h, s in self._state.items()}

    def move_to(self, hand: str, target: Pose, stop: threading.Event) -> MotionOutcome:
        start = self._state[hand].pose
        for i in range(1, self._steps + 1):
            if not self._step(stop):
                return MotionOutcome(FailureCode.STOPPED, "stopped before reaching the target")
            t = i / self._steps
            position = tuple(
                a + (b - a) * t for a, b in zip(start.position, target.position, strict=True)
            )
            orientation = start.orientation if t < 1 else target.orientation
            with self._lock:
                self._state[hand].pose = Pose(target.frame, position, orientation)
        return MotionOutcome()

    def open(self, hand: str, stop: threading.Event) -> MotionOutcome:
        for _ in range(self._steps):
            if not self._step(stop):
                return MotionOutcome(FailureCode.STOPPED, "stopped while opening")
        with self._lock:
            self._state[hand].held = False
            self._state[hand].contact = {}
        return MotionOutcome()

    def close(self, hand: str, stop: threading.Event) -> MotionOutcome:
        for _ in range(self._steps):
            if not self._step(stop):
                return MotionOutcome(FailureCode.STOPPED, "stopped while closing")
        with self._lock:
            state = self._state[hand]
            if hand in self._reachable and hand not in self._slipping:
                state.held = True
                state.contact = {"thumb": 1.0, "index": 1.0}
                return MotionOutcome()
            state.held = False
            state.contact = {}
        if hand in self._slipping:
            return MotionOutcome(FailureCode.SLIPPED, "object slipped during the grasp")
        return MotionOutcome()

    def hold(self) -> None:
        with self._lock:
            self.hold_calls += 1

    def shutdown(self) -> None:
        self.shut_down = True

    def _step(self, stop: threading.Event) -> bool:
        """Wait one control step; False when stop was requested."""
        return not stop.wait(self._step_time_s)
