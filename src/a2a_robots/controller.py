"""Runs skills on one robot: busy flag, preemption, timeouts, results.

One controller owns one backend, so every chat served by the process shares
the same busy flag and two chats cannot move the same robot at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import threading

from .backend import MotionOutcome, RobotBackend
from .limits import SafetyLimits
from .skills import FailureCode, Pose, RequestError, Skill, SkillRequest, SkillResult

logger = logging.getLogger(__name__)


class RobotController:
    def __init__(self, backend: RobotBackend, limits: SafetyLimits) -> None:
        self.backend = backend
        self.limits = limits
        self._stop = threading.Event()
        self._busy_task: str | None = None
        self._stopping = 0
        self._worker: asyncio.Future[MotionOutcome] | None = None
        # Target of each hand's last successful move. A backend stops within its
        # settle tolerance, so the next step is measured from here instead of
        # the observed pose. Cleared by anything that may leave the hand elsewhere.
        self._commanded: dict[str, Pose] = {}

    async def run(self, request: SkillRequest, task_id: str | None) -> SkillResult:
        if request.skill is Skill.OBSERVE:
            return await self._result(request.skill)
        if request.skill is Skill.STOP:
            await self.stop()
            return await self._result(request.skill)
        return await self._run_motion(request, task_id)

    async def stop(self) -> None:
        """Stop any motion now and freeze every hand where it is.

        No motion can start until the hands are frozen, including in the gap
        after a stopped motion releases the busy flag and before ``hold`` ends.
        """
        self._stopping += 1
        try:
            self._stop.set()
            await self._wait_for_worker()
            await asyncio.to_thread(self.backend.hold)
            self._commanded.clear()
        finally:
            self._stopping -= 1

    def cancel(self, task_id: str | None) -> None:
        """Stop the motion started by ``task_id``, if it is the one running."""
        # A run without a task id is stored as "", so None matches it here.
        if (task_id or "") == self._busy_task:
            self._stop.set()

    async def aclose(self) -> None:
        await self.stop()
        await asyncio.to_thread(self.backend.shutdown)

    async def _run_motion(self, request: SkillRequest, task_id: str | None) -> SkillResult:
        if request.pose is not None and request.pose.frame != self.limits.workspace.frame:
            raise RequestError(f'pose.frame must be "{self.limits.workspace.frame}"')
        # No await between this check and claiming the flag, so it is atomic on the loop.
        if self._stopping:
            return await self._result(request.skill, FailureCode.BUSY, "the robot is stopping")
        if self._busy_task is not None:
            return await self._result(
                request.skill, FailureCode.BUSY, "another skill is running on this robot"
            )
        self._busy_task = task_id or ""
        self._stop.clear()
        try:
            if request.skill is Skill.MOVE_TO:
                assert request.hand is not None and request.pose is not None
                states = await asyncio.to_thread(self.backend.observe)
                current = self._commanded.get(request.hand, states[request.hand].pose)
                refusal = self.limits.check_move(current, request.pose)
                if refusal is not None:
                    return await self._result(request.skill, *refusal, limits=True)
                call = functools.partial(
                    self.backend.move_to, request.hand, request.pose, self._stop
                )
            elif request.skill is Skill.OPEN:
                call = functools.partial(self.backend.open, request.hand, self._stop)
            else:
                call = functools.partial(self.backend.close, request.hand, self._stop)
            self._commanded.pop(request.hand, None)
            outcome = await self._run_worker(call)
            if (
                request.skill is Skill.MOVE_TO
                and outcome.failure_code is None
                and not self._stop.is_set()
            ):
                self._commanded[request.hand] = request.pose
            return await self._result(request.skill, outcome.failure_code, outcome.detail)
        finally:
            self._busy_task = None

    async def _run_worker(self, call: functools.partial[MotionOutcome]) -> MotionOutcome:
        self._worker = asyncio.ensure_future(asyncio.to_thread(call))
        try:
            return await asyncio.wait_for(asyncio.shield(self._worker), self.limits.skill_timeout_s)
        except TimeoutError:
            await self.stop()
            return MotionOutcome(
                FailureCode.TIMEOUT, f"skill exceeded {self.limits.skill_timeout_s} s"
            )
        except asyncio.CancelledError:
            # tasks/cancel cancels this coroutine, not the thread: stop the hand too.
            await asyncio.shield(self.stop())
            raise

    async def _wait_for_worker(self) -> None:
        worker = self._worker
        if worker is not None and not worker.done():
            with contextlib.suppress(Exception):
                await asyncio.shield(worker)

    async def _result(
        self,
        skill: Skill,
        failure_code: FailureCode | None = None,
        detail: str | None = None,
        *,
        limits: bool = False,
    ) -> SkillResult:
        states = await asyncio.to_thread(self.backend.observe)
        return SkillResult(
            skill=skill,
            success=failure_code is None,
            failure_code=failure_code,
            detail=detail,
            final_pose={hand: s.pose.to_dict() for hand, s in states.items()},
            contact={hand: dict(s.contact) for hand, s in states.items()},
            held={hand: s.held for hand, s in states.items()},
            limits=self.limits.as_dict() if limits else None,
        )
