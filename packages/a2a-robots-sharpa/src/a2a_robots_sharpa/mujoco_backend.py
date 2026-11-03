"""Sharpa Wave hands simulated in MuJoCo.

Works with ``dual_sharpa_wave.xml`` from sharpa-robotics/sharpa-urdf-usd-xml and
with any model that uses the same names: each hand ``<hand>`` has a floating
base driven by the position actuators ``<hand>_{x,y,z,roll,pitch,yaw}_joint_ctrl``,
a palm body ``<hand>_hand_C_MC``, and finger joints ``<hand>_<finger>_*``, each
with a ``<joint>_ctrl`` position actuator.

The hand pose is the palm body frame in world coordinates. ``move_to`` asks a
planner for a joint trajectory and streams it one target per control step, the
way a real arm takes joint targets from a host-side planner. The simulation only
advances while a motion runs, so the scene is frozen between skills.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from a2a_robots import FailureCode, HandState, MotionOutcome, Pose, RobotBackend
from a2a_robots.planning import PlanError, Planner

from .planner import Chain, MujocoPlanner

BASE_JOINTS = ("x", "y", "z", "roll", "pitch", "yaw")
PALM = "palm"


@dataclass(frozen=True, slots=True)
class _Hand:
    palm_body: int
    base_joints: tuple[str, ...]
    base_qpos: np.ndarray
    base_dofs: np.ndarray
    base_actuators: np.ndarray
    finger_qpos: np.ndarray
    finger_dofs: np.ndarray
    finger_actuators: np.ndarray
    open_targets: np.ndarray
    close_targets: np.ndarray
    geom_part: dict[int, str]


class SharpaMujocoBackend(RobotBackend):
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        hands: Iterable[str] = ("left", "right"),
        frame: str = "world",
        realtime: bool = True,
        control_dt: float = 0.02,
        settle_time_s: float = 3.0,
        position_tolerance_m: float = 0.002,
        orientation_tolerance_rad: float = 0.02,
        grasp_closure: float = 0.6,
        planner: Planner | None = None,
    ) -> None:
        """``realtime`` paces the simulation to wall-clock time; tests turn it off.
        ``grasp_closure`` is how far ``close`` bends the flexion joints, as a
        fraction of their range. ``planner`` defaults to a ``MujocoPlanner`` on
        the same model, which checks the path against the simulated scene."""
        self._model = model
        self._data = mujoco.MjData(model)
        self._frame = frame
        self._realtime = realtime
        self._substeps = max(1, round(control_dt / model.opt.timestep))
        self._control_dt = self._substeps * model.opt.timestep
        self._settle_steps = max(1, round(settle_time_s / self._control_dt))
        self._pos_tol = position_tolerance_m
        self._rot_tol = orientation_tolerance_rad
        self._hands = {name: self._index_hand(name, grasp_closure) for name in hands}
        self._planner = planner or MujocoPlanner(
            model,
            {name: Chain(f"{name}_hand_C_MC", h.base_joints) for name, h in self._hands.items()},
        )
        self._lock = threading.Lock()
        with self._lock:
            mujoco.mj_forward(self._model, self._data)
            self._hold_locked()

    @property
    def hands(self) -> frozenset[str]:
        return frozenset(self._hands)

    def observe(self) -> dict[str, HandState]:
        with self._lock:
            return {name: self._state(hand) for name, hand in self._hands.items()}

    def move_to(self, hand: str, target: Pose, stop: threading.Event) -> MotionOutcome:
        h = self._hands[hand]
        with self._lock:
            start = dict(zip(h.base_joints, self._data.qpos[h.base_qpos], strict=True))
            if isinstance(self._planner, MujocoPlanner):
                self._planner.set_scene(self._data.qpos)
            try:
                trajectory = self._planner.plan(hand, start, target, self._control_dt)
            except PlanError as error:
                return MotionOutcome(error.failure_code, error.detail)
        if trajectory.joints != h.base_joints:
            raise ValueError(f"planner returned joints {trajectory.joints}, not {h.base_joints}")
        points = iter(trajectory.points)

        def track() -> bool:
            # Stream the next target each control step, then wait for the hand to settle.
            point = next(points, None)
            if point is not None:
                self._data.ctrl[h.base_actuators] = point
                return False
            pos_err, rot_err = self._pose_error(h, target)
            speed = np.abs(self._data.qvel[h.base_dofs]).max()
            return pos_err <= self._pos_tol and rot_err <= self._rot_tol and speed < 1e-2

        if self._run(stop, track, extra_steps=len(trajectory.points)):
            return MotionOutcome()
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "stopped before reaching the target")
        return MotionOutcome(FailureCode.UNREACHABLE, "the hand did not settle at the target")

    def open(self, hand: str, stop: threading.Event) -> MotionOutcome:
        return self._move_fingers(self._hands[hand], self._hands[hand].open_targets, stop)

    def close(self, hand: str, stop: threading.Event) -> MotionOutcome:
        h = self._hands[hand]
        touched = False

        def watch() -> None:
            nonlocal touched
            touched = touched or bool(self._contact(h))

        outcome = self._move_fingers(h, h.close_targets, stop, watch)
        if outcome.failure_code is not None:
            return outcome
        with self._lock:
            held = self._held(h)
        if touched and not held:
            return MotionOutcome(FailureCode.SLIPPED, "object slipped during the grasp")
        return MotionOutcome()

    def hold(self) -> None:
        with self._lock:
            self._hold_locked()

    def shutdown(self) -> None:
        """Nothing to release: the model and data are freed with the backend."""

    # Motion

    def _move_fingers(self, h: _Hand, targets: np.ndarray, stop: threading.Event, watch=None):
        with self._lock:
            self._data.ctrl[h.finger_actuators] = targets
        best = np.inf
        stalled = 0

        def settled() -> bool:
            # Fingers stopped by an object or by each other never reach their
            # targets, so treat them as settled once the error stops shrinking.
            nonlocal best, stalled
            if watch is not None:
                watch()
            error = np.abs(self._data.qpos[h.finger_qpos] - targets).max(initial=0.0)
            if error < best - 5e-3:
                best, stalled = error, 0
            else:
                stalled += 1
            return error < 1e-2 or stalled * self._control_dt >= 0.2

        if self._run(stop, settled):
            return MotionOutcome()
        if stop.is_set():
            return MotionOutcome(FailureCode.STOPPED, "stopped while moving the fingers")
        return MotionOutcome(FailureCode.TIMEOUT, "the fingers did not settle")

    def _run(self, stop: threading.Event, done, extra_steps: int = 0) -> bool:
        """Step one control period at a time until ``done()`` or stop.

        ``done()`` runs after each step and may set the next targets. Returns True
        when it became true within ``extra_steps`` plus the settle time.
        """
        start = time.monotonic()
        for i in range(1, extra_steps + self._settle_steps + 1):
            if stop.is_set():
                return False
            with self._lock:
                mujoco.mj_step(self._model, self._data, nstep=self._substeps)
                if done():
                    return True
            if self._realtime:
                remaining = start + i * self._control_dt - time.monotonic()
                if remaining > 0 and stop.wait(remaining):
                    return False
        return False

    def _hold_locked(self) -> None:
        for h in self._hands.values():
            self._data.ctrl[h.base_actuators] = self._data.qpos[h.base_qpos]
            self._data.ctrl[h.finger_actuators] = self._data.qpos[h.finger_qpos]
            self._data.qvel[h.base_dofs] = 0.0
            self._data.qvel[h.finger_dofs] = 0.0

    # Kinematics

    def _pose_error(self, h: _Hand, target: Pose) -> tuple[float, float]:
        pos_err = np.linalg.norm(np.asarray(target.position) - self._data.xpos[h.palm_body])
        rot = np.zeros(3)
        mujoco.mju_subQuat(
            rot, np.asarray(target.orientation, dtype=float), self._data.xquat[h.palm_body]
        )
        return float(pos_err), float(np.linalg.norm(rot))

    # State

    def _state(self, h: _Hand) -> HandState:
        position = tuple(float(v) for v in self._data.xpos[h.palm_body])
        quat = self._data.xquat[h.palm_body]
        orientation = tuple(float(v) for v in (quat if quat[0] >= 0 else -quat))
        return HandState(Pose(self._frame, position, orientation), self._contact(h), self._held(h))

    def _contact(self, h: _Hand) -> dict[str, float]:
        """Normal force in newtons on each part of the hand from anything outside it."""
        forces: dict[str, float] = {}
        wrench = np.zeros(6)
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            part1 = h.geom_part.get(contact.geom1)
            part2 = h.geom_part.get(contact.geom2)
            if (part1 is None) == (part2 is None):
                continue
            mujoco.mj_contactForce(self._model, self._data, i, wrench)
            part = part1 or part2
            forces[part] = forces.get(part, 0.0) + float(abs(wrench[0]))
        return {part: round(force, 4) for part, force in forces.items() if force > 1e-6}

    def _held(self, h: _Hand) -> bool:
        contact = self._contact(h)
        return "thumb" in contact and any(p not in ("thumb", PALM) for p in contact)

    # Model indexing

    def _index_hand(self, name: str, grasp_closure: float) -> _Hand:
        m = self._model
        base_joints = [_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_{j}_joint") for j in BASE_JOINTS]
        palm = _id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}_hand_C_MC")
        base_set = set(base_joints)
        fingers = [
            j for j in range(m.njnt) if m.joint(j).name.startswith(f"{name}_") and j not in base_set
        ]
        if not fingers:
            raise ValueError(f"model has no finger joints for hand {name!r}")
        ranges = m.jnt_range[fingers]
        open_targets = np.clip(0.0, ranges[:, 0], ranges[:, 1])
        flexion = np.array(["_AA" not in m.joint(j).name for j in fingers])
        close_targets = np.where(
            flexion, ranges[:, 0] + grasp_closure * (ranges[:, 1] - ranges[:, 0]), open_targets
        )
        geom_part = {}
        for g in range(m.ngeom):
            body = m.body(m.geom_bodyid[g]).name
            if not body.startswith(f"{name}_"):
                continue
            part = body.split("_")[1]
            geom_part[g] = PALM if part in ("hand", "root") else part
        return _Hand(
            palm_body=palm,
            base_joints=tuple(m.joint(j).name for j in base_joints),
            base_qpos=m.jnt_qposadr[base_joints],
            base_dofs=m.jnt_dofadr[base_joints],
            base_actuators=np.array([_actuator(m, j) for j in base_joints]),
            finger_qpos=m.jnt_qposadr[fingers],
            finger_dofs=m.jnt_dofadr[fingers],
            finger_actuators=np.array([_actuator(m, j) for j in fingers]),
            open_targets=open_targets,
            close_targets=close_targets,
            geom_part=geom_part,
        )


def load_model(
    source: str | Path | mujoco.MjSpec, hands: Iterable[str] = ("left", "right")
) -> mujoco.MjModel:
    """Compile a model for ``SharpaMujocoBackend`` from an MJCF path, or from a
    spec the caller has edited (for example to add objects to the scene).

    The arm a hand is mounted on carries its weight, so the hand bodies get
    gravity compensation and the floating base does not sag. Objects in the
    scene still fall. MuJoCo only applies this when it is set before compiling.
    """
    spec = source if isinstance(source, mujoco.MjSpec) else mujoco.MjSpec.from_file(str(source))
    prefixes = tuple(f"{hand}_" for hand in hands)
    for body in spec.bodies:
        if body.name.startswith(prefixes):
            body.gravcomp = 1.0
    return spec.compile()


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"model has no {kind.name[6:].lower()} named {name!r}")
    return index


def _actuator(model: mujoco.MjModel, joint: int) -> int:
    return _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{model.joint(joint).name}_ctrl")
