"""A joint-space planner that uses a MuJoCo model of the robot and its scene.

For each hand it knows a chain: the joints that carry the hand and the body
whose pose ``move_to`` targets. It solves inverse kinematics, interpolates in
joint space, and rejects a path on which the hand would hit something. Objects
the hand already touches at the start (such as one it holds) are not obstacles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from a2a_robots import FailureCode, Pose
from a2a_robots.planning import JointTrajectory, PlanError, Planner, interpolate


@dataclass(frozen=True, slots=True)
class Chain:
    """``joints`` move ``body``; ``move_to`` targets the pose of ``body``.

    Six joints reach any nearby pose; a seven-joint arm works too.
    """

    body: str
    joints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Chain:
    joints: tuple[str, ...]
    body: int
    qpos: np.ndarray
    dofs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    max_speeds: np.ndarray
    geoms: frozenset[int]


class MujocoPlanner(Planner):
    def __init__(
        self,
        model: mujoco.MjModel,
        chains: Mapping[str, Chain],
        *,
        max_speed_m_s: float = 0.05,
        max_speed_rad_s: float = 0.5,
        penetration_m: float = 1e-3,
    ) -> None:
        """``max_speed_*`` cap slide and hinge joint speeds along the path.
        A contact deeper than ``penetration_m`` counts as a collision."""
        self._model = model
        self._data = mujoco.MjData(model)
        self._penetration = penetration_m
        self._chains = {
            hand: self._index(chain, max_speed_m_s, max_speed_rad_s)
            for hand, chain in chains.items()
        }

    def set_scene(self, qpos: Sequence[float]) -> None:
        """Copy the full simulator state the next plan checks collisions against."""
        self._data.qpos[:] = qpos

    def plan(
        self, hand: str, start: Mapping[str, float], target: Pose, dt: float
    ) -> JointTrajectory:
        chain = self._chains[hand]
        data = self._data
        q0 = np.array([start[name] for name in chain.joints])
        data.qpos[chain.qpos] = q0
        goal = self._solve(chain, target)
        if goal is None:
            raise PlanError(FailureCode.UNREACHABLE, "no joint solution reaches the target")
        points = interpolate(q0, goal, chain.max_speeds, dt)

        data.qpos[chain.qpos] = q0
        touching = self._contacts(chain, depth=0.0)
        for point in points:
            data.qpos[chain.qpos] = point
            hit = self._contacts(chain, depth=self._penetration) - touching
            if hit:
                names = ", ".join(sorted(self._name(g) for g in hit))
                raise PlanError(FailureCode.UNREACHABLE, f"the path collides with {names}")
        return JointTrajectory(chain.joints, tuple(points), dt)

    def _solve(self, chain: _Chain, target: Pose) -> np.ndarray | None:
        """Joint values that put the chain body at ``target`` (damped least squares)."""
        m, data = self._model, self._data
        goal_pos = np.asarray(target.position, dtype=float)
        goal_quat = np.asarray(target.orientation, dtype=float)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        error = np.zeros(6)
        for _ in range(100):
            mujoco.mj_kinematics(m, data)
            mujoco.mj_comPos(m, data)
            error[:3] = goal_pos - data.xpos[chain.body]
            mujoco.mju_subQuat(error[3:], goal_quat, data.xquat[chain.body])
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-5:
                return data.qpos[chain.qpos].copy()
            mujoco.mj_jacBody(m, data, jacp, jacr, chain.body)
            jac = np.vstack((jacp, jacr))[:, chain.dofs]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), error)
            data.qpos[chain.qpos] = np.clip(data.qpos[chain.qpos] + step, chain.lower, chain.upper)
        return None

    def _contacts(self, chain: _Chain, depth: float) -> set[int]:
        """Geoms outside the hand that it touches deeper than ``depth``."""
        mujoco.mj_fwdPosition(self._model, self._data)
        outside = set()
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 in chain.geoms) != (g2 in chain.geoms) and contact.dist <= -depth:
                outside.add(g2 if g1 in chain.geoms else g1)
        return outside

    def _name(self, geom: int) -> str:
        name = self._model.geom(geom).name
        return name or self._model.body(self._model.geom_bodyid[geom]).name

    def _index(self, chain: Chain, max_m: float, max_rad: float) -> _Chain:
        m = self._model
        joints = [_id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in chain.joints]
        body = _id(m, mujoco.mjtObj.mjOBJ_BODY, chain.body)
        # Every geom attached below the first chain joint moves with the hand.
        root = m.jnt_bodyid[joints[0]]
        geoms = set()
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            while b not in (root, 0):
                b = m.body_parentid[b]
            if b == root:
                geoms.add(g)
        slide = m.jnt_type[joints] == mujoco.mjtJoint.mjJNT_SLIDE
        limited = m.jnt_limited[joints].astype(bool)
        return _Chain(
            joints=tuple(chain.joints),
            body=body,
            qpos=m.jnt_qposadr[joints],
            dofs=m.jnt_dofadr[joints],
            lower=np.where(limited, m.jnt_range[joints, 0], -np.inf),
            upper=np.where(limited, m.jnt_range[joints, 1], np.inf),
            max_speeds=np.where(slide, max_m, max_rad),
            geoms=frozenset(geoms),
        )


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"model has no {kind.name[6:].lower()} named {name!r}")
    return index
