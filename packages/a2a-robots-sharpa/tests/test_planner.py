import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from a2a_robots import FailureCode, Pose
from a2a_robots.planning import JointTrajectory, PlanError, Planner
from a2a_robots_sharpa import Chain, MujocoPlanner, SharpaMujocoBackend, load_model

MINI = Path(__file__).with_name("mini_dual_hand.xml")
IDENTITY = (1.0, 0.0, 0.0, 0.0)
BASE = tuple(f"right_{j}_joint" for j in ("x", "y", "z", "roll", "pitch", "yaw"))
OBSTACLE = (0.03, 0.15, 0.05)  # in front of the right hand's palm and fingers
GRASPED = (0.035, 0.115)


def model(block=None):
    spec = mujoco.MjSpec.from_file(str(MINI))
    if block is not None:
        geom = spec.worldbody.add_geom()
        geom.name = "block"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.005, 0.03, 0.015]
        geom.pos = list(block)
    return load_model(spec)


def position(b: SharpaMujocoBackend, hand: str = "right") -> np.ndarray:
    return np.array(b.observe()[hand].pose.position)


def start(m: mujoco.MjModel) -> dict[str, float]:
    return {name: 0.0 for name in BASE}


def test_plan_is_speed_limited_and_ends_at_the_target():
    m = model()
    planner = MujocoPlanner(m, {"right": Chain("right_hand_C_MC", BASE)}, max_speed_m_s=0.05)
    trajectory = planner.plan("right", start(m), Pose("world", (0.02, 0.15, 0.0), IDENTITY), 0.02)

    assert trajectory.joints == BASE
    assert trajectory.dt == 0.02
    xs = [0.0] + [point[0] for point in trajectory.points]
    assert xs[-1] == pytest.approx(0.02, abs=1e-4)
    assert max(abs(b - a) for a, b in zip(xs, xs[1:], strict=False)) / 0.02 <= 0.05 * 1.01
    assert len(trajectory.points) * 0.02 >= 0.02 / 0.05


def test_path_through_an_obstacle_is_refused_before_anything_moves():
    b = SharpaMujocoBackend(model(OBSTACLE), realtime=False)
    before = position(b)

    outcome = b.move_to("right", Pose("world", (0.05, 0.15, 0.0), IDENTITY), threading.Event())

    assert outcome.failure_code is FailureCode.UNREACHABLE
    assert outcome.detail == "the path collides with block"
    assert position(b) == pytest.approx(before)


def test_the_other_hand_is_an_obstacle():
    b = SharpaMujocoBackend(model(), realtime=False)
    outcome = b.move_to("left", Pose("world", (0.0, 0.12, 0.0), IDENTITY), threading.Event())
    assert outcome.failure_code is FailureCode.UNREACHABLE
    assert "right_palm" in outcome.detail


def test_a_clear_path_beside_an_obstacle_is_allowed():
    b = SharpaMujocoBackend(model(OBSTACLE), realtime=False)
    outcome = b.move_to("right", Pose("world", (0.0, 0.15, 0.01), IDENTITY), threading.Event())
    assert outcome.failure_code is None


def test_an_object_the_hand_already_touches_is_not_an_obstacle():
    b = SharpaMujocoBackend(model((GRASPED[0], 0.15, GRASPED[1])), realtime=False)
    assert b.close("right", threading.Event()).failure_code is None
    assert b.observe()["right"].held
    planner = MujocoPlanner(b._model, {"right": Chain("right_hand_C_MC", BASE)})
    planner.set_scene(b._data.qpos)
    current = {name: float(b._data.joint(name).qpos[0]) for name in BASE}

    trajectory = planner.plan("right", current, Pose("world", (0.0, 0.15, 0.005), IDENTITY), 0.02)

    assert trajectory.points


def test_move_streams_the_trajectory_in_real_time():
    b = SharpaMujocoBackend(model(), realtime=True)
    t0 = time.monotonic()
    outcome = b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), threading.Event())
    assert outcome.failure_code is None
    # 1 cm at 5 cm/s with minimum-jerk timing takes at least 0.375 s.
    assert time.monotonic() - t0 >= 0.37


class Scripted(Planner):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def plan(self, hand, start, target, dt):
        self.calls.append((hand, dict(start), target, dt))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_a_custom_planner_is_used_and_its_trajectory_is_executed():
    points = tuple((0.002 * i, 0.0, 0.0, 0.0, 0.0, 0.0) for i in range(1, 6))
    planner = Scripted(JointTrajectory(BASE, points, 0.02))
    b = SharpaMujocoBackend(model(), realtime=False, planner=planner)

    outcome = b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), threading.Event())

    assert outcome.failure_code is None
    hand, current, _, dt = planner.calls[0]
    assert hand == "right" and dt == pytest.approx(0.02)
    assert current["right_y_joint"] == pytest.approx(0.0)
    assert position(b) == pytest.approx((0.01, 0.15, 0.0), abs=0.002)


def test_a_plan_error_becomes_the_skill_outcome():
    planner = Scripted(PlanError(FailureCode.UNREACHABLE, "no path"))
    b = SharpaMujocoBackend(model(), realtime=False, planner=planner)
    outcome = b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), threading.Event())
    assert (outcome.failure_code, outcome.detail) == (FailureCode.UNREACHABLE, "no path")


def test_a_trajectory_for_other_joints_is_rejected():
    planner = Scripted(JointTrajectory(("other",), ((0.0,),), 0.02))
    b = SharpaMujocoBackend(model(), realtime=False, planner=planner)
    with pytest.raises(ValueError, match="planner returned joints"):
        b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), threading.Event())
