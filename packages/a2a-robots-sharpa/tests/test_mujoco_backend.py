import asyncio
import os
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from a2a_robots import FailureCode, Pose, RobotController, SafetyLimits, Workspace
from a2a_robots.skills import parse_request
from a2a_robots_sharpa import SharpaMujocoBackend, load_model

MINI = Path(__file__).with_name("mini_dual_hand.xml")
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def backend(*, block=None, **kwargs) -> SharpaMujocoBackend:
    """The mini model, optionally with a fixed block at ``block`` = (x, z) by the right hand."""
    spec = mujoco.MjSpec.from_file(str(MINI))
    if block is not None:
        geom = spec.worldbody.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.005, 0.03, 0.015]
        geom.pos = [block[0], 0.15, block[1]]
    return SharpaMujocoBackend(load_model(spec), **{"realtime": False, **kwargs})


def position(b: SharpaMujocoBackend, hand: str = "right") -> np.ndarray:
    return np.array(b.observe()[hand].pose.position)


def turned(angle: float) -> tuple[float, ...]:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), angle)
    return tuple(quat)


GRASPED = (0.035, 0.115)  # thumb and index both reach the block
THUMB_ONLY = (0.03, 0.1)


def test_observe_reports_each_palm_in_the_world_frame():
    states = backend().observe()
    assert set(states) == {"left", "right"}
    assert states["right"].pose.frame == "world"
    assert states["right"].pose.position == pytest.approx((0.0, 0.15, 0.0))
    assert states["left"].pose.position == pytest.approx((0.0, -0.15, 0.0))
    assert states["right"].pose.orientation == pytest.approx(IDENTITY)
    assert states["right"].contact == {}
    assert states["right"].held is False


def test_move_to_reaches_the_target_and_leaves_the_other_hand():
    b = backend()
    left = position(b, "left")
    target = Pose("world", (0.01, 0.15, 0.01), turned(0.05))

    outcome = b.move_to("right", target, threading.Event())

    assert outcome.failure_code is None
    state = b.observe()["right"]
    assert state.pose.position == pytest.approx(target.position, abs=0.002)
    assert state.pose.orientation == pytest.approx(target.orientation, abs=0.01)
    assert position(b, "left") == pytest.approx(left, abs=1e-4)


def test_base_does_not_sag_under_gravity():
    b = backend()
    b.move_to("right", Pose("world", (0.0, 0.15, 0.01), IDENTITY), threading.Event())
    assert position(b)[2] == pytest.approx(0.01, abs=0.002)


def test_target_outside_the_base_joint_range_is_unreachable():
    b = backend()
    before = position(b)
    outcome = b.move_to("right", Pose("world", (2.5, 0.15, 0.0), IDENTITY), threading.Event())
    assert outcome.failure_code is FailureCode.UNREACHABLE
    assert position(b) == pytest.approx(before)


def test_stop_set_before_moving_leaves_the_hand_in_place():
    b = backend()
    stop = threading.Event()
    stop.set()
    outcome = b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), stop)
    assert outcome.failure_code is FailureCode.STOPPED
    assert position(b) == pytest.approx((0.0, 0.15, 0.0), abs=1e-4)


def test_stop_interrupts_a_realtime_move_part_way():
    b = backend(realtime=True)
    stop = threading.Event()
    threading.Timer(0.03, stop.set).start()

    outcome = b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), stop)

    assert outcome.failure_code is FailureCode.STOPPED
    assert 0.0 < position(b)[0] < 0.009


def test_realtime_paces_the_simulation_to_the_wall_clock():
    b = backend(realtime=True)
    start = time.monotonic()
    b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), threading.Event())
    # A move that settles returns right after its last step, without waiting it out.
    assert time.monotonic() - start >= b._data.time - b._control_dt


def test_close_on_an_object_holds_it_and_open_releases_it():
    b = backend(block=GRASPED)
    stop = threading.Event()

    assert b.close("right", stop).failure_code is None
    state = b.observe()["right"]
    assert state.held is True
    assert {"thumb", "index"} <= set(state.contact)

    assert b.open("right", stop).failure_code is None
    state = b.observe()["right"]
    assert state.held is False
    assert state.contact == {}


def test_close_on_nothing_succeeds_without_holding():
    b = backend()
    assert b.close("right", threading.Event()).failure_code is None
    assert b.observe()["right"].held is False


def test_close_that_touches_but_cannot_hold_is_a_slip():
    b = backend(block=THUMB_ONLY)
    outcome = b.close("right", threading.Event())
    assert outcome.failure_code is FailureCode.SLIPPED
    assert b.observe()["right"].held is False


def test_close_bends_flexion_joints_and_keeps_abduction_straight():
    b = backend()
    b.close("right", threading.Event())
    qpos = {name: b._data.joint(f"right_{name}").qpos[0] for name in ("index_MCP_AA", "index_PIP")}
    assert qpos["index_MCP_AA"] == pytest.approx(0.0, abs=0.01)
    assert qpos["index_PIP"] == pytest.approx(0.96, abs=0.05)


def test_hold_freezes_every_actuator_at_its_current_position():
    b = backend()
    b.move_to("right", Pose("world", (0.01, 0.15, 0.0), IDENTITY), threading.Event())
    b._data.ctrl[:] = 1.0
    b.hold()
    for i in range(b._model.nu):
        joint = b._model.actuator_trnid[i, 0]
        assert b._data.ctrl[i] == pytest.approx(b._data.qpos[b._model.jnt_qposadr[joint]])


def test_a_model_without_the_expected_names_is_rejected():
    with pytest.raises(ValueError, match="no joint named 'top_x_joint'"):
        SharpaMujocoBackend(load_model(MINI), hands=("top",))


async def test_controller_refuses_an_oversized_move_and_runs_a_grasp():
    b = backend(block=GRASPED)
    limits = SafetyLimits(Workspace("world", (-0.3, -0.5, -0.3), (0.3, 0.5, 0.3)))
    controller = RobotController(b, limits)
    hands = b.hands

    def req(skill, **args):
        return parse_request({"skill": skill, "args": args}, hands)

    far = {"frame": "world", "units": "m", "position": [0.1, 0.15, 0.0],
           "orientation": {"quaternion": list(IDENTITY)}}  # fmt: skip
    refused = await controller.run(req("move_to", hand="right", pose=far), "t1")
    assert refused.failure_code is FailureCode.STEP_TOO_LARGE
    assert position(b) == pytest.approx((0.0, 0.15, 0.0), abs=1e-4)

    grasp = await controller.run(req("close", hand="right"), "t2")
    assert grasp.success and grasp.held["right"] is True
    await asyncio.wait_for(controller.aclose(), 1)


# The real Sharpa Wave model. CI downloads it; locally run
#   uv run python -m a2a_robots_sharpa.model <dir>
# and set SHARPA_MODEL_DIR=<dir>.
REAL_MODEL_DIR = os.environ.get("SHARPA_MODEL_DIR")


@pytest.mark.skipif(not REAL_MODEL_DIR, reason="SHARPA_MODEL_DIR is not set")
def test_real_sharpa_model_moves_and_grasps():
    from a2a_robots_sharpa import fetch_model

    spec = mujoco.MjSpec.from_file(str(fetch_model(REAL_MODEL_DIR)))
    can = spec.worldbody.add_geom()
    can.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    can.size = [0.02, 0.04, 0]
    can.pos = [0.05, 0.18, 0.11]
    can.quat = [0.7071068, 0.7071068, 0, 0]
    b = SharpaMujocoBackend(load_model(spec), realtime=False)
    stop = threading.Event()

    assert b.observe()["right"].pose.position == pytest.approx((0.0, 0.15, 0.0), abs=1e-4)
    target = Pose("world", (0.01, 0.15, 0.01), IDENTITY)
    assert b.move_to("right", target, stop).failure_code is None
    assert position(b) == pytest.approx(target.position, abs=0.002)
    b.move_to("right", Pose("world", (0.0, 0.15, 0.0), IDENTITY), stop)

    assert b.close("right", stop).failure_code is None
    assert b.observe()["right"].held is True
    assert b.open("right", stop).failure_code is None
    assert b.observe()["right"].held is False
