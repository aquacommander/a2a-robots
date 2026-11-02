import asyncio

import pytest
from helpers import make_limits, pose

from a2a_robots import FailureCode, FakeBackend, RequestError, RobotController, Skill
from a2a_robots.skills import parse_request

HANDS = frozenset({"left", "right"})


def req(skill: str, **args):
    return parse_request({"skill": skill, "args": args}, HANDS)


def right_position(backend: FakeBackend):
    return backend.observe()["right"].pose.position


async def test_observe_reports_every_hand_in_the_result_schema(controller):
    result = (await controller.run(req("observe"), "t1")).to_dict()
    assert result["success"] is True
    assert result["failure_code"] is None
    assert set(result["final_pose"]) == {"left", "right"}
    assert result["final_pose"]["right"]["units"] == "m"
    assert result["held"] == {"left": False, "right": False}
    assert "limits" not in result


async def test_move_within_limits_reaches_the_target(controller, backend):
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.13)), "t1")
    assert result.success
    assert right_position(backend) == pytest.approx((0.3, 0.1, 0.13))


async def test_oversized_move_is_refused_before_anything_moves(controller, backend):
    before = right_position(backend)
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.3)), "t1")
    assert not result.success
    assert result.failure_code is FailureCode.STEP_TOO_LARGE
    assert result.limits == {"max_step_m": 0.05, "max_step_rad": 0.1}
    assert right_position(backend) == before


async def test_oversized_rotation_is_refused(controller):
    quarter_turn = (0.7071068, 0.7071068, 0, 0)
    result = await controller.run(
        req("move_to", hand="right", pose=pose(0.3, 0.1, 0.1, quarter_turn)), "t1"
    )
    assert result.failure_code is FailureCode.STEP_TOO_LARGE


async def test_target_outside_workspace_is_refused():
    controller = RobotController(FakeBackend(), make_limits(max_step_m=10))
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.9)), "t1")
    assert result.failure_code is FailureCode.OUT_OF_WORKSPACE


async def test_wrong_frame_is_a_request_error(controller):
    with pytest.raises(RequestError, match='frame must be "world"'):
        await controller.run(
            req("move_to", hand="right", pose=pose(0.3, 0.1, 0.1, frame="tool")), "t1"
        )


async def test_second_motion_fails_fast_with_busy_while_observe_still_works():
    backend = FakeBackend(steps=50, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    first = asyncio.create_task(controller.run(req("open", hand="left"), "t1"))
    await asyncio.sleep(0.05)

    busy = await asyncio.wait_for(controller.run(req("close", hand="right"), "t2"), 0.2)
    observed = await asyncio.wait_for(controller.run(req("observe"), "t3"), 0.2)

    assert busy.failure_code is FailureCode.BUSY
    assert observed.success
    assert (await first).success


async def test_stop_preempts_a_running_move_and_freezes_the_pose():
    backend = FakeBackend(steps=100, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(
        controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    )
    await asyncio.sleep(0.1)

    stopped = await asyncio.wait_for(controller.run(req("stop"), "t2"), 0.5)
    moved = await move

    assert stopped.success
    assert moved.failure_code is FailureCode.STOPPED
    assert 0.1 < right_position(backend)[2] < 0.14
    assert backend.hold_calls == 1
    # The robot is free again after a stop.
    assert (await controller.run(req("observe"), "t3")).success
    assert (await controller.run(req("open", hand="right"), "t4")).success


async def test_timeout_stops_the_hand_instead_of_leaving_it_moving():
    backend = FakeBackend(steps=1000, step_time_s=0.01)
    controller = RobotController(backend, make_limits(skill_timeout_s=0.1))
    result = await controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    z = right_position(backend)[2]
    await asyncio.sleep(0.05)

    assert result.failure_code is FailureCode.TIMEOUT
    assert right_position(backend)[2] == z
    assert backend.hold_calls == 1


async def test_cancelling_the_task_stops_the_backend_thread():
    backend = FakeBackend(steps=100, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(
        controller.run(req("move_to", hand="right", pose=pose(0.3, 0.1, 0.14)), "t1")
    )
    await asyncio.sleep(0.05)
    controller.cancel("t1")
    move.cancel()
    with pytest.raises(asyncio.CancelledError):
        await move
    z = right_position(backend)[2]
    await asyncio.sleep(0.05)

    assert right_position(backend)[2] == z
    assert backend.hold_calls == 1
    assert (await controller.run(req("open", hand="right"), "t2")).success


async def test_cancel_for_another_task_does_not_stop_the_motion():
    backend = FakeBackend(steps=20, step_time_s=0.01)
    controller = RobotController(backend, make_limits())
    move = asyncio.create_task(controller.run(req("open", hand="right"), "t1"))
    await asyncio.sleep(0.03)
    controller.cancel("other")
    assert (await move).success


async def test_grasp_reports_held_and_contact(controller):
    result = await controller.run(req("close", hand="right"), "t1")
    assert result.success
    assert result.held["right"] is True
    assert result.contact["right"]
    released = await controller.run(req("open", hand="right"), "t2")
    assert released.held["right"] is False


async def test_slip_is_a_completed_result_with_success_false():
    controller = RobotController(
        FakeBackend(reachable_objects={"right"}, slipping={"right"}), make_limits()
    )
    result = await controller.run(req("close", hand="right"), "t1")
    assert result.skill is Skill.CLOSE
    assert not result.success
    assert result.failure_code is FailureCode.SLIPPED
    assert result.held["right"] is False


async def test_aclose_stops_and_shuts_down_the_backend(controller, backend):
    await controller.aclose()
    assert backend.shut_down
    assert backend.hold_calls == 1
