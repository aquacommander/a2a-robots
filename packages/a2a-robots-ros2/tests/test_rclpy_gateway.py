"""RclpyGateway against a live ROS 2 graph: servers and a publisher run on
their own node in this process. Skipped without ROS 2 unless
A2A_ROBOTS_REQUIRE_ROS is set, as in the Jazzy CI job."""

import asyncio
import json
import os
import threading
import time
import uuid

import pytest

if os.environ.get("A2A_ROBOTS_REQUIRE_ROS"):
    import rclpy
else:
    rclpy = pytest.importorskip("rclpy")

from a2a_adapter import CancelledByAdapterError  # noqa: E402
from example_interfaces.action import Fibonacci  # noqa: E402
from example_interfaces.msg import Int64  # noqa: E402
from example_interfaces.srv import AddTwoInts  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from a2a_robots_ros2 import RequestError, Ros2BridgeAdapter, RosError  # noqa: E402
from a2a_robots_ros2.rclpy_gateway import RclpyGateway  # noqa: E402


class Robot:
    """Fibonacci action (rejects order < 0; ``step_s`` between steps),
    AddTwoInts service and an Int64 topic, under a unique namespace."""

    def __init__(self):
        self.ns = f"/t{uuid.uuid4().hex[:8]}"
        self.step_s = 0.02
        self.cancelled = threading.Event()
        self.context = rclpy.Context()
        rclpy.init(context=self.context)
        self.node = Node("robot", namespace=self.ns, context=self.context)
        group = ReentrantCallbackGroup()
        ActionServer(
            self.node,
            Fibonacci,
            "fibonacci",
            self._execute,
            goal_callback=lambda g: GoalResponse.REJECT if g.order < 0 else GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.node.create_service(
            AddTwoInts,
            "add",
            lambda req, res: setattr(res, "sum", req.a + req.b) or res,
            callback_group=group,
        )
        self.publisher = self.node.create_publisher(Int64, "count", 10)
        self.executor = MultiThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _execute(self, goal):
        sequence = [0, 1]
        for _ in range(goal.request.order):
            if goal.is_cancel_requested:
                self.cancelled.set()
                goal.canceled()
                return Fibonacci.Result(sequence=sequence)
            sequence.append(sequence[-1] + sequence[-2])
            goal.publish_feedback(Fibonacci.Feedback(sequence=sequence))
            time.sleep(self.step_s)
        goal.succeed()
        return Fibonacci.Result(sequence=sequence)

    def skills(self, **timeouts):
        def entry(id, kind, name, type):
            spec = {"id": id, "kind": kind, "ros_name": f"{self.ns}/{name}", "type": type}
            if id in timeouts:
                spec["timeout_s"] = timeouts[id]
            return spec

        return [
            entry("fib", "action", "fibonacci", "example_interfaces/action/Fibonacci"),
            entry("add", "service", "add", "example_interfaces/srv/AddTwoInts"),
            entry("count", "topic", "count", "example_interfaces/msg/Int64"),
        ]

    def close(self):
        self.executor.shutdown()
        self.thread.join()
        self.node.destroy_node()
        self.context.try_shutdown()


@pytest.fixture
def robot():
    robot = Robot()
    yield robot
    robot.close()


@pytest.fixture
async def bridge(robot):
    adapters = []

    def make(skills=None, **kwargs):
        adapter = Ros2BridgeAdapter(skills or robot.skills(), RclpyGateway(), **kwargs)
        adapters.append(adapter)
        return adapter

    yield make
    for adapter in adapters:
        await adapter.close()


def ctx(task_id="t1"):
    class Context:
        message = None

    context = Context()
    context.task_id = task_id
    return context


def request(skill, **args):
    return json.dumps({"skill": skill, "args": args})


async def test_an_action_streams_accepted_feedback_and_the_result(bridge):
    adapter = bridge(max_feedback_hz=None)
    lines = [
        json.loads(line) async for line in adapter.stream(request("fib", order=4), context=ctx())
    ]
    assert lines[0] == {"skill": "fib", "status": "accepted"}
    assert [line["feedback"]["sequence"][-1] for line in lines[1:-1]] == [1, 2, 3, 5]
    assert lines[-1] == {
        "skill": "fib",
        "status": "succeeded",
        "result": {"sequence": [0, 1, 1, 2, 3, 5]},
    }


async def test_cancel_reaches_the_action_server(bridge, robot):
    robot.step_s = 0.1
    adapter = bridge(stream_feedback=False)
    run = asyncio.create_task(adapter.invoke(request("fib", order=50), context=ctx()))
    await asyncio.sleep(0.5)  # the goal is running

    await adapter.cancel(context=ctx())
    with pytest.raises(CancelledByAdapterError):
        await asyncio.wait_for(run, 5)
    assert robot.cancelled.wait(5)


async def test_a_timed_out_goal_is_cancelled_on_the_server(bridge, robot):
    robot.step_s = 0.1
    adapter = bridge(robot.skills(fib=0.5), stream_feedback=False)
    with pytest.raises(RosError, match="timed out after 0.5 s; the goal was cancelled"):
        await adapter.invoke(request("fib", order=50), context=ctx())
    assert robot.cancelled.wait(5)


async def test_a_rejected_goal_fails(bridge):
    with pytest.raises(RosError, match="rejected the goal"):
        await bridge(stream_feedback=False).invoke(request("fib", order=-1), context=ctx())


async def test_a_service_call_returns_the_response(bridge):
    result = await bridge(stream_feedback=False).invoke(request("add", a=2, b=3), context=ctx())
    assert json.loads(result[0].text) == {"skill": "add", "response": {"sum": 5}}


async def test_args_that_do_not_fit_the_message_are_a_request_error(bridge):
    adapter = bridge(stream_feedback=False)
    with pytest.raises(RequestError, match="args do not fit example_interfaces/srv/AddTwoInts"):
        await adapter.invoke(request("add", c=1), context=ctx())
    with pytest.raises(RequestError, match="args do not fit"):
        await adapter.invoke(request("fib", order="many"), context=ctx())


async def test_a_topic_read_returns_the_latest_message(bridge, robot):
    adapter = bridge(stream_feedback=False)
    read = asyncio.create_task(adapter.invoke(request("count"), context=ctx()))
    for n in range(1, 100):  # until the subscription is matched and has a message
        robot.publisher.publish(Int64(data=n))
        await asyncio.sleep(0.02)
        if read.done():
            break
    result = json.loads((await read)[0].text)
    assert result["skill"] == "count" and result["message"]["data"] >= 1
    assert 0 <= result["age_s"] < 1


async def test_a_missing_server_times_out_naming_it(bridge, robot):
    skills = robot.skills(fib=0.3)
    skills[0]["ros_name"] = f"{robot.ns}/nobody"
    with pytest.raises(RosError, match=f"no goal was accepted.*{robot.ns}/nobody"):
        await bridge(skills, stream_feedback=False).invoke(request("fib", order=1), context=ctx())


def test_an_unknown_interface_type_fails_at_startup():
    skills = [{"id": "x", "kind": "topic", "ros_name": "/x", "type": "no_such_pkg/msg/X"}]
    with pytest.raises(ValueError, match="x: cannot load no_such_pkg/msg/X"):
        Ros2BridgeAdapter(skills, RclpyGateway())


async def test_close_stops_the_executor_thread(robot):
    gateway = RclpyGateway()
    Ros2BridgeAdapter(robot.skills(), gateway)
    thread = gateway._thread
    assert thread.is_alive()
    await gateway.close()
    assert not thread.is_alive()
    await gateway.close()  # a second close is a no-op
