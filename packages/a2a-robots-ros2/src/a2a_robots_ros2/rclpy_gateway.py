"""The ``RosGateway`` for a live ROS 2 graph, built on rclpy.

rclpy comes with a ROS 2 install, so source its setup script before importing
this module. The gateway owns one node in its own rclpy context and spins it
on a background thread; results reach the event loop through
``call_soon_threadsafe``, so ROS callbacks never block it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from rosidl_runtime_py import message_to_ordereddict, set_message_fields
from rosidl_runtime_py.utilities import get_action, get_message, get_service

from .config import Kind, SkillSpec
from .gateway import GoalHandle, GoalResult, RequestError, RosError, RosGateway, TopicSample

_POLL_S = 0.05  # how often to check for a server or a first message
_LOADERS = {Kind.ACTION: get_action, Kind.SERVICE: get_service, Kind.TOPIC: get_message}
_STATUS = {
    GoalStatus.STATUS_SUCCEEDED: "succeeded",
    GoalStatus.STATUS_ABORTED: "aborted",
    GoalStatus.STATUS_CANCELED: "canceled",
}


class RclpyGateway(RosGateway):
    """Topics are read with the sensor-data QoS (best effort, keep last 5),
    which matches both reliable and best-effort publishers."""

    def __init__(self, node_name: str = "a2a_ros2_bridge") -> None:
        self._node_name = node_name
        self._actions: dict[str, tuple[ActionClient, Any]] = {}
        self._services: dict[str, tuple[Client, Any]] = {}
        self._latest: dict[str, tuple[Any, float]] = {}  # written on the executor thread
        self._context: rclpy.Context | None = None
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None

    def prepare(self, skills: Sequence[SkillSpec]) -> None:
        types = {}
        for spec in skills:
            try:
                types[spec.id] = _LOADERS[spec.kind](spec.type)
            except (AttributeError, ImportError, ValueError) as e:
                raise ValueError(
                    f"{spec.id}: cannot load {spec.type} ({e}); "
                    "is its package installed and sourced?"
                ) from None

        self._context = rclpy.Context()
        rclpy.init(context=self._context)
        self._node = node = Node(self._node_name, context=self._context)
        for spec in skills:
            interface = types[spec.id]
            if spec.kind is Kind.ACTION:
                self._actions[spec.id] = (ActionClient(node, interface, spec.ros_name), interface)
            elif spec.kind is Kind.SERVICE:
                client = node.create_client(interface, spec.ros_name)
                self._services[spec.id] = (client, interface)
            else:
                node.create_subscription(
                    interface,
                    spec.ros_name,
                    partial(self._on_message, spec.id),
                    qos_profile_sensor_data,
                )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(node)
        self._thread = threading.Thread(
            target=self._executor.spin, name=f"{self._node_name}-executor", daemon=True
        )
        self._thread.start()

    async def send_goal(
        self,
        skill: SkillSpec,
        goal: dict[str, Any],
        on_feedback: Callable[[dict[str, Any]], None],
    ) -> GoalHandle:
        client, action = self._actions[skill.id]
        message = _build(skill, action.Goal(), goal)
        await _wait_until(client.server_is_ready)
        loop = asyncio.get_running_loop()

        def feedback(feedback_message: Any) -> None:
            _call_soon(loop, on_feedback, _to_dict(feedback_message.feedback))

        sending = client.send_goal_async(message, feedback_callback=feedback)
        try:
            handle = await _on_loop(sending, loop)
        except asyncio.CancelledError:
            # The goal may still be accepted: cancel it then, so it does not run unseen.
            sending.add_done_callback(_cancel_if_accepted)
            raise
        if not handle.accepted:
            raise RosError(f"{skill.id}: the action server rejected the goal")
        return _RclpyGoal(handle)

    async def call_service(self, skill: SkillSpec, request: dict[str, Any]) -> dict[str, Any]:
        client, service = self._services[skill.id]
        message = _build(skill, service.Request(), request)
        await _wait_until(client.service_is_ready)
        future = client.call_async(message)
        try:
            response = await _on_loop(future, asyncio.get_running_loop())
        except asyncio.CancelledError:
            client.remove_pending_request(future)
            raise
        return _to_dict(response)

    async def read_topic(self, skill: SkillSpec) -> TopicSample:
        while (latest := self._latest.get(skill.id)) is None:
            await asyncio.sleep(_POLL_S)
        message, received = latest
        return TopicSample(_to_dict(message), time.monotonic() - received)

    async def close(self) -> None:
        if self._executor is None:
            return
        executor, self._executor = self._executor, None
        await asyncio.to_thread(self._shutdown, executor)

    def _on_message(self, skill_id: str, message: Any) -> None:
        self._latest[skill_id] = (message, time.monotonic())

    def _shutdown(self, executor: SingleThreadedExecutor) -> None:
        executor.shutdown()
        if self._thread is not None:
            self._thread.join()
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None:
            self._context.try_shutdown()


class _RclpyGoal(GoalHandle):
    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._result = handle.get_result_async()  # asked now, so the result is never missed

    async def result(self) -> GoalResult:
        response = await _on_loop(self._result, asyncio.get_running_loop())
        status = _STATUS.get(response.status)
        if status is None:
            raise RosError(f"the goal ended with unexpected status {response.status}")
        return GoalResult(status, _to_dict(response.result))

    async def cancel(self) -> None:
        response = await _on_loop(self._handle.cancel_goal_async(), asyncio.get_running_loop())
        # A goal that already finished, or that the server no longer knows, is not running.
        if response.return_code == CancelGoal.Response.ERROR_REJECTED:
            raise RosError("the action server refused to cancel the goal")


def _build(skill: SkillSpec, message: Any, fields: dict[str, Any]) -> Any:
    try:
        set_message_fields(message, fields)
    except (AttributeError, TypeError, ValueError) as e:
        raise RequestError(f"{skill.id}: args do not fit {skill.type}: {e}") from None
    return message


def _to_dict(message: Any) -> dict[str, Any]:
    return dict(message_to_ordereddict(message))


async def _wait_until(ready: Callable[[], bool]) -> None:
    # The adapter's timeout bounds this wait.
    while not ready():
        await asyncio.sleep(_POLL_S)


def _on_loop(future: Future, loop: asyncio.AbstractEventLoop) -> asyncio.Future[Any]:
    """An asyncio future that follows an rclpy future completed on the executor thread."""
    result = loop.create_future()

    def transfer() -> None:
        if result.done():  # the awaiting call was cancelled
            return
        if future.cancelled():
            result.cancel()
        elif (error := future.exception()) is not None:
            result.set_exception(error)
        else:
            result.set_result(future.result())

    future.add_done_callback(lambda _: _call_soon(loop, transfer))
    return result


def _call_soon(loop: asyncio.AbstractEventLoop, callback: Callable[..., None], *args: Any) -> None:
    try:
        loop.call_soon_threadsafe(callback, *args)
    except RuntimeError:  # the event loop is closed
        pass


def _cancel_if_accepted(sending: Future) -> None:
    if not sending.cancelled() and sending.exception() is None and sending.result().accepted:
        sending.result().cancel_goal_async()
