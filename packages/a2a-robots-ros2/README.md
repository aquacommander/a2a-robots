# a2a-robots-ros2

An allowlisted ROS 2 bridge for [a2a-robots](../../README.md): it exposes the ROS 2 actions, services and topics you list as A2A skills, for robot bring-up and debugging.

This is not the typed robot skill agent. `a2a_robots.RobotAdapter` gives planners one contract (`move_to`, `stop`, the busy rule, safety limits, one result shape) on any robot; this bridge passes configured ROS calls through and adds no safety limits of its own. Anything physical, such as joint limits or an emergency stop, stays in the robot's ROS stack.

This release has the bridge and the interface to ROS (`RosGateway`). The rclpy gateway that talks to a live ROS 2 graph comes next.

## Allowlist

Each entry maps one skill to one ROS interface. Nothing else can be reached through the bridge.

```python
from a2a_robots_ros2 import Ros2BridgeAdapter, serve_bridge

skills = [
    {
        "id": "rotate",
        "kind": "action",
        "ros_name": "/turtle1/rotate_absolute",
        "type": "turtlesim/action/RotateAbsolute",
        "description": "Turn the turtle to an absolute heading in radians.",
    },
    {"id": "spawn", "kind": "service", "ros_name": "/spawn", "type": "turtlesim/srv/Spawn"},
    {"id": "pose", "kind": "topic", "ros_name": "/turtle1/pose", "type": "turtlesim/msg/Pose"},
]
serve_bridge(Ros2BridgeAdapter(skills, gateway))  # http://127.0.0.1:9200/
```

| Field | Meaning |
| --- | --- |
| `id` | Skill name callers use. |
| `kind` | `action`, `service`, or `topic` (returns the latest message). |
| `ros_name` | Absolute ROS name. |
| `type` | Interface type, such as `turtlesim/action/RotateAbsolute`. |
| `description` | Optional. Shown in the Agent Card. |
| `timeout_s` | Optional. Bounds the whole call, including waiting for the server. Defaults: 60 s for actions, 10 s otherwise. |
| `exclusive` | Optional. Refuse a new call to this skill while one is still running. |

The allowlist is checked when the adapter is built, and the gateway loads every interface type then, so a typo or a missing message package fails at startup.

## Requests and results

Requests are JSON, sent as a text part or a data part. `args` holds the goal or request fields, with nested messages as nested objects; a topic skill takes none.

```json
{"skill": "rotate", "args": {"theta": 1.57}}
```

| Kind | Result |
| --- | --- |
| action | `{"skill": "rotate", "status": "succeeded", "result": {...}}`, where `status` is `succeeded`, `aborted` or `canceled` |
| service | `{"skill": "spawn", "response": {...}}` |
| topic | `{"skill": "pose", "message": {...}, "age_s": 0.04}`, where `age_s` is the time since the message arrived |

A goal that ran but did not succeed is a completed A2A task with its status, like a skill result with `success: false`. A malformed request, a rejected goal, a missing server or a timeout fails the task.

Feedback is streamed by default. The reply is then newline-delimited JSON in the response artifact: for an action an `accepted` line, then feedback lines (at most 2 per second; set `max_feedback_hz`; feedback that arrives before acceptance is held until the `accepted` line), then the result as the last line. Feedback appears in the response artifact, not as task status messages. With `stream_feedback=False` the reply is just the result, as JSON text and as an `application/json` data part.

## Cancel and timeouts

- `tasks/cancel` cancels the goal and waits for the action server's answer before the task is reported cancelled. A cancel that arrives while the goal is still being sent cancels it as soon as it is accepted. A timeout or shutdown cancels the goal too.
- If the server does not confirm the cancel within `cancel_timeout_s` (5 s by default), or the cancel fails, the bridge logs that the goal may still be running and tries again when the call stops and on shutdown. a2a-adapter still reports the task cancelled after a failed cancel, so check the robot.
- ROS 2 services cannot be cancelled. Cancelling a service task, or its timeout, stops the wait, but the call may still complete on the robot.
- `serve_bridge()` binds `127.0.0.1` by default, because anyone who can reach the server can move the robot.

## ROS 2

rclpy is not on PyPI. It comes with a ROS 2 install and is built for that distribution's Python, so it is not a dependency of this package. ROS 2 Jazzy (Python 3.12) and newer are supported; Humble (Python 3.10) is below the Python 3.11 minimum.

## Tests

The tests use a fake `RosGateway`, so they need no ROS install:

```bash
uv sync --all-packages --extra dev
uv run pytest packages/a2a-robots-ros2
```
