# a2a-robots

Expose a robot as an [A2A](https://a2a-protocol.org) agent. An agent plans the work and sends one task-level skill at a time; the adapter validates each request, enforces safety limits, runs the skill on the robot backend, and returns a fixed result format.

Built on [a2a-adapter](https://github.com/hybroai/a2a-adapter). Early development: the core and a fake robot are here; MuJoCo and Sharpa backends come next.

## Skills

Requests are JSON, sent as a text part or a data part:

```json
{"skill": "move_to", "args": {"hand": "right", "pose": {
  "frame": "world", "units": "m",
  "position": [0.30, -0.20, 0.15],
  "orientation": {"quaternion": [1, 0, 0, 0]}}}}
```

| Skill | Args | Notes |
| --- | --- | --- |
| `observe` | none | Read only. Always allowed, even during motion. |
| `move_to` | `hand`, `pose` | `frame`, `units` (`"m"`) and a unit quaternion are required. |
| `open` | `hand` | |
| `close` | `hand` | `held` reports whether something is now held. |
| `stop` | none | Stops motion now and freezes every hand. Always allowed. |

Every skill returns the same result, as JSON text and as an `application/json` data part:

```json
{"skill": "close", "success": false, "failure_code": "slipped",
 "detail": "object slipped during the grasp",
 "final_pose": {"right": {...}}, "contact": {"right": {}}, "held": {"right": false}}
```

`failure_code` is one of `step_too_large`, `out_of_workspace`, `busy`, `stopped`, `timeout`, `slipped`, `unreachable`. A skill that ran but did not succeed is a completed A2A task with `success: false`, so the agent can observe and retry. Only a malformed request fails the A2A task, and nothing moves.

## Safety

- A move larger than `max_step_m` or `max_step_rad`, or a target outside the workspace box, is refused before anything moves. The result includes the limits. Moves are never clamped.
- One skill runs at a time per robot. A second motion request fails at once with `busy` instead of queueing.
- `stop`, `tasks/cancel`, timeouts and shutdown all stop the backend. Backends run in a worker thread and poll a stop flag every control step, so stopping does not depend on cancelling a Python task.
- Only the target pose is checked, not the path to it. That is fine in simulation; real hardware also needs the robot's own path and collision checks.
- `serve_robot()` binds `127.0.0.1` by default, because anyone who can reach the server can move the robot.

## Quick start

```bash
uv sync --extra dev
uv run python examples/fake_robot.py   # serves on http://127.0.0.1:9100/
```

```bash
curl -s http://127.0.0.1:9100/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "1",
    "parts": [{"kind": "data", "data": {"skill": "observe"}}]}}}'
```

## Adding a robot

This repository is a monorepo. The shared package stays `a2a-robots`; each real robot is a sibling package named `a2a-robots-<name>` (for example `a2a-robots-sharpa`) that depends on it. `FakeBackend` stays in the core package only as a dependency-free test double.

A robot package subclasses `RobotBackend` and implements `hands`, `observe`, `move_to`, `open`, `close`, `hold` and `shutdown`. Motion methods are synchronous, run in a worker thread, and must check the `stop` event at every control step and return `MotionOutcome(FailureCode.STOPPED)` once it is set. See `src/a2a_robots/fake.py`.

## Development

```bash
uv sync --extra dev
uv run ruff format --check . && uv run ruff check .
uv run pytest
```

## License

Apache-2.0
