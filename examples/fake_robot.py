"""Serve the in-memory fake robot on http://127.0.0.1:9100/."""

from a2a_robots import FakeBackend, RobotAdapter, SafetyLimits, Workspace, serve_robot

limits = SafetyLimits(
    workspace=Workspace("world", (0.0, -0.5, 0.0), (0.6, 0.5, 0.5)),
    max_step_m=0.05,
    max_step_rad=0.1,
)
adapter = RobotAdapter(
    FakeBackend(reachable_objects={"right"}),
    limits,
    name="Fake robot",
    description="Two simulated hands for trying a2a-robots without hardware.",
)

if __name__ == "__main__":
    serve_robot(adapter, port=9100)
