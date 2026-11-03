"""Serve the simulated Sharpa Wave hands on http://127.0.0.1:9100/.

Downloads the model on first run. A can stands in front of the right hand, so
``close`` on the right hand grasps it.
"""

import mujoco

from a2a_robots import RobotAdapter, SafetyLimits, Workspace, serve_robot
from a2a_robots_sharpa import SharpaMujocoBackend, fetch_model, load_model

spec = mujoco.MjSpec.from_file(str(fetch_model()))
can = spec.worldbody.add_geom()
can.name = "can"
can.type = mujoco.mjtGeom.mjGEOM_CYLINDER
can.size = [0.02, 0.04, 0]
can.pos = [0.05, 0.18, 0.11]
can.quat = [0.7071068, 0.7071068, 0, 0]

limits = SafetyLimits(workspace=Workspace("world", (-0.3, -0.5, -0.3), (0.3, 0.5, 0.4)))
adapter = RobotAdapter(
    SharpaMujocoBackend(load_model(spec)),
    limits,
    name="Sharpa Wave (MuJoCo)",
    description="Two simulated Sharpa Wave hands on floating bases.",
)

if __name__ == "__main__":
    serve_robot(adapter, port=9100)
