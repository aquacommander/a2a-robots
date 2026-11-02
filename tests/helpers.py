"""Shared test helpers."""

from a2a_robots import SafetyLimits, Workspace

WORKSPACE = Workspace("world", (0.0, -0.5, 0.0), (0.6, 0.5, 0.5))


def pose(x: float, y: float, z: float, quaternion=(1, 0, 0, 0), frame="world") -> dict:
    return {
        "frame": frame,
        "units": "m",
        "position": [x, y, z],
        "orientation": {"quaternion": list(quaternion)},
    }


def make_limits(**overrides) -> SafetyLimits:
    return SafetyLimits(
        workspace=WORKSPACE,
        **{"max_step_m": 0.05, "max_step_rad": 0.1, "skill_timeout_s": 5.0, **overrides},
    )
