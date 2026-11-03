import pytest

from a2a_robots.planning import interpolate


def speeds(points, start, dt):
    previous = start
    for point in points:
        yield [abs(b - a) / dt for a, b in zip(previous, point, strict=True)]
        previous = point


def test_interpolate_ends_at_the_goal_and_respects_each_joint_speed():
    start, goal, max_speeds, dt = (0.0, 1.0), (0.1, 0.5), (0.05, 0.5), 0.02
    points = interpolate(start, goal, max_speeds, dt)

    assert points[-1] == pytest.approx(goal)
    assert points[0] != pytest.approx(start)
    for step in speeds(points, start, dt):
        assert step[0] <= max_speeds[0] * 1.01
        assert step[1] <= max_speeds[1] * 1.01
    # Starts and ends slowly (minimum-jerk), so the first step is smaller than the middle one.
    assert abs(points[0][0]) < abs(points[len(points) // 2][0] - points[len(points) // 2 - 1][0])


def test_interpolate_with_no_motion_returns_the_goal_once():
    assert interpolate((0.2, 0.3), (0.2, 0.3), (1.0, 1.0), 0.02) == [(0.2, 0.3)]
