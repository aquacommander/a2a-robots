import pytest

from a2a_robots_ros2 import Kind, parse_skills

ROTATE = {
    "id": "rotate",
    "kind": "action",
    "ros_name": "/turtle1/rotate_absolute",
    "type": "turtlesim/action/RotateAbsolute",
}


def test_entries_parse_with_per_kind_default_timeouts():
    skills = parse_skills(
        [
            {**ROTATE, "description": "Turn the turtle."},
            {"id": "spawn", "kind": "service", "ros_name": "/spawn", "type": "turtlesim/srv/Spawn"},
            {
                "id": "pose",
                "kind": "topic",
                "ros_name": "/turtle1/pose",
                "type": "turtlesim/msg/Pose",
                "timeout_s": 2,
                "exclusive": True,
            },
        ]
    )
    rotate, spawn, pose = skills
    assert rotate.kind is Kind.ACTION and rotate.timeout_s == 60.0
    assert rotate.description == "Turn the turtle."
    assert spawn.kind is Kind.SERVICE and spawn.timeout_s == 10.0
    assert pose.timeout_s == 2.0 and pose.exclusive is True
    assert not rotate.exclusive


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"id": ""}, "id"),
        ({"id": "has space"}, "id"),
        ({"kind": "publisher"}, "kind"),
        ({"kind": ["action"]}, "kind"),
        ({"ros_name": "turtle1/rotate_absolute"}, "ros_name"),
        ({"ros_name": "/turtle1/"}, "ros_name"),
        ({"type": "turtlesim/srv/Spawn"}, "type"),
        ({"type": "RotateAbsolute"}, "type"),
        ({"description": 3}, "description"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"timeout_s": True}, "timeout_s"),
        ({"timeout_s": float("inf")}, "timeout_s"),
        ({"exclusive": "yes"}, "exclusive"),
    ],
)
def test_bad_fields_are_rejected_by_name(change, field):
    with pytest.raises(ValueError, match=rf"^skills\[0\]\.{field}:"):
        parse_skills([{**ROTATE, **change}])


def test_missing_required_field_is_rejected():
    entry = dict(ROTATE)
    del entry["type"]
    with pytest.raises(ValueError, match=r"skills\[0\]\.type"):
        parse_skills([entry])


def test_unknown_fields_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unknown field.*publish"):
        parse_skills([{**ROTATE, "publish": True}])
    with pytest.raises(ValueError, match=r"skills\[1\]\.id: duplicate"):
        parse_skills([ROTATE, ROTATE])


def test_an_empty_allowlist_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        parse_skills([])
