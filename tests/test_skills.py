import pytest
from helpers import pose

from a2a_robots import RequestError, Skill
from a2a_robots.skills import parse_request

HANDS = frozenset({"left", "right"})


def test_parses_move_to_with_frame_units_and_orientation():
    request = parse_request(
        {"skill": "move_to", "args": {"hand": "right", "pose": pose(0.3, 0.1, 0.12)}}, HANDS
    )
    assert request.skill is Skill.MOVE_TO
    assert request.hand == "right"
    assert request.pose.position == (0.3, 0.1, 0.12)
    assert request.pose.orientation == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("skill", ["observe", "stop"])
def test_hand_free_skills_take_no_args(skill):
    assert parse_request({"skill": skill}, HANDS).skill == skill
    with pytest.raises(RequestError, match="unexpected args"):
        parse_request({"skill": skill, "args": {"hand": "right"}}, HANDS)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({"skill": "grasp"}, "skill must be one of"),
        ({"skill": "open", "args": {"hand": "middle"}}, "hand must be one of"),
        ({"skill": "open", "args": {"hand": "right"}, "extra": 1}, "unknown request fields"),
        ({"skill": "move_to", "args": {"hand": "right"}}, "pose must be an object"),
        (
            {"skill": "move_to", "args": {"hand": "right", "pose": {"frame": "world"}}},
            "pose is missing",
        ),
        (
            {
                "skill": "move_to",
                "args": {"hand": "right", "pose": {**pose(0, 0, 0), "units": "mm"}},
            },
            'units must be "m"',
        ),
        (
            {"skill": "move_to", "args": {"hand": "right", "pose": pose(0, 0, 0, (1, 1, 0, 0))}},
            "unit quaternion",
        ),
        (
            {
                "skill": "move_to",
                "args": {"hand": "right", "pose": {**pose(0, 0, 0), "position": [0, 0]}},
            },
            "3 finite numbers",
        ),
        (
            {
                "skill": "move_to",
                "args": {"hand": "right", "pose": {**pose(0, 0, 0), "position": [0, True, 0]}},
            },
            "3 finite numbers",
        ),
    ],
)
def test_rejects_malformed_requests(payload, message):
    with pytest.raises(RequestError, match=message):
        parse_request(payload, HANDS)
