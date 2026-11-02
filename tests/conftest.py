import pytest
from helpers import make_limits

from a2a_robots import FakeBackend, RobotController


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend(reachable_objects={"right"})


@pytest.fixture
def controller(backend: FakeBackend) -> RobotController:
    return RobotController(backend, make_limits())
