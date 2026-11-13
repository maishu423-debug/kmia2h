import pytest

from helpers import load_cfg


@pytest.fixture
def cfg():
    return load_cfg()
