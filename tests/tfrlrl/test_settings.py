import pytest

from tfrlrl.settings import is_valid_environment


@pytest.mark.parametrize('env_id, is_valid',
                         [
                             (
                                 'CartPole-v1-1234',
                                 False,
                             ),
                             (
                                 'CartPole-v1',
                                 True,
                             ),
                         ])
def test_is_valid_environment(env_id, is_valid):
    """Test that is_valid_environment returns False for invalid environments."""
    assert is_valid_environment(env_id) is is_valid
