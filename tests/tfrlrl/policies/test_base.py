import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.policies.base import BasePolicy, PolicyException, UniformActionSamplingPolicy


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Acrobot-v1'])
def test_init_with_discrete_action_space(env_id: str):
    """
    Test that UniformActionSamplingPolicy can be initialized with discrete action space environments.

    :param env_id: The Gymnasium environment ID with a discrete action space.
    """
    policy = UniformActionSamplingPolicy(env_id)
    assert isinstance(policy, BasePolicy)
    assert policy._env.spec.id == env_id


@pytest.mark.parametrize('env_id', ['Ant-v4', 'HalfCheetah-v4'])
def test_init_with_continuous_action_space_raises_exception(env_id: str):
    """
    Test that UniformActionSamplingPolicy raises exception with continuous action space environments.

    :param env_id: The Gymnasium environment ID with a continuous action space.
    """
    with pytest.raises(PolicyException) as exc_info:
        UniformActionSamplingPolicy(env_id)
    assert 'discrete action spaces only' in str(exc_info.value)


@pytest.mark.parametrize('env_id', ['CartPole-v1'])
@given(n_actions=st.integers(min_value=10, max_value=100))
@settings(deadline=None)
def test_generate_action_returns_valid_action(env_id: str, n_actions: int):
    """
    Test that generate_action returns valid actions from the discrete action space.

    :param env_id: The Gymnasium environment ID to be used.
    :param n_actions: The number of actions to generate.
    """
    policy = UniformActionSamplingPolicy(env_id)
    dummy_observation = np.zeros(policy._env.observation_space.shape)

    for _ in range(n_actions):
        action = policy.generate_action(dummy_observation)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < policy._env.action_space.n
