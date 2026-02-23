import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import (
    Tensor,
)

from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.linear_soft_max import LinearSoftMax


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_linear_softmax_policy_init_with_discrete_action_space(env_id: str):
    """
    Test that LinearSoftMax can be initialized with discrete action space environments.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.

    """
    env = gym.make(env_id)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, feature_fn)
    assert isinstance(policy, LinearSoftMax)
    assert policy._env.spec.id == env_id


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(observation=st.integers(min_value=0, max_value=47), seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_linear_softmax_policy_generate_action(env_id: str, observation: int, seed: int):
    """
    Test the generate_action function of the LinearSoftMax policy.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        observation: A valid observation (state) from the environment.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, feature_fn)
    action = policy.generate_action(np.array([observation]))

    assert isinstance(action, np.ndarray)
    assert action.size == 1
    assert action.flat[0] in [0, 1, 2, 3]


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(n_observations=st.integers(min_value=2, max_value=2), seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_linear_softmax_policy_calculate_log_probabilities(env_id: str, n_observations: int, seed: int):
    """
    Test the generate_action function of the LinearSoftMax policy.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        n_observations: The number of observations for which to calculate the log-probabilities.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    observations = np.array([np.random.randint(0, 47) for _ in range(n_observations)])[np.newaxis, :]
    actions = np.array([policy.generate_action(observations[0, i]) for i in range(n_observations)])

    log_probs = policy.calculate_log_probabilities(observations, actions)

    assert isinstance(log_probs, Tensor)
    assert log_probs.shape == (n_observations,)
    np.testing.assert_array_less(log_probs.detach().numpy(), 0)


# TODO: Add tests check the probabilities are as expected.
# TODO: Add tests check that the values of the log-probabilities are correct.
# TODO: Add tests with end-to-end sampling to ensure consistency with expected sizes.
