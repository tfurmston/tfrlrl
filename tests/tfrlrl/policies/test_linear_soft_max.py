import gymnasium as gym
import numpy as np
import pytest

from tfrlrl.features.onehot import construct_one_hot_feature_function
from tfrlrl.policies.linear_soft_max import LinearSoftMax


@pytest.mark.parametrize('env_id', ['CliffWalking-v0'])
def test_init_with_discrete_action_space(env_id: str):
    """
    Test that LinearSoftMax can be initialized with discrete action space environments.

    :param env_id: The Gymnasium environment ID with a discrete action space.
    """
    env = gym.make(env_id)
    softmax_parameters = np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(env.observation_space.n * (env.action_space.n - 1), ),
    )
    feature_fn = construct_one_hot_feature_function(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, softmax_parameters, feature_fn)
    assert isinstance(policy, LinearSoftMax)
    assert policy._env.spec.id == env_id
