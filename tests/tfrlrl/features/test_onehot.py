import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import construct_one_hot_feature_function


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_returns_callable(env_id: str):
    """
    Test that construct_one_hot_feature_function returns a callable function.

    :param env_id: The Gymnasium environment ID to be used.
    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = construct_one_hot_feature_function(S, A)
    assert callable(feature_fn)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_feature_function_returns_numpy_array(env_id: str):
    """
    Test that the feature function returns a numpy array for valid observations.

    :param env_id: The Gymnasium environment ID to be used.
    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = construct_one_hot_feature_function(S, A)

    # Test with the initial observation
    observation, _ = env.reset()
    features = feature_fn(observation)

    assert isinstance(features, np.ndarray)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(observation=st.integers(min_value=0, max_value=47))
@settings(deadline=None)
def test_feature_function_output_shape(env_id: str, observation: int):
    """
    Test that the feature function returns the correct output shape.

    :param env_id: The Gymnasium environment ID to be used.
    :param observation: A valid observation (state) from the environment.
    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = construct_one_hot_feature_function(S, A)
    features = feature_fn(observation)

    # The feature function should return a matrix of shape (A, S * (A - 1))
    expected_shape = (A, S * (A - 1))
    assert features.shape == expected_shape


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_feature_function_one_hot_encoding(env_id: str):
    """
    Test that the feature function produces correct one-hot encoded features.

    :param env_id: The Gymnasium environment ID to be used.
    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = construct_one_hot_feature_function(S, A)

    # Test a specific observation
    observation = 0
    features = feature_fn(observation)

    # Check that each row (except possibly the first action) has exactly one non-zero element
    # The first action is excluded to avoid linear dependency
    for i in range(1, A):
        row_sum = np.sum(features[i, :])
        assert row_sum == 1.0, f'Row {i} should sum to 1.0, got {row_sum}'
        assert np.sum(features[i, :] == 1.0) == 1, f'Row {i} should have exactly one 1.0'


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(observation=st.integers(min_value=0, max_value=47))
@settings(deadline=None)
def test_feature_function_consistent_output(env_id: str, observation: int):
    """
    Test that the feature function produces consistent output for the same observation.

    :param env_id: The Gymnasium environment ID to be used.
    :param observation: A valid observation (state) from the environment.
    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = construct_one_hot_feature_function(S, A)

    # Call the feature function multiple times with the same observation
    features1 = feature_fn(observation)
    features2 = feature_fn(observation)

    # Results should be identical
    np.testing.assert_array_equal(features1, features2)
