from typing import List

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.base import EnvironmentException
from tfrlrl.features.onehot import OneHotFeatureFunction


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_returns_callable(env_id: str):
    """
    Test that OneHotFeatureFunction acts as a callable function.

    Args:
        env_id: The Gymnasium environment ID to be used.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = OneHotFeatureFunction(S, A)
    assert callable(feature_fn)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_feature_function_returns_numpy_array(env_id: str):
    """
    Test that the feature function returns a numpy array for valid observations.

    Args:
        env_id: The Gymnasium environment ID to be used.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = OneHotFeatureFunction(S, A)

    # Test with the initial observation
    observation, _ = env.reset()
    features = feature_fn(np.array([observation]))

    assert isinstance(features, np.ndarray)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(observation=st.integers(min_value=0, max_value=47))
@settings(deadline=None)
def test_feature_function_output_shape_with_single_observation(env_id: str, observation: int):
    """
    Test that the feature function returns the correct output shape in case of single observations.

    Args:
        env_id: The Gymnasium environment ID to be used.
        observation: A valid observation (state) from the environment.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = OneHotFeatureFunction(S, A)
    features = feature_fn(np.array([observation]))

    # The feature function should return a matrix of shape (S * (A - 1), A)
    expected_shape = (S * (A - 1), A)
    assert features.shape == expected_shape


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(n_observations=st.integers(min_value=2, max_value=10))
@settings(deadline=None)
def test_feature_function_output_shape_with_multiple_observations(env_id: str, n_observations: int):
    """
    Test that the feature function returns the correct output shape in case of multiple observations.

    Args:
        env_id: The Gymnasium environment ID to be used.
        n_observations: The number of observations for which to construct the features.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    observations = np.array([np.random.randint(0, 47) for _ in range(n_observations)])[np.newaxis, :]

    feature_fn = OneHotFeatureFunction(S, A)
    features = feature_fn(observations)

    # The feature function should return a matrix of shape (S * (A - 1), A, n_observations)
    expected_shape = (S * (A - 1), A, n_observations)
    assert features.shape == expected_shape


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_feature_function_one_hot_encoding(env_id: str):
    """
    Test that the feature function produces correct one-hot encoded features.

    Args:
        env_id: The Gymnasium environment ID to be used.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = OneHotFeatureFunction(S, A)

    # Test a specific observation
    observation = 0
    features = feature_fn(np.array([observation]))

    # Check that each row (except possibly the first action) has exactly one non-zero element
    # The first action is excluded to avoid linear dependency
    for i in range(1, A):
        row_sum = np.sum(features[:, i])
        assert row_sum == 1.0, f'Row {i} should sum to 1.0, got {row_sum}'
        assert np.sum(features[:, i] == 1.0) == 1, f'Row {i} should have exactly one 1.0'


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(observation=st.integers(min_value=0, max_value=47))
@settings(deadline=None)
def test_feature_function_consistent_output(env_id: str, observation: int):
    """
    Test that the feature function produces consistent output for the same observation.

    Args:
        env_id: The Gymnasium environment ID to be used.
        observation: A valid observation (state) from the environment.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = OneHotFeatureFunction(S, A)

    # Call the feature function multiple times with the same observation
    features1 = feature_fn(np.array([observation]))
    features2 = feature_fn(np.array([observation]))

    # Results should be identical
    np.testing.assert_array_equal(features1, features2)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_feature_function_n_features(env_id: str):
    """
    Test that n_features returns the correct number of features.

    Args:
        env_id: The Gymnasium environment ID to be used.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = OneHotFeatureFunction(S, A)
    assert feature_fn.n_features == S * (A - 1)


@pytest.mark.parametrize(
    'env_id, size',
    [
        (
            'CliffWalking-v1',
            [2, 1],
        ),
        (
            'CliffWalking-v1',
            [1, 1, 2],
        ),
    ],
)
def test_one_hot_feature_function_with_incorrect_shapes(env_id: str, size: List[int]):
    """
    Test that one-hot feature function throws error with incorrect observation sizes.

    Args:
        env_id: The Gymnasium environment ID to be used.
        size: The size of the observation matrix.

    """
    env = gym.make(env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = OneHotFeatureFunction(S, A)

    with pytest.raises(EnvironmentException):
        feature_fn(np.random.randint(0, 47, size=size))
