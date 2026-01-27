import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.baselines.linear import LinearBaseline
from tfrlrl.sampling.sampler import Sampler


class TestLinearBaseline:
    """Class that encapsulates the unit tests for the LinearBaseline class."""

    @pytest.mark.parametrize('env_id,obs_dim', [('CartPole-v1', 4), ('MountainCar-v0', 2)])
    @given(n_steps=st.integers(min_value=5, max_value=100))
    @settings(deadline=2000)
    def test_calculate_features_output_shape(self, env_id: str, obs_dim: int, n_steps: int):
        """
        Test that calculate_features returns a feature matrix with the correct shape.

        The feature matrix should have shape (n_steps, 2*obs_dim + 4) where:
        - obs_dim features for original observations
        - obs_dim features for squared observations
        - 3 features for time step polynomial (t/100, (t/100)^2, (t/100)^3)
        - 1 constant feature (ones)

        :param env_id: The Gymnasium environment ID.
        :param obs_dim: The dimensionality of the observation space.
        :param n_steps: The number of time steps.
        """
        baseline = LinearBaseline()

        # Create random observations with correct shape
        observation_matrix = np.random.randn(n_steps, obs_dim)
        time_steps = np.arange(n_steps)

        # Calculate features
        features = baseline.calculate_features(observation_matrix, time_steps)

        # Verify shape
        expected_shape = (n_steps, 2 * obs_dim + 4)
        assert features.shape == expected_shape, (
            f'Feature matrix shape {features.shape} does not match expected shape {expected_shape}'
        )

    @given(
        n_steps=st.integers(min_value=5, max_value=50),
        obs_dim=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(deadline=2000)
    def test_calculate_features_observation_clipping(self, n_steps: int, obs_dim: int, seed: int):
        """
        Test that observations are clipped to the range [-10, 10] in the feature calculation.

        This test verifies that extreme observation values are properly clipped before
        being included in the feature matrix.

        :param n_steps: The number of time steps.
        :param obs_dim: The dimensionality of the observation space.
        :param seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        baseline = LinearBaseline()

        # Create observations with values outside [-10, 10] range
        observation_matrix = np.random.uniform(low=-100, high=100, size=(n_steps, obs_dim))
        time_steps = np.arange(n_steps)

        # Calculate features
        features = baseline.calculate_features(observation_matrix, time_steps)

        # Extract observation features (first 2*obs_dim columns)
        # First obs_dim columns are clipped observations
        # Next obs_dim columns are clipped observations squared
        observation_features = features[:, :obs_dim]
        observation_squared_features = features[:, obs_dim : 2 * obs_dim]

        # Verify that observation features are clipped to [-10, 10]
        assert np.all(observation_features >= -10), 'Some observation features are below -10'
        assert np.all(observation_features <= 10), 'Some observation features are above 10'

        # Verify that squared observation features are in [0, 100]
        assert np.all(observation_squared_features >= 0), 'Some squared observation features are negative'
        assert np.all(observation_squared_features <= 100), 'Some squared observation features are above 100'

    @given(obs_dim=st.integers(min_value=1, max_value=10))
    @settings(deadline=2000)
    def test_calculate_features_time_step_normalization(self, obs_dim: int):
        """
        Test that time steps are correctly normalized by 100.0 in the feature calculation.

        The feature matrix should contain time step features as:
        - Column 2*obs_dim: t/100
        - Column 2*obs_dim + 1: (t/100)^2
        - Column 2*obs_dim + 2: (t/100)^3
        - Column 2*obs_dim + 3: 1.0 (constant)

        :param obs_dim: The dimensionality of the observation space.
        """
        baseline = LinearBaseline()

        # Create specific time steps to test normalization
        time_steps = np.array([0, 50, 100, 200])
        n_steps = len(time_steps)

        # Create dummy observations (values don't matter for this test)
        observation_matrix = np.zeros((n_steps, obs_dim))

        # Calculate features
        features = baseline.calculate_features(observation_matrix, time_steps)

        # Extract time step features (columns 2*obs_dim to 2*obs_dim + 3)
        time_feature_t1 = features[:, 2 * obs_dim]  # t/100
        time_feature_t2 = features[:, 2 * obs_dim + 1]  # (t/100)^2
        time_feature_t3 = features[:, 2 * obs_dim + 2]  # (t/100)^3
        constant_feature = features[:, 2 * obs_dim + 3]  # constant (ones)

        # Expected normalized time steps
        expected_t1 = time_steps / 100.0
        expected_t2 = (time_steps / 100.0) ** 2
        expected_t3 = (time_steps / 100.0) ** 3

        # Verify time step normalization
        np.testing.assert_allclose(
            time_feature_t1,
            expected_t1,
            rtol=1e-6,
            atol=1e-9,
            err_msg='Linear time feature (t/100) does not match expected values',
        )
        np.testing.assert_allclose(
            time_feature_t2,
            expected_t2,
            rtol=1e-6,
            atol=1e-9,
            err_msg='Quadratic time feature (t/100)^2 does not match expected values',
        )
        np.testing.assert_allclose(
            time_feature_t3,
            expected_t3,
            rtol=1e-6,
            atol=1e-9,
            err_msg='Cubic time feature (t/100)^3 does not match expected values',
        )

        # Verify constant feature is all ones
        np.testing.assert_allclose(
            constant_feature,
            np.ones(n_steps),
            rtol=1e-6,
            atol=1e-9,
            err_msg='Constant feature should be all ones',
        )

    @pytest.mark.parametrize('env_id', ['CartPole-v1', 'MountainCar-v0'])
    @given(n_steps=st.integers(min_value=10, max_value=50))
    @settings(deadline=2000)
    def test_calculate_features_with_sampled_episodes(self, env_id: str, n_steps: int):
        """
        Test calculate_features with real episode data sampled from a Gymnasium environment.

        This integration test verifies that the feature calculation works correctly
        with actual environment observations and time steps.

        :param env_id: The Gymnasium environment ID to sample from.
        :param n_steps: The number of steps to sample from the environment.
        """
        baseline = LinearBaseline()
        env = gym.make(env_id)

        # Determine observation dimension based on environment
        if hasattr(env.observation_space, 'n'):
            # Discrete observation space (like FrozenLake)
            raise EnvironmentError('Linear baseline can only be used on continuous observation spaces.')
        else:
            # Box observation space (like CartPole)
            obs_dim = env.observation_space.shape[0]

        # Sample steps from the environment
        sampler = Sampler(env_id, n_steps=n_steps)
        observations = []
        time_steps = []

        for sample in sampler:
            observations.append(sample.observation)
            time_steps.append(sample.time_step)

        # Convert to numpy arrays
        observation_matrix = np.concatenate(observations, axis=1).T
        time_steps_array = np.array(time_steps)

        # Calculate features
        features = baseline.calculate_features(observation_matrix, time_steps_array)

        # Verify shape
        expected_shape = (n_steps, 2 * obs_dim + 4)
        assert features.shape == expected_shape, (
            f'Feature matrix shape {features.shape} does not match expected shape {expected_shape}'
        )

        # Verify no NaN or Inf values in features
        assert not np.any(np.isnan(features)), 'Feature matrix contains NaN values'
        assert not np.any(np.isinf(features)), 'Feature matrix contains infinite values'

        # Verify constant column is all ones
        constant_column = features[:, -1]
        np.testing.assert_allclose(
            constant_column,
            np.ones(n_steps),
            rtol=1e-6,
            atol=1e-9,
            err_msg='Constant feature column should be all ones',
        )
