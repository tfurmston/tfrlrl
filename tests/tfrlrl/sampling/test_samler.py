import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tfrlrl.sampling.sampler import Sampler


class TestSampler:
    """This class encapsulates the unit tests for the Sampler class."""

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(st.integers(min_value=10, max_value=100))
    def test_sample_n_steps_without_limit(self, env_id: str, n_steps: int):
        """
        Test that n-steps can be sampled from the environment and that the outputs follow the expected format.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        sampler = Sampler(env_id)
        for n in range(n_steps):
            observation, action, next_observation, reward, done, info = next(sampler)
            assert isinstance(observation, np.ndarray)
            assert isinstance(next_observation, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(st.integers(min_value=10, max_value=100))
    def test_sample_n_steps_with_limit(self, env_id: str, n_steps: int):
        """
        Test that n-steps can be sampled from the environment and the limit on n_steps is respected.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        sampler = Sampler(env_id, n_steps)
        assert len(list(sampler)) == n_steps
