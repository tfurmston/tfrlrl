import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.policies.base import UniformActionSamplingPolicy
from tfrlrl.sampling.sampler import Sampler


class TestSampler:
    """Class that encapsulates the unit tests for the Sampler class."""

    @pytest.mark.parametrize('env_id', ['CartPole-v1', 'CliffWalking-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=2000)
    def test_sample_n_steps_without_limit(self, env_id: str, n_steps: int):
        """
        Test that n-steps can be sampled from the environment and that the outputs follow the expected format.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        sampler = Sampler(env_id)
        for _ in range(n_steps):
            sample = sampler.__next__()
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)

    @pytest.mark.parametrize('env_id', ['CartPole-v1', 'CliffWalking-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=2000)
    def test_sample_n_steps_with_policy(self, env_id: str, n_steps: int):
        """
        Test that n-steps can be sampled from the environment with a custom policy.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        policy = UniformActionSamplingPolicy(env_id)
        sampler = Sampler(env_id, n_steps=n_steps, policy=policy)
        for sample in sampler:
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)
            # Verify action is valid for the environment
            assert isinstance(sample.action, (int, np.integer))
            if env_id == 'CartPole-v1':
                assert 0 <= sample.action < 2  # CartPole has 2 actions
            elif env_id == 'CliffWalking-v1':
                assert 0 <= sample.action < 4  # Cliff Walking has 4 actions

    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=2000)
    def test_sample_with_env_kwargs(self, n_steps: int):
        """
        Test that environment kwargs are correctly passed through to the environment construction.

        Uses FrozenLake-v1 with is_slippery parameter to verify kwargs functionality.

        :param n_steps: The number of steps to sample from the environment.
        """
        env_id = 'FrozenLake-v1'
        sampler = Sampler(env_id, n_steps=n_steps, is_slippery=False)
        for sample in sampler:
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)
            # Verify action is valid for FrozenLake (4 actions: 0, 1, 2, 3)
            assert isinstance(sample.action, (int, np.integer))
            assert 0 <= sample.action < 4

    @pytest.mark.parametrize('env_id', ['CartPole-v1', 'CliffWalking-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=50))
    @settings(deadline=4000)
    def test_reset_allows_multiple_iterations(self, env_id: str, n_steps: int):
        """
        Test that the reset method allows the sampler to be iterated over multiple times.

        This test verifies that after exhausting the iterator, calling reset() allows
        the sampler to be used again for a fresh iteration.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment per iteration.
        """
        sampler = Sampler(env_id, n_steps=n_steps)

        # First iteration: collect all samples
        first_iteration_samples = list(sampler)
        assert len(first_iteration_samples) == n_steps

        # Verify that the iterator is exhausted
        with pytest.raises(StopIteration):
            next(sampler)

        # Reset the sampler
        sampler.reset()

        # Second iteration: should be able to iterate again
        second_iteration_samples = list(sampler)
        assert len(second_iteration_samples) == n_steps

        # Verify that both iterations produced valid samples
        for sample in first_iteration_samples + second_iteration_samples:
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)
