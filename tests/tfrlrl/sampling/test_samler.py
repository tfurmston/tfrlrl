import numpy as np
import pytest
import ray
from hypothesis import given, settings
from hypothesis import strategies as st
from ray.util.iter import ParallelIteratorWorker, from_actors

from tfrlrl.sampling.sampler import RaySampler, Sampler


class TestSampler:
    """This class encapsulates the unit tests for the Sampler class."""

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=None)
    def test_sample_n_steps_without_limit(self, env_id: str, n_steps: int, test_ray_cluster):
        """
        Test that n-steps can be sampled from the environment and that the outputs follow the expected format.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        sampler = Sampler.remote(env_id)
        for n in range(n_steps):
            observation, action, next_observation, reward, done, info = ray.get(sampler.__next__.remote())
            assert isinstance(observation, np.ndarray)
            assert isinstance(next_observation, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    def test_sample_n_steps_with_limit(self, env_id: str, n_steps: int, test_ray_cluster):
        """
        Test that n-steps can be sampled from the environment and the limit on n_steps is respected.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        sampler = Sampler.remote(env_id, n_steps)
        assert len(list(sampler.__next__.remote())) == n_steps


class TestRaySampler:

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100), n_envs=st.integers(min_value=1, max_value=4))
    @settings(deadline=None)
    def test_ray_sample_n_steps_without_limit(self, env_id: str, n_steps: int, n_envs: int, test_ray_cluster):
        ray_sampler = RaySampler(env_id, n_envs)
        for n in range(n_steps):
            samples = next(ray_sampler)
            assert len(samples) == n_envs
            for observation, action, next_observation, reward, done, info in samples:
                assert isinstance(observation, np.ndarray)
                assert isinstance(next_observation, np.ndarray)
                assert isinstance(reward, float)
                assert isinstance(done, bool)
                assert isinstance(info, dict)
