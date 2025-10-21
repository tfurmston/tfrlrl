import numpy as np
import pytest
import ray
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.policies.base import UniformActionSamplingPolicy
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
            sample = ray.get(sampler.__next__.remote())
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=None)
    def test_sample_n_steps_with_policy(self, env_id: str, n_steps: int, test_ray_cluster):
        """
        Test that n-steps can be sampled from the environment with a custom policy.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        :param test_ray_cluster: PyTest fixture to start Ray cluster.
        """
        policy = UniformActionSamplingPolicy(env_id)
        sampler = Sampler.remote(env_id, policy=policy)
        for n in range(n_steps):
            sample = ray.get(sampler.__next__.remote())
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)
            # Verify action is valid for the environment
            assert isinstance(sample.action, (int, np.integer))
            assert 0 <= sample.action < 2  # CartPole has 2 actions


class TestRaySampler:
    """A test class for testing the RaySampler class."""

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100), n_envs=st.integers(min_value=1, max_value=4))
    @settings(deadline=None)
    def test_ray_sample_n_steps_without_limit(self, env_id: str, n_steps: int, n_envs: int, test_ray_cluster):
        """
        Test that n-steps can be sampled from the environment and that any number of steps can be sampled.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        :param n_envs: The number of environments from which to sample.
        :param test_ray_cluster: PyTest fixture to start Ray cluster.
        """
        ray_sampler = RaySampler(env_id, n_envs)
        for n in range(n_steps):
            samples = next(ray_sampler)
            assert isinstance(samples, ray_sampler.steps_cls)
            assert isinstance(samples.env_ids, list)
            assert isinstance(samples.time_steps, np.ndarray)
            assert isinstance(samples.observations, np.ndarray)
            assert isinstance(samples.actions, np.ndarray)
            assert isinstance(samples.next_observations, np.ndarray)
            assert isinstance(samples.rewards, np.ndarray)
            assert isinstance(samples.dones, np.ndarray)

            assert len(samples.env_ids) == n_envs
            assert samples.time_steps.shape == (n_envs,)
            assert samples.observations.shape == (4, n_envs)
            assert samples.actions.shape == (n_envs,)
            assert samples.next_observations.shape == (4, n_envs)
            assert samples.rewards.shape == (n_envs,)
            assert samples.dones.shape == (n_envs,)

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=None)
    def test_ray_sample_n_steps_with_limit(self, env_id: str, n_steps: int, test_ray_cluster):
        """
        Test that n-steps can be sampled from the environment and the limit on n_steps is respected.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        :param test_ray_cluster: PyTest fixture to start Ray cluster.
        """
        n_envs = 2
        sampler = RaySampler(env_id, n_envs, n_steps)
        samples = list(sampler)
        assert len(samples) == n_steps
        for sample in samples:
            assert isinstance(sample, sampler.steps_cls)
            assert isinstance(sample.env_ids, list)
            assert isinstance(sample.time_steps, np.ndarray)
            assert isinstance(sample.observations, np.ndarray)
            assert isinstance(sample.actions, np.ndarray)
            assert isinstance(sample.next_observations, np.ndarray)
            assert isinstance(sample.rewards, np.ndarray)
            assert isinstance(sample.dones, np.ndarray)

            assert len(sample.env_ids) == n_envs
            assert sample.time_steps.shape == (n_envs,)
            assert sample.observations.shape == (4, n_envs)
            assert sample.actions.shape == (n_envs,)
            assert sample.next_observations.shape == (4, n_envs)
            assert sample.rewards.shape == (n_envs,)
            assert sample.dones.shape == (n_envs,)
