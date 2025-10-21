import gymnasium as gym
import pytest
import ray
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.replay_buffer.replay_buffer import ReplayBuffer
from tfrlrl.sampling.sampler import RaySampler, Sampler


@pytest.mark.parametrize('env_id', ['CartPole-v1'])
@given(n_steps=st.integers(min_value=10, max_value=100))
@settings(deadline=None)
def test_add_step(env_id: str, n_steps: int, test_ray_cluster):
    """
    Test that step can be sampled from the environment and added to replay buffer.

    :param env_id: The Gym environment ID to be used in the sampling.
    :param n_steps: The number of steps to sample from the environment.
    :param test_ray_cluster: PyTest fixture to start Ray cluster.
    """
    buffer = ReplayBuffer(
        env_id=env_id,
        buffer_size=100,
    )

    sampler = Sampler.remote(env_id)
    for n in range(n_steps):
        sample = ray.get(sampler.__next__.remote())
        buffer.add_step(sample)


@pytest.mark.parametrize('env_id', ['CartPole-v1'])
@given(n_steps=st.integers(min_value=10, max_value=1000))
@settings(deadline=None)
def test_add_steps(env_id: str, n_steps: int, test_ray_cluster):
    """
    Test that n-steps can be sampled from the environment and added to replay buffer.

    :param env_id: The Gym environment ID to be used in the sampling.
    :param n_steps: The number of steps to sample from the environment.
    :param test_ray_cluster: PyTest fixture to start Ray cluster.
    """
    n_envs = 2
    buffer_size = 1000

    buffer = ReplayBuffer(
        env_id=env_id,
        buffer_size=buffer_size,
    )

    sampler = RaySampler(env_id, n_envs, n_steps)
    for sample in sampler:
        buffer.add_steps(sample)
    assert buffer._indx == n_envs * n_steps % buffer_size


@pytest.mark.parametrize('env_id', ['CartPole-v1'])
@given(
    n_steps=st.integers(min_value=100, max_value=1000),
    n_samples=st.integers(min_value=10, max_value=50))
@settings(deadline=None)
def test_sample(env_id: str, n_steps: int, n_samples: int, test_ray_cluster):
    """
    Test sampling from replay buffer.

    :param env_id: The Gym environment ID to be used in the sampling.
    :param n_steps: The number of steps to sample from the environment.
    :param n_samples: The number of steps to sample from the replay buffer
    :param test_ray_cluster: PyTest fixture to start Ray cluster.
    """
    n_envs = 2
    buffer_size = 1000

    buffer = ReplayBuffer(
        env_id=env_id,
        buffer_size=buffer_size,
    )

    sampler = RaySampler(env_id, n_envs, n_steps)
    for sample in sampler:
        buffer.add_steps(sample)

    env = gym.make(env_id)
    samples = buffer.sample(n_samples)
    assert samples.observations.shape == env.observation_space.shape + (n_samples, )
    assert samples.next_observations.shape == env.observation_space.shape + (n_samples, )
    if isinstance(env.action_space, gym.spaces.Discrete):
        assert samples.actions.shape == (1, n_samples)
    else:
        assert samples.actions.shape == env.action_space.shape + (n_samples, )
    assert samples.rewards.shape == (1, n_samples)
    assert samples.dones.shape == (1, n_samples)
