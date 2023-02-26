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
        d_obs=4,
        d_action=1,
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
        d_obs=4,
        d_action=1,
        buffer_size=buffer_size,
    )

    sampler = RaySampler(env_id, n_envs, n_steps)
    for sample in sampler:
        buffer.add_steps(sample)
    assert buffer._indx == n_envs * n_steps % buffer_size
