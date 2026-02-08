import gymnasium as gym
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import (
    DummyStatisticsCollector,
)
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.sampling.episodic_sampler import (
    EpisodicSampler,
)


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
def test_sample_single_action_from_dense_network_policy(env_id):
    """
    Test sampling a single action with a DenseNetworkPolicy.

    env_id: The environment I.D. from which to sample episodes.
    """
    env = gym.make(env_id)

    hidden_space_dims = [16, 32]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    action = policy.generate_action(env.observation_space.sample())
    assert action.shape == (1,)


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
@given(
    n_episodes=st.integers(min_value=0, max_value=20),
    h1_dimensions=st.integers(min_value=8, max_value=32),
    h2_dimensions=st.integers(min_value=24, max_value=48),
)
@settings(deadline=None)
def test_sample_episode_with_dense_network_policy(env_id: str, n_episodes: int, h1_dimensions: int, h2_dimensions: int):
    """
    Test sampling of episodes with a DenseNetworkPolicy.

    env_id: The environment I.D. from which to sample episodes.
    n_episodes: The number of episodes to sample.
    h1_dimensions: The number of units in the first hidden layer.
    h2_dimensions: The number of units in the second hidden layer.
    """
    statistics_collector = DummyStatisticsCollector()

    hidden_space_dims = [h1_dimensions, h2_dimensions]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    sampler = EpisodicSampler(
        env_id=env_id,
        statistics_collector=statistics_collector,
        n_episodes=n_episodes,
        policy=policy,
    )
    samples = list(sampler)
    assert len(samples) == n_episodes
