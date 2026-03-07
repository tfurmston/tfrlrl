import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import EpisodicSampler
from tfrlrl.sampling.statistics_collection import EpisocidPolicyGradientStatisticsCollector


class TestEpisocidPolicyGradientStatisticsCollector:
    """Tests for the EpisocidPolicyGradientStatisticsCollector class."""

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=1, max_value=5))
    @settings(deadline=5000)
    def test_reset_clears_statistics(self, env_id: str, n_episodes: int):
        """
        Test that the reset method clears out all collected statistics.

        Args:
            env_id: The Gym environment ID to be used in testing.
            n_episodes: The number of episodes to sample before testing reset.

        """
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = EpisocidPolicyGradientStatisticsCollector(env_id)
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
            is_slippery=False,
        )

        # Collect some episodes
        for _ in sampler:
            pass

        # Verify that statistics were collected
        assert len(stats_collector.steps) > 0

        # Reset the statistics collector
        stats_collector.reset()

        # Verify that statistics are cleared
        assert len(stats_collector.steps) == 0
        assert stats_collector.steps == []

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=1, max_value=5))
    @settings(deadline=5000)
    def test_aggregate_statistics_return_types_and_dimensions(self, env_id: str, n_episodes: int):
        """
        Test that aggregate_statistics returns two numpy arrays with expected dimensions.

        The first array should have length equal to the number of sampled steps,
        and the second should have length equal to the number of policy parameters.

        Args:
            env_id: The Gym environment ID to be used in testing.
            n_episodes: The number of episodes to sample.

        """
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = EpisocidPolicyGradientStatisticsCollector(env_id)
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
            is_slippery=False,
        )

        # Collect episodes
        for statistics in sampler:
            # Verify return types
            assert isinstance(statistics.total_reward, np.int64)
            assert isinstance(statistics.observations, np.ndarray)
            assert isinstance(statistics.actions, np.ndarray)
            assert isinstance(statistics.total_expected_rewards, np.ndarray)
            assert statistics.observations.shape[-1] == statistics.actions.shape[-1]
            assert statistics.observations.shape[-1] == statistics.total_expected_rewards.shape[-1]

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=5))
    @settings(deadline=5000)
    def test_reset_and_reuse(self, env_id: str, n_episodes: int):
        """
        Test that after reset, the collector can be reused for new episodes.

        Args:
            env_id: The Gym environment ID to be used in testing.
            n_episodes: The number of episodes to sample in each iteration.

        """
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = EpisocidPolicyGradientStatisticsCollector(env_id)
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
            is_slippery=False,
        )

        # First collection
        statistics1 = [x for x in sampler]

        # Reset everything
        sampler.reset()

        # Second collection
        statistics2 = [x for x in sampler]

        # Verify both collections produced valid results
        assert len(statistics1) == n_episodes
        assert len(statistics2) == n_episodes

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1', 'InvertedPendulum-v5'])
    @given(n_episodes=st.integers(min_value=1, max_value=5))
    @settings(deadline=5000)
    def test_merge_statistics(self, env_id: str, n_episodes: int):
        """
        Test that merge_statistics returns two numpy arrays with expected dimensions.

        Args:
            env_id: The Gym environment ID to be used in testing.
            n_episodes: The number of episodes to sample.

        """
        env = gym.make(env_id)

        if env_id == 'InvertedPendulum-v5':
            hidden_space_dims = [16, 32]
            policy = DenseNetworkPolicy(
                env_id=env_id,
                hidden_space_dims=hidden_space_dims,
            )
            env_kwargs = {}
        else:
            feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
            policy = LinearSoftMax(env_id, feature_fn)
            env_kwargs = {'is_slippery': False}

        stats_collector = EpisocidPolicyGradientStatisticsCollector(env_id)
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
            **env_kwargs,
        )
        statistics = sampler.sample()

        # Verify return types
        assert isinstance(statistics.total_reward, np.ndarray)
        assert isinstance(statistics.observations, np.ndarray)
        assert isinstance(statistics.actions, np.ndarray)
        assert isinstance(statistics.total_expected_rewards, np.ndarray)

        assert len(statistics.total_reward) == n_episodes
        assert statistics.observations.shape[-1] == statistics.actions.shape[-1]
        assert statistics.observations.shape[-1] == statistics.total_expected_rewards.shape[-1]

        assert len(statistics.observations.shape) == 2

        assert len(statistics.actions.shape) == 2
        assert len(statistics.total_expected_rewards.shape) == 1
