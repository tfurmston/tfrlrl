"""Tests for statistics collection classes."""

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import construct_one_hot_feature_function
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

        :param env_id: The Gym environment ID to be used in testing.
        :param n_episodes: The number of episodes to sample before testing reset.
        """
        env = gym.make(env_id)
        S = env.observation_space.n
        A = env.action_space.n

        feature_fn = construct_one_hot_feature_function(S=S, A=A)
        softmax_parameters = np.random.random(size=S * (A - 1))
        pol = LinearSoftMax(
            env_id,
            softmax_parameters,
            feature_fn,
        )

        stats_collector = EpisocidPolicyGradientStatisticsCollector(pol)
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=pol,
            is_slippery=False,
        )

        # Collect some episodes
        for _ in sampler:
            pass

        # Verify that statistics were collected
        assert len(stats_collector.rewards) > 0
        assert len(stats_collector.log_pol_grads) > 0

        # Reset the statistics collector
        stats_collector.reset()

        # Verify that statistics are cleared
        assert len(stats_collector.rewards) == 0
        assert len(stats_collector.log_pol_grads) == 0
        assert stats_collector.rewards == []
        assert stats_collector.log_pol_grads == []

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=1, max_value=5))
    @settings(deadline=5000)
    def test_aggregate_statistics_return_types_and_dimensions(self, env_id: str, n_episodes: int):
        """
        Test that aggregate_statistics returns two numpy arrays with expected dimensions.

        The first array should have length equal to the number of sampled steps,
        and the second should have length equal to the number of policy parameters.

        :param env_id: The Gym environment ID to be used in testing.
        :param n_episodes: The number of episodes to sample.
        """
        env = gym.make(env_id)
        S = env.observation_space.n
        A = env.action_space.n

        # Verify dimensions
        n_params = S * (A - 1)  # Number of policy parameters

        feature_fn = construct_one_hot_feature_function(S=S, A=A)
        softmax_parameters = np.random.random(size=n_params)
        pol = LinearSoftMax(
            env_id,
            softmax_parameters,
            feature_fn,
        )

        stats_collector = EpisocidPolicyGradientStatisticsCollector(pol)
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=pol,
            is_slippery=False,
        )

        # Collect episodes
        for rewards, episode_gradient in sampler:
            # Verify return types
            assert isinstance(rewards, np.ndarray)
            assert isinstance(episode_gradient, np.ndarray)

            print(episode_gradient.shape)
            # Episode gradient should have length equal to number of policy parameters
            assert episode_gradient.shape == (n_params,)
            assert len(episode_gradient) == n_params

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=5))
    @settings(deadline=5000)
    def test_reset_and_reuse(self, env_id: str, n_episodes: int):
        """
        Test that after reset, the collector can be reused for new episodes.

        :param env_id: The Gym environment ID to be used in testing.
        :param n_episodes: The number of episodes to sample in each iteration.
        """
        env = gym.make(env_id)
        S = env.observation_space.n
        A = env.action_space.n

        feature_fn = construct_one_hot_feature_function(S=S, A=A)
        softmax_parameters = np.random.random(size=S * (A - 1))
        pol = LinearSoftMax(
            env_id,
            softmax_parameters,
            feature_fn,
        )

        stats_collector = EpisocidPolicyGradientStatisticsCollector(pol)
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=pol,
            is_slippery=False,
        )

        # First collection
        for _ in sampler:
            pass

        first_rewards, first_gradient = stats_collector.aggregate_statistics()
        first_n_steps = len(first_rewards)

        # Reset everything
        stats_collector.reset()
        sampler.reset()

        # Second collection
        for _ in sampler:
            pass

        second_rewards, second_gradient = stats_collector.aggregate_statistics()
        second_n_steps = len(second_rewards)

        # Verify both collections produced valid results
        assert first_n_steps > 0
        assert second_n_steps > 0
        assert first_gradient.shape == second_gradient.shape
        assert first_rewards.shape[0] == first_n_steps
        assert second_rewards.shape[0] == second_n_steps
