import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import construct_one_hot_feature_function
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import EpisodicSampler


class TestEpisodicSampler:
    """Class that encapsulates the unit tests for the EpisodicSampler class."""

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_sample_n_episodes_without_limit(self, env_id: str, n_episodes: int):
        """
        Test that n-episodes can be sampled from the environment and that the outputs follow the expected format.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
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
        sampler = EpisodicSampler(env_id, n_episodes=n_episodes, policy=pol)
        for rewards, gradients in sampler:
            assert isinstance(gradients, np.ndarray)
            assert isinstance(rewards, np.ndarray)

    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_sample_episode_with_env_kwargs(self, n_episodes: int):
        """
        Test that environment kwargs are correctly passed through to the environment construction.

        Uses FrozenLake-v1 with is_slippery parameter to verify kwargs functionality.

        :param n_steps: The number of steps to sample from the environment.
        """
        env_id = 'FrozenLake-v1'
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
        sampler = EpisodicSampler(
            env_id,
            n_episodes=n_episodes,
            policy=pol,
            is_slippery=False,
        )
        for rewards, gradients in sampler:
            assert isinstance(gradients, np.ndarray)
            assert isinstance(rewards, np.ndarray)
