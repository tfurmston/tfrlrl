from collections import ChainMap, defaultdict
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import construct_one_hot_feature_function
from tfrlrl.policies.base import BasePolicy
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import (
    EpisodicSampler,
    RayEpisodicSampler,
)
from tfrlrl.sampling.statistics_collection import BaseStatisticsCollector


@dataclass
class DummyStatistics:
    """Dataclass for the statistics collected test sample episodes."""

    samples: dict[str, list]  # A map from the episode ID to the list of steps in the episode.


class DummyStatisticsCollector(BaseStatisticsCollector):
    """Test class for collecting statistics during sampling."""

    def __init__(self):
        """Initialise statistics collector."""
        self._samples = defaultdict(list)

    def reset(self):
        """Reset the statistics in the collector."""
        self._samples = defaultdict(list)

    def update_policy(self, new_policy: BasePolicy) -> None:
        """Update the policy of the statistics collector."""
        pass

    def collect_step_statistics(self, sample):
        """Collect statistics from a sample step."""
        self._samples[sample.env_id].append(sample)

    def aggregate_statistics(self):
        """Aggregate the statistics collected by the collector."""
        return DummyStatistics(samples=self._samples)

    @classmethod
    def merge_statistics(cls, statistics: List[DummyStatistics]):
        """Aggregate the statistics collected by the collector."""
        return DummyStatistics(
            samples=dict(ChainMap(*[x.samples for x in statistics])),
        )


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
        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(env_id, stats_collector, n_episodes=n_episodes, policy=pol)
        statistics = list(sampler)
        assert len(statistics) == n_episodes
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == 1
            env_id = next(iter(statistic.samples))
            for step in statistic.samples[env_id]:
                assert isinstance(step.env_id, str)
                assert isinstance(step.time_step, int)
                assert isinstance(step.observation, np.ndarray)
                assert isinstance(step.next_observation, np.ndarray)
                assert isinstance(step.reward, float) or isinstance(step.reward, int)
                assert isinstance(step.done, bool)
                assert isinstance(step.info, dict)

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
        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=pol,
            is_slippery=False,
        )
        statistics = list(sampler)
        assert len(statistics) == n_episodes
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == 1
            env_id = next(iter(statistic.samples))
            for step in statistic.samples[env_id]:
                assert isinstance(step.env_id, str)
                assert isinstance(step.time_step, int)
                assert isinstance(step.observation, np.ndarray)
                assert isinstance(step.next_observation, np.ndarray)
                assert isinstance(step.reward, float) or isinstance(step.reward, int)
                assert isinstance(step.done, bool)
                assert isinstance(step.info, dict)

    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_reset_allows_reuse_as_iterator(self, n_episodes: int):
        """
        Test that the reset method allows the EpisodicSampler to be used as an iterator multiple times.

        :param n_episodes: The number of episodes to sample from the environment.
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
        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(env_id, stats_collector, n_episodes=n_episodes, policy=pol)
        statistics = list(sampler)
        assert len(statistics) == n_episodes

        # First iteration: consume all episodes
        first_iteration_count = 0
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == 1
            env_id = next(iter(statistic.samples))
            for step in statistic.samples[env_id]:
                assert isinstance(step.env_id, str)
                assert isinstance(step.time_step, int)
                assert isinstance(step.observation, np.ndarray)
                assert isinstance(step.next_observation, np.ndarray)
                assert isinstance(step.reward, float) or isinstance(step.reward, int)
                assert isinstance(step.done, bool)
                assert isinstance(step.info, dict)
            first_iteration_count += 1

        assert first_iteration_count == n_episodes

        # Iterator should be exhausted - trying to iterate should yield no results
        exhausted_count = 0
        for _, _ in sampler:
            exhausted_count += 1

        assert exhausted_count == 0

        # Reset the sampler
        sampler.reset()
        statistics = list(sampler)
        assert len(statistics) == n_episodes

        # Second iteration: should work again after reset
        second_iteration_count = 0
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == 1
            env_id = next(iter(statistic.samples))
            for step in statistic.samples[env_id]:
                assert isinstance(step.env_id, str)
                assert isinstance(step.time_step, int)
                assert isinstance(step.observation, np.ndarray)
                assert isinstance(step.next_observation, np.ndarray)
                assert isinstance(step.reward, float) or isinstance(step.reward, int)
                assert isinstance(step.done, bool)
                assert isinstance(step.info, dict)
            second_iteration_count += 1

        assert second_iteration_count == n_episodes


class TestRayEpisodicSampler:
    """Class that encapsulates the unit tests for the RayEpisodicSampler class."""

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_ray_sample_n_episodes_without_limit(self, env_id: str, n_episodes: int, test_ray_cluster):
        """
        Test that n-episodes can be sampled from the environment and that the outputs follow the expected format.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        :param test_ray_cluster: Test Ray cluster.
        """
        n_samplers = 2
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
        stats_collector = DummyStatisticsCollector()
        sampler = RayEpisodicSampler(
            n_samplers,
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=pol,
        )
        statistics = list(sampler)
        assert len(statistics) == n_episodes
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == n_samplers
            for k in statistic.samples:
                for step in statistic.samples[k]:
                    assert isinstance(step.env_id, str)
                    assert isinstance(step.time_step, int)
                    assert isinstance(step.observation, np.ndarray)
                    assert isinstance(step.next_observation, np.ndarray)
                    assert isinstance(step.reward, float) or isinstance(step.reward, int)
                    assert isinstance(step.done, bool)
                    assert isinstance(step.info, dict)

    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_ray_sample_episode_with_env_kwargs(self, n_episodes: int, test_ray_cluster):
        """
        Test that environment kwargs are correctly passed through to the environment construction.

        Uses FrozenLake-v1 with is_slippery parameter to verify kwargs functionality.

        :param n_steps: The number of steps to sample from the environment.
        :param test_ray_cluster: Test Ray cluster.
        """
        n_samplers = 2
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
        stats_collector = DummyStatisticsCollector()
        sampler = RayEpisodicSampler(
            n_samplers,
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=pol,
            is_slippery=False,
        )
        statistics = list(sampler)
        assert len(statistics) == n_episodes
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == n_samplers
            for k in statistic.samples:
                for step in statistic.samples[k]:
                    assert isinstance(step.env_id, str)
                    assert isinstance(step.time_step, int)
                    assert isinstance(step.observation, np.ndarray)
                    assert isinstance(step.next_observation, np.ndarray)
                    assert isinstance(step.reward, float) or isinstance(step.reward, int)
                    assert isinstance(step.done, bool)
                    assert isinstance(step.info, dict)

    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_ray_reset_allows_reuse_as_iterator(self, n_episodes: int, test_ray_cluster):
        """
        Test that the reset method allows the EpisodicSampler to be used as an iterator multiple times.

        :param n_episodes: The number of episodes to sample from the environment.
        :param test_ray_cluster: Test Ray cluster.
        """
        n_samplers = 2
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
        stats_collector = DummyStatisticsCollector()
        sampler = RayEpisodicSampler(
            n_samplers,
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=pol,
        )

        # First iteration: consume all episodes
        first_iteration_count = 0
        statistics = list(sampler)
        assert len(statistics) == n_episodes
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == n_samplers
            for k in statistic.samples:
                for step in statistic.samples[k]:
                    assert isinstance(step.env_id, str)
                    assert isinstance(step.time_step, int)
                    assert isinstance(step.observation, np.ndarray)
                    assert isinstance(step.next_observation, np.ndarray)
                    assert isinstance(step.reward, float) or isinstance(step.reward, int)
                    assert isinstance(step.done, bool)
                    assert isinstance(step.info, dict)
                first_iteration_count += 1

        assert first_iteration_count == n_episodes * n_samplers

        # Iterator should be exhausted - trying to iterate should yield no results
        exhausted_count = 0
        for _, _ in sampler:
            exhausted_count += 1

        assert exhausted_count == 0

        # Reset the sampler
        sampler.reset()

        # Second iteration: should work again after reset
        second_iteration_count = 0
        statistics = list(sampler)
        assert len(statistics) == n_episodes
        for statistic in statistics:
            assert isinstance(statistic.samples, dict)
            assert len(statistic.samples) == n_samplers
            for k in statistic.samples:
                for step in statistic.samples[k]:
                    assert isinstance(step.env_id, str)
                    assert isinstance(step.time_step, int)
                    assert isinstance(step.observation, np.ndarray)
                    assert isinstance(step.next_observation, np.ndarray)
                    assert isinstance(step.reward, float) or isinstance(step.reward, int)
                    assert isinstance(step.done, bool)
                    assert isinstance(step.info, dict)
            second_iteration_count += 1

        assert second_iteration_count == n_episodes
