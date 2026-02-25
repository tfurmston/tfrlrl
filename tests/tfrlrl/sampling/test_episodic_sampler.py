import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import (
    DummyStatistics,
    DummyStatisticsCollector,
)
from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import (
    EpisodicSampler,
    RayEpisodicSampler,
)


class TestEpisodicSampler:
    """Class that encapsulates the unit tests for the EpisodicSampler class."""

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_sample_n_episodes_without_limit(self, env_id: str, n_episodes: int):
        """
        Test that n-episodes can be sampled from the environment and that the outputs follow the expected format.

        Args:
            env_id: The Gym environment ID to be used in the sampling.
            n_episodes: The number of n_episodes to sample from the environment.

        """
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(env_id, stats_collector, n_episodes=n_episodes, policy=policy)
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

        Args:
            n_episodes: The number of episodes to sample from the environment.

        """
        env_id = 'FrozenLake-v1'
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
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
    def test_sample_episode_with_policy_update(self, n_episodes: int):
        """
        Test that updating the policy.

        Uses LinearSoftmaxPolicy to test update of policies.

        Args:
            n_episodes: The number of episodes to sample from the environment.

        """
        env_id = 'FrozenLake-v1'
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
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

        sampler.reset()
        sampler.update(state_dict=policy.get_state())

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

        Args:
            n_episodes: The number of episodes to sample from the environment.

        """
        env_id = 'FrozenLake-v1'
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
        )
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

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_sample_without_limits(self, env_id: str, n_episodes: int):
        """
        Test the sample function of the EpisodicSampler class and that the outputs follow the expected format.

        Args:
            env_id: The Gym environment ID to be used in the sampling.
            n_episodes: The number of episodes to sample from the environment.

        """
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = EpisodicSampler(
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
        )
        statistics = sampler.sample()
        assert isinstance(statistics, DummyStatistics)
        assert isinstance(statistics.samples, dict)
        for k in statistics.samples:
            for step in statistics.samples[k]:
                assert isinstance(step.env_id, str)
                assert isinstance(step.time_step, int)
                assert isinstance(step.observation, np.ndarray)
                assert isinstance(step.next_observation, np.ndarray)
                assert isinstance(step.reward, float) or isinstance(step.reward, int)
                assert isinstance(step.done, bool)
                assert isinstance(step.info, dict)


class TestRayEpisodicSampler:
    """Class that encapsulates the unit tests for the RayEpisodicSampler class."""

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_ray_sample_n_episodes_without_limit(self, env_id: str, n_episodes: int, test_ray_cluster):
        """
        Test that n-episodes can be sampled from the environment and that the outputs follow the expected format.

        Args:
            env_id: The Gym environment ID to be used in the sampling.
            n_episodes: The number of episodes to sample from the environment.
            test_ray_cluster: Test Ray cluster.

        """
        n_samplers = 2
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = RayEpisodicSampler(
            n_samplers,
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
        )
        statistics = list(sampler)
        assert len(statistics) == n_episodes // n_samplers
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

        Args:
            n_episodes: The number of episodes to sample from the environment.
            test_ray_cluster: Test Ray cluster.

        """
        n_samplers = 2
        env_id = 'FrozenLake-v1'
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = RayEpisodicSampler(
            n_samplers,
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
            is_slippery=False,
        )
        statistics = list(sampler)
        assert len(statistics) == n_episodes // n_samplers
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

        Args:
            n_episodes: The number of episodes to sample from the environment.
            test_ray_cluster: Test Ray cluster.

        """
        n_samplers = 2
        env_id = 'FrozenLake-v1'
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = RayEpisodicSampler(
            n_samplers,
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
        )

        # First iteration: consume all episodes
        first_iteration_count = 0
        statistics = list(sampler)
        assert len(statistics) == n_episodes // n_samplers
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
        assert len(statistics) == n_episodes // n_samplers
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

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(n_episodes=st.integers(min_value=2, max_value=10))
    @settings(deadline=2000)
    def test_ray_sample_without_limits(self, env_id: str, n_episodes: int, test_ray_cluster):
        """
        Test the sample function of the EpisodicSampler class and that the outputs follow the expected format.

        Args:
            env_id: The Gym environment ID to be used in the sampling.
            n_episodes: The number of episodes to sample from the environment.
            test_ray_cluster: Test Ray cluster.

        """
        n_samplers = 2
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        stats_collector = DummyStatisticsCollector()
        sampler = RayEpisodicSampler(
            n_samplers,
            env_id,
            stats_collector,
            n_episodes=n_episodes,
            policy=policy,
        )
        statistics = sampler.sample()
        assert isinstance(statistics, DummyStatistics)
        assert isinstance(statistics.samples, dict)
        for k in statistics.samples:
            for step in statistics.samples[k]:
                assert isinstance(step.env_id, str)
                assert isinstance(step.time_step, int)
                assert isinstance(step.observation, np.ndarray)
                assert isinstance(step.next_observation, np.ndarray)
                assert isinstance(step.reward, float) or isinstance(step.reward, int)
                assert isinstance(step.done, bool)
                assert isinstance(step.info, dict)
