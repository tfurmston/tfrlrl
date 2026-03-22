import copy

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.baselines.linear import LinearBaseline
from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import EpisodicSampler
from tfrlrl.sampling.statistics_collection import EpisocidPolicyGradientStatisticsCollector
from tfrlrl.training_algorithms.sgd import train_policy_gradient


class TestTrainPolicyGradient:
    """Unit tests for the train_policy_gradient function."""

    @pytest.mark.parametrize(
        'env_id, use_baseline',
        [
            (
                'FrozenLake-v1',
                False,
            ),
            (
                'InvertedPendulum-v5',
                False,
            ),
            (
                'InvertedPendulum-v5',
                True,
            ),
        ],
    )
    @given(
        n_iterations=st.integers(min_value=2, max_value=5),
        n_episodes=st.integers(min_value=10, max_value=100),
        alpha=st.floats(min_value=0.00001, max_value=0.0001),
    )
    @settings(deadline=5000)
    def test_train_policy_gradient_returns_policy(
        self,
        env_id: str,
        use_baseline: bool,
        n_iterations: int,
        n_episodes: int,
        alpha: float,
    ):
        """
        Test that train_policy_gradient executes successfully and returns a policy.

        Args:
            env_id: The Gym environment ID to be used in training.
            use_baseline: A Boolean indicating whether to use a linear baseline.
            n_iterations: The number of policy updates to perform.
            n_episodes: The number of episodes to sample during each policy update.
            alpha: The initial step size for stochastic gradient ascent.

        """
        env = gym.make(env_id)

        if env_id == 'InvertedPendulum-v5':
            policy = DenseNetworkPolicy(
                env_id=env_id,
                hidden_space_dims=[16, 32],
            )
        else:
            feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
            policy = LinearSoftMax(env_id, feature_fn)

        if use_baseline:
            baseline = LinearBaseline(env_id=env_id)
        else:
            baseline = None

        # Train the policy
        trained_policy = train_policy_gradient(
            env_id=env_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            alpha=alpha,
            baseline=baseline,
        )

        # Verify that a policy is returned
        assert trained_policy is not None
        if env_id == 'InvertedPendulum-v5':
            assert isinstance(trained_policy, DenseNetworkPolicy)
        else:
            assert isinstance(trained_policy, LinearSoftMax)

        # Verify that the returned policy is the same object that was passed in
        assert trained_policy is policy

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(
        n_iterations=st.integers(min_value=2, max_value=5),
        n_episodes=st.integers(min_value=10, max_value=100),
        alpha=st.floats(min_value=0.001, max_value=0.01),
    )
    @settings(deadline=5000)
    def test_train_policy_gradient_updates_policy(
        self,
        env_id: str,
        n_iterations: int,
        n_episodes: int,
        alpha: float,
    ):
        """
        Test that train_policy_gradient executes successfully and updates the policy.

        Args:
            env_id: The Gym environment ID to be used in training.
            n_iterations: The number of policy updates to perform.
            n_episodes: The number of episodes to sample during each policy update.
            alpha: The initial step size for stochastic gradient ascent.

        """
        env = gym.make(env_id)

        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        original_paramters = copy.deepcopy(list(policy.get_parameters()))

        # Train the policy - We set reward_schedule to ensure the policy is updated.
        trained_policy = train_policy_gradient(
            env_id=env_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            alpha=alpha,
            is_slippery=False,
            reward_schedule=(1, 1, 1),
        )

        # Verify that a policy is returned
        assert trained_policy is not None
        assert isinstance(trained_policy, LinearSoftMax)

        # Verify that the returned policy is the same object that was passed in
        assert trained_policy is policy

        updated_parameters = list(policy.get_parameters())
        parameter_diff = original_paramters[0].detach().numpy() - updated_parameters[0].detach().numpy()
        assert np.sum(np.abs(parameter_diff)) > 0

    def test_frozen_lake_regression_test(self):
        """Perform a regression test to ensure that a policy can still be optimised on FrozenLake."""
        env_id = 'FrozenLake-v1'
        n_iterations = 100
        n_episodes = 100
        alpha = 1.0

        env = gym.make(env_id)

        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)

        trained_policy = train_policy_gradient(
            env_id=env_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            alpha=alpha,
            is_slippery=False,
        )

        statistics_collector = EpisocidPolicyGradientStatisticsCollector(env_id)
        sampler = EpisodicSampler(
            env_id=env_id,
            n_episodes=n_episodes,
            policy=trained_policy,
            statistics_collector=statistics_collector,
            is_slippery=False,
        )

        statistics = sampler.sample()
        assert np.average(statistics.total_reward) > 0.8

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'env_id, use_baseline',
        [
            (
                'FrozenLake-v1',
                False,
            ),
            (
                'InvertedPendulum-v5',
                False,
            ),
            (
                'InvertedPendulum-v5',
                True,
            ),
        ],
    )
    @given(
        n_iterations=st.integers(min_value=2, max_value=5),
        n_episodes=st.integers(min_value=10, max_value=20),
        alpha=st.floats(min_value=0.0001, max_value=0.001),
    )
    @settings(deadline=7000)
    def test_ray_train_policy_gradient_returns_policy(
        self,
        env_id: str,
        use_baseline: bool,
        n_iterations: int,
        n_episodes: int,
        alpha: float,
        test_ray_cluster,
    ):
        """
        Test that train_policy_gradient executes successfully and returns a policy.

        Args:
            env_id: The Gym environment ID to be used in training.
            use_baseline: A Boolean indicating whether to use a linear baseline.
            n_iterations: The number of policy updates to perform.
            n_episodes: The number of episodes to sample during each policy update.
            alpha: The initial step size for stochastic gradient ascent.
            test_ray_cluster: Test Ray cluster.

        """
        env = gym.make(env_id)
        n_samplers = 2
        if env_id == 'InvertedPendulum-v5':
            policy = DenseNetworkPolicy(
                env_id=env_id,
                hidden_space_dims=[16, 32],
            )
        else:
            feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
            policy = LinearSoftMax(env_id, feature_fn)

        if use_baseline:
            baseline = LinearBaseline(env_id=env_id)
        else:
            baseline = None

        # Train the policy
        trained_policy = train_policy_gradient(
            env_id=env_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            alpha=alpha,
            n_samplers=n_samplers,
            baseline=baseline,
        )

        # Verify that a policy is returned
        assert trained_policy is not None
        if env_id == 'InvertedPendulum-v5':
            assert isinstance(trained_policy, DenseNetworkPolicy)
        else:
            assert isinstance(trained_policy, LinearSoftMax)

        # Verify that the returned policy is the same object that was passed in
        assert trained_policy is policy
