import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import construct_one_hot_feature_function
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.training_algorithms.sgd import train_policy_gradient


class TestTrainPolicyGradient:
    """Class that encapsulates the unit tests for the train_policy_gradient function."""

    @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    @given(
        n_iterations=st.integers(min_value=2, max_value=10),
        n_episodes=st.integers(min_value=10, max_value=100),
        alpha=st.floats(min_value=1.0, max_value=10.0),
    )
    @settings(deadline=5000)
    def test_train_policy_gradient_returns_policy(
        self,
        env_id: str,
        n_iterations: int,
        n_episodes: int,
        alpha: float,
    ):
        """
        Test that train_policy_gradient executes successfully and returns a policy.

        :param env_id: The Gym environment ID to be used in training.
        :param n_iterations: The number of policy updates to perform.
        :param n_episodes: The number of episodes to sample during each policy update.
        :param alpha: The initial step size for stochastic gradient ascent.
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

        # Store initial parameters to verify they change during training
        initial_parameters = pol.get_parameters().copy()

        # Train the policy
        trained_policy = train_policy_gradient(
            env_id=env_id,
            policy=pol,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            alpha=alpha,
            is_slippery=False,
        )

        # Verify that a policy is returned
        assert trained_policy is not None
        assert isinstance(trained_policy, LinearSoftMax)

        # Verify that the returned policy is the same object that was passed in
        assert trained_policy is pol

        # Verify that the policy parameters have been updated during training
        final_parameters = trained_policy.get_parameters()
        assert initial_parameters.shape == final_parameters.shape

    # @pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
    # @given(
    #     n_iterations=st.integers(min_value=2, max_value=10),
    #     n_episodes=st.integers(min_value=10, max_value=100),
    #     alpha=st.floats(min_value=1.0, max_value=10.0),
    # )
    # @settings(deadline=5000)
    # def test_ray_train_policy_gradient_returns_policy(
    #     self,
    #     env_id: str,
    #     n_iterations: int,
    #     n_episodes: int,
    #     alpha: float,
    #     test_ray_cluster,
    # ):
    #     """
    #     Test that train_policy_gradient executes successfully and returns a policy.

    #     :param env_id: The Gym environment ID to be used in training.
    #     :param n_iterations: The number of policy updates to perform.
    #     :param n_episodes: The number of episodes to sample during each policy update.
    #     :param alpha: The initial step size for stochastic gradient ascent.
    #     :param test_ray_cluster: Test Ray cluster.
    #     """
    #     n_samplers = 2
    #     env = gym.make(env_id)
    #     S = env.observation_space.n
    #     A = env.action_space.n

    #     feature_fn = construct_one_hot_feature_function(S=S, A=A)
    #     softmax_parameters = np.random.random(size=S * (A - 1))
    #     pol = LinearSoftMax(
    #         env_id,
    #         softmax_parameters,
    #         feature_fn,
    #     )

    #     # Store initial parameters to verify they change during training
    #     initial_parameters = pol.get_parameters().copy()

    #     # Train the policy
    #     trained_policy = train_policy_gradient(
    #         env_id=env_id,
    #         policy=pol,
    #         n_iterations=n_iterations,
    #         n_episodes=n_episodes,
    #         alpha=alpha,
    #         n_samplers=n_samplers,
    #         is_slippery=False,
    #     )

    #     # Verify that a policy is returned
    #     assert trained_policy is not None
    #     assert isinstance(trained_policy, LinearSoftMax)

    #     # Verify that the returned policy is the same object that was passed in
    #     assert trained_policy is pol

    #     # Verify that the policy parameters have been updated during training
    #     final_parameters = trained_policy.get_parameters()
    #     assert initial_parameters.shape == final_parameters.shape
