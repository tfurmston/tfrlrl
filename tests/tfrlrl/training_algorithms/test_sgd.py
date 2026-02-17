import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.training_algorithms.sgd import train_policy_gradient


class TestTrainPolicyGradient:
    """Class that encapsulates the unit tests for the train_policy_gradient function."""

    @pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
    @given(
        n_iterations=st.integers(min_value=2, max_value=10),
        n_episodes=st.integers(min_value=10, max_value=100),
        alpha=st.floats(min_value=0.0001, max_value=0.001),
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

        Args:
            env_id: The Gym environment ID to be used in training.
            n_iterations: The number of policy updates to perform.
            n_episodes: The number of episodes to sample during each policy update.
            alpha: The initial step size for stochastic gradient ascent.

        """
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[16, 32],
        )

        # Train the policy
        trained_policy = train_policy_gradient(
            env_id=env_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            alpha=alpha,
        )

        # Verify that a policy is returned
        assert trained_policy is not None
        assert isinstance(trained_policy, DenseNetworkPolicy)

        # Verify that the returned policy is the same object that was passed in
        assert trained_policy is policy

    @pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
    @given(
        n_iterations=st.integers(min_value=2, max_value=10),
        n_episodes=st.integers(min_value=10, max_value=100),
        alpha=st.floats(min_value=0.0001, max_value=0.001),
    )
    @settings(deadline=5000)
    def test_ray_train_policy_gradient_returns_policy(
        self,
        env_id: str,
        n_iterations: int,
        n_episodes: int,
        alpha: float,
        test_ray_cluster,
    ):
        """
        Test that train_policy_gradient executes successfully and returns a policy.

        Args:
            env_id: The Gym environment ID to be used in training.
            n_iterations: The number of policy updates to perform.
            n_episodes: The number of episodes to sample during each policy update.
            alpha: The initial step size for stochastic gradient ascent.
            test_ray_cluster: Test Ray cluster.

        """
        n_samplers = 2
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[16, 32],
        )

        # Train the policy
        trained_policy = train_policy_gradient(
            env_id=env_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            alpha=alpha,
            n_samplers=n_samplers,
        )

        # Verify that a policy is returned
        assert trained_policy is not None
        assert isinstance(trained_policy, DenseNetworkPolicy)

        # Verify that the returned policy is the same object that was passed in
        assert trained_policy is policy
