import copy
from typing import Tuple

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import (
    sum as torch_sum,
)
from torch import (
    tensor,
)
from torch.optim import (
    SGD,
)

from tfrlrl.baselines.linear import LinearBaseline
from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import EpisodicSampler
from tfrlrl.sampling.statistics_collection import EpisocidPolicyGradientStatisticsCollector
from tfrlrl.training_algorithms.tnpg import (
    calculate_steepest_gradient_direction,
    construct_fim_vector_product_fn,
    train_policy_gradient,
)


@pytest.mark.parametrize(
    'env_id, expected_shape',
    [
        ('FrozenLake-v1', (49,)),
        ('InvertedPendulum-v5', (690,)),
    ],
)
@given(
    n_episodes=st.integers(min_value=10, max_value=100),
)
@settings(deadline=None)
def test_calculate_steepest_gradient_direction(env_id: str, expected_shape: Tuple[int], n_episodes: int):
    """
    Test that calculate_steepest_gradient_direction returns the correct policy gradient.

    The steepest gradient direction is verified against central finite differences of the loss function
    L(θ) = -sum_t log π_θ(a_t | s_t) * R_t with respect to each policy parameter element.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        expected_shape: The expected shape of the gradient.
        n_episodes: The number of episodes to sample when estimating the gradient.

    """
    env = gym.make(env_id)
    statistics_collector = EpisocidPolicyGradientStatisticsCollector(env_id)

    if env_id == 'FrozenLake-v1':
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)
        sampler = EpisodicSampler(
            env_id=env_id,
            n_episodes=n_episodes,
            policy=policy,
            statistics_collector=statistics_collector,
            is_slippery=False,
        )
    elif env_id == 'InvertedPendulum-v5':
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[16, 32],
        )
        sampler = EpisodicSampler(
            env_id=env_id,
            n_episodes=n_episodes,
            policy=policy,
            statistics_collector=statistics_collector,
        )
    else:
        raise ValueError('Unexpected environment: %s', env_id)

    alpha = 0.1
    optimizer = SGD(policy.get_parameters(), lr=alpha)
    statistics = sampler.sample()
    sgd = calculate_steepest_gradient_direction(
        policy,
        statistics,
        optimizer,
    )
    assert isinstance(sgd, np.ndarray)
    assert sgd.shape == expected_shape

    eps = 0.001
    policy_dict = copy.deepcopy(policy.get_state())

    fd_grad_parts = []
    for param_name, param_value in policy_dict.items():
        param_shape = tuple(param_value.shape)
        fd_grad = np.zeros(param_shape)

        for idx in np.ndindex(param_shape):
            dict_plus = copy.deepcopy(policy_dict)
            dict_minus = copy.deepcopy(policy_dict)

            dict_plus[param_name][idx] += eps
            policy.set_state(dict_plus)
            log_probs_plus = policy.calculate_log_probabilities(statistics.observations, statistics.actions)
            loss_plus = (-torch_sum(log_probs_plus * tensor(statistics.total_expected_rewards))).item()

            dict_minus[param_name][idx] -= eps
            policy.set_state(dict_minus)
            log_probs_minus = policy.calculate_log_probabilities(statistics.observations, statistics.actions)
            loss_minus = (-torch_sum(log_probs_minus * tensor(statistics.total_expected_rewards))).item()

            fd_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        fd_grad_parts.append(fd_grad.flatten())

    policy.set_state(policy_dict)

    fd_gradient = np.concatenate(fd_grad_parts)

    np.testing.assert_almost_equal(
        sgd,
        fd_gradient,
        decimal=2,
    )


@pytest.mark.parametrize(
    'env_id, expected_shape',
    [
        ('FrozenLake-v1', (49,)),
        ('InvertedPendulum-v5', (50,)),
    ],
)
@given(
    n_episodes=st.integers(min_value=2, max_value=10),
)
@settings(deadline=None)
def test_construct_fim_vector_product_fn(env_id: str, expected_shape: Tuple[int], n_episodes: int):
    """
    Test that construct_fim_vector_product_fn returns a function for calculate the FIM-vector product.

    The production of the Fisher Information matrix (FIM) against a vector will be verified by
    performing the multiplication directly.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        expected_shape: The expected shape of the gradient.
        n_episodes: The number of episodes to sample when estimating the gradient.

    """
    env = gym.make(env_id)
    statistics_collector = EpisocidPolicyGradientStatisticsCollector(env_id)

    if env_id == 'FrozenLake-v1':
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)
        sampler = EpisodicSampler(
            env_id=env_id,
            n_episodes=n_episodes,
            policy=policy,
            statistics_collector=statistics_collector,
            is_slippery=False,
        )
    elif env_id == 'InvertedPendulum-v5':
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[4, 4],
        )
        sampler = EpisodicSampler(
            env_id=env_id,
            n_episodes=n_episodes,
            policy=policy,
            statistics_collector=statistics_collector,
        )
    else:
        raise ValueError('Unexpected environment: %s', env_id)

    alpha = 0.1
    optimizer = SGD(policy.get_parameters(), lr=alpha)
    statistics = sampler.sample()
    sgd = calculate_steepest_gradient_direction(
        policy,
        statistics,
        optimizer,
    )
    assert isinstance(sgd, np.ndarray)
    assert sgd.shape == expected_shape

    calculate_fim_vector_product = construct_fim_vector_product_fn(
        policy=policy,
        statistics=statistics,
    )

    v = calculate_fim_vector_product(sgd)

    jacobian = policy.calculate_jacobian(
        statistics.observations,
        statistics.actions,
    )

    jac_blocks = []
    for _, jac_param in jacobian.items():
        n_steps = jac_param.shape[1]
        jac_block = jac_param.detach().numpy()[0].reshape(n_steps, -1)
        jac_blocks.append(jac_block)
    J = np.concatenate(jac_blocks, axis=1)

    v_expected = np.matmul(J.T, np.matmul(J, sgd))

    np.testing.assert_allclose(v, v_expected, rtol=1e-1)


def test_construct_fim_vector_product_fn_subsamples_state_action_pairs(monkeypatch):
    """
    Test that construct_fim_vector_product_fn only uses a subsample of state-action pairs when n_samples_fim is given.

    The subsample of state-action pairs used within the calculation is verified by mocking the random number
    generator used to select the subsample indices, and comparing the result directly against the Fisher
    Information matrix-vector product calculated using only the observations and actions at those indices.

    Args:
        monkeypatch: The PyTest monkeypatch fixture, used to control the random subsample of indices selected.

    """
    env_id = 'FrozenLake-v1'
    n_episodes = 10
    env = gym.make(env_id)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)
    statistics_collector = EpisocidPolicyGradientStatisticsCollector(env_id)
    sampler = EpisodicSampler(
        env_id=env_id,
        n_episodes=n_episodes,
        policy=policy,
        statistics_collector=statistics_collector,
        is_slippery=False,
    )
    statistics = sampler.sample()

    n_steps = statistics.actions.shape[-1]
    n_samples_fim = max(1, n_steps // 2)
    idx = np.arange(n_samples_fim)

    class _FakeRNG:
        def choice(self, n, size, replace):
            return idx

    monkeypatch.setattr(np.random, 'default_rng', lambda: _FakeRNG())

    alpha = 0.1
    optimizer = SGD(policy.get_parameters(), lr=alpha)
    sgd = calculate_steepest_gradient_direction(
        policy,
        statistics,
        optimizer,
    )

    calculate_fim_vector_product = construct_fim_vector_product_fn(
        policy=policy,
        statistics=statistics,
        n_samples_fim=n_samples_fim,
    )
    v = calculate_fim_vector_product(sgd)

    jacobian = policy.calculate_jacobian(
        statistics.observations[..., idx],
        statistics.actions[..., idx],
    )
    jac_blocks = []
    for _, jac_param in jacobian.items():
        n_steps_sub = jac_param.shape[1]
        jac_block = jac_param.detach().numpy()[0].reshape(n_steps_sub, -1)
        jac_blocks.append(jac_block)
    J = np.concatenate(jac_blocks, axis=1)

    v_expected = np.matmul(J.T, np.matmul(J, sgd))

    np.testing.assert_allclose(v, v_expected, rtol=1e-6)


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
    n_episodes=st.integers(min_value=2, max_value=5),
    lr=st.floats(min_value=0.0000001, max_value=0.000001),
)
@settings(deadline=5000)
def test_train_policy_gradient_returns_policy(
    env_id: str,
    use_baseline: bool,
    n_iterations: int,
    n_episodes: int,
    lr: float,
):
    """
    Test that train_policy_gradient executes successfully and returns a policy.

    Args:
        env_id: The Gym environment ID to be used in training.
        use_baseline: A Boolean indicating whether to use a linear baseline.
        n_iterations: The number of policy updates to perform.
        n_episodes: The number of episodes to sample during each policy update.
        lr: The base learning rate for the SGD optimizer used to apply the natural policy gradient.

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

    trained_policy = train_policy_gradient(
        env_id=env_id,
        policy=policy,
        n_iterations=n_iterations,
        n_episodes=n_episodes,
        lr=lr,
        baseline=baseline,
        n_iters_cg=2,
        n_samples_fim=20,
    )

    assert trained_policy is not None
    if env_id == 'InvertedPendulum-v5':
        assert isinstance(trained_policy, DenseNetworkPolicy)
    else:
        assert isinstance(trained_policy, LinearSoftMax)

    assert trained_policy is policy


@pytest.mark.parametrize('env_id', ['FrozenLake-v1'])
@given(
    n_iterations=st.integers(min_value=2, max_value=5),
    n_episodes=st.integers(min_value=2, max_value=5),
    lr=st.floats(min_value=0.00001, max_value=0.0001),
)
@settings(deadline=5000)
def test_train_policy_gradient_updates_policy(env_id: str, n_iterations: int, n_episodes: int, lr: float):
    """
    Test that train_policy_gradient executes successfully and updates the policy.

    Args:
        env_id: The Gym environment ID to be used in training.
        n_iterations: The number of policy updates to perform.
        n_episodes: The number of episodes to sample during each policy update.
        lr: The base learning rate for the SGD optimizer used to apply the natural policy gradient.

    """
    env = gym.make(env_id)

    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    original_parameters = copy.deepcopy(list(policy.get_parameters()))

    # Train the policy - We set reward_schedule to ensure the policy is updated.
    trained_policy = train_policy_gradient(
        env_id=env_id,
        policy=policy,
        n_iterations=n_iterations,
        n_episodes=n_episodes,
        lr=lr,
        is_slippery=False,
        reward_schedule=(1, 1, 1),
        n_iters_cg=2,
        n_samples_fim=20,
    )

    assert trained_policy is not None
    assert isinstance(trained_policy, LinearSoftMax)
    assert trained_policy is policy

    updated_parameters = list(policy.get_parameters())
    parameter_diff = original_parameters[0].detach().numpy() - updated_parameters[0].detach().numpy()
    assert np.sum(np.abs(parameter_diff)) > 0
