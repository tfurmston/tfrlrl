import copy
from typing import Tuple

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import (
    Tensor,
    tensor,
)
from torch import (
    sum as torch_sum,
)
from torch.optim import (
    SGD,
)

from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import EpisodicSampler
from tfrlrl.sampling.statistics_collection import EpisocidPolicyGradientStatisticsCollector
from tfrlrl.training_algorithms.tnpg import calculate_steepest_gradient_direction


@pytest.mark.parametrize(
    'env_id, expected_shape',
    [
        ('FrozenLake-v1', (1, 49)),
        ('InvertedPendulum-v5', (1, 690)),
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
    assert isinstance(sgd, Tensor)
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

    fd_gradient = np.concatenate(fd_grad_parts)[np.newaxis, :]

    np.testing.assert_almost_equal(
        sgd.detach().numpy(),
        fd_gradient,
        decimal=2,
    )
