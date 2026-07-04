import copy

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import (
    DummyStatisticsCollector,
)
from tfrlrl.policies.base import PolicyException
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.sampling.episodic_sampler import (
    EpisodicSampler,
)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_dense_network_policy_with_discrete_environment(env_id):
    """
    Test the DenseNetworkPolicy throws environment error on discrete domain.

    Args:
        env_id: The environment I.D. from which to sample episodes.

    """
    hidden_space_dims = [16, 32]
    with pytest.raises(PolicyException):
        DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=hidden_space_dims,
        )


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
def test_sample_single_action_from_dense_network_policy(env_id):
    """
    Test sampling a single action with a DenseNetworkPolicy.

    Args:
        env_id: The environment I.D. from which to sample episodes.

    """
    env = gym.make(env_id)

    hidden_space_dims = [16, 32]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    action = policy.generate_action(env.observation_space.sample())
    assert action.shape == (1,)


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
@given(
    n_episodes=st.integers(min_value=0, max_value=20),
    h1_dimensions=st.integers(min_value=8, max_value=32),
    h2_dimensions=st.integers(min_value=24, max_value=48),
)
@settings(deadline=None)
def test_sample_episode_with_dense_network_policy(env_id: str, n_episodes: int, h1_dimensions: int, h2_dimensions: int):
    """
    Test sampling of episodes with a DenseNetworkPolicy.

    Args:
        env_id: The environment I.D. from which to sample episodes.
        n_episodes: The number of episodes to sample.
        h1_dimensions: The number of units in the first hidden layer.
        h2_dimensions: The number of units in the second hidden layer.

    """
    statistics_collector = DummyStatisticsCollector()

    hidden_space_dims = [h1_dimensions, h2_dimensions]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    sampler = EpisodicSampler(
        env_id=env_id,
        statistics_collector=statistics_collector,
        n_episodes=n_episodes,
        policy=policy,
    )
    samples = list(sampler)
    assert len(samples) == n_episodes


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
def test_calculate_log_probabilities_from_dense_network_policy_single_observation(env_id):
    """
    Test calculate_log_probabilities with a single observation with a DenseNetworkPolicy.

    Args:
        env_id: The environment I.D. from which to sample episodes.

    """
    env = gym.make(env_id)

    hidden_space_dims = [16, 32]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    log_probability = policy.calculate_log_probabilities(
        env.observation_space.sample()[..., np.newaxis],
        env.action_space.sample(),
    )
    assert log_probability.shape == (1,)


@pytest.mark.parametrize(
    'env_id, extend_actions',
    [
        (
            'InvertedPendulum-v5',
            True,
        ),
        (
            'InvertedPendulum-v5',
            False,
        ),
    ],
)
@given(
    n_observations=st.integers(min_value=1, max_value=20),
)
@settings(deadline=None)
def test_calculate_log_probabilities_from_dense_network_policy_multiple_observations(
    env_id, n_observations, extend_actions
):
    """
    Test calculate_log_probabilities with multiple observations with a DenseNetworkPolicy.

    Args:
        env_id: The environment I.D. from which to sample episodes.
        n_observations: The number of observations to sample.
        extend_actions: A Boolean indicating whether to extend the diemsions of the actions.

    """
    env = gym.make(env_id)

    hidden_space_dims = [16, 32]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    observations = np.concatenate(
        [env.observation_space.sample()[..., np.newaxis] for _ in range(n_observations)],
        axis=1,
    )
    if extend_actions:
        actions = np.concatenate(
            [env.action_space.sample()[..., np.newaxis] for _ in range(n_observations)],
            axis=1,
        )
    else:
        actions = np.concatenate(
            [env.action_space.sample() for _ in range(n_observations)],
        )

    log_probabilities = policy.calculate_log_probabilities(
        observations,
        actions,
    )

    if extend_actions:
        assert log_probabilities.shape == (1, n_observations)
    else:
        assert log_probabilities.shape == (n_observations,)


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
def test_calculate_log_probabilities_from_functional_single_observation(env_id):
    """
    Test make_log_prob_fn with a single observation with a DenseNetworkPolicy.

    Args:
        env_id: The environment I.D. from which to sample episodes.

    """
    env = gym.make(env_id)

    hidden_space_dims = [16, 32]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    observation = env.observation_space.sample()[..., np.newaxis]
    action = env.action_space.sample()
    log_probability = (
        policy.calculate_log_probabilities(
            observation,
            action,
        )
        .detach()
        .numpy()
    )
    log_prob_fn, params = policy.make_log_prob_fn(
        observation,
        action,
    )
    log_probability_from_functional = log_prob_fn(params).detach().numpy()
    np.testing.assert_allclose(
        log_probability,
        log_probability_from_functional,
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    'env_id, extend_actions',
    [
        (
            'InvertedPendulum-v5',
            True,
        ),
        (
            'InvertedPendulum-v5',
            False,
        ),
    ],
)
@given(
    n_observations=st.integers(min_value=1, max_value=20),
)
@settings(deadline=None)
def test_calculate_log_probabilities_from_functional_multiple_observations(env_id, n_observations, extend_actions):
    """
    Test make_log_prob_fn with multiple observations with a DenseNetworkPolicy.

    Args:
        env_id: The environment I.D. from which to sample episodes.
        n_observations: The number of observations to sample.
        extend_actions: A Boolean indicating whether to extend the diemsions of the actions.

    """
    env = gym.make(env_id)

    hidden_space_dims = [16, 32]
    policy = DenseNetworkPolicy(
        env_id=env_id,
        hidden_space_dims=hidden_space_dims,
    )

    observations = np.concatenate(
        [env.observation_space.sample()[..., np.newaxis] for _ in range(n_observations)],
        axis=1,
    )
    if extend_actions:
        actions = np.concatenate(
            [env.action_space.sample()[..., np.newaxis] for _ in range(n_observations)],
            axis=1,
        )
    else:
        actions = np.concatenate(
            [env.action_space.sample() for _ in range(n_observations)],
        )

    log_probabilities = (
        policy.calculate_log_probabilities(
            observations,
            actions,
        )
        .detach()
        .numpy()
    )
    log_prob_fn, params = policy.make_log_prob_fn(
        observations,
        actions,
    )
    log_probability_from_functional = log_prob_fn(params).detach().numpy()
    np.testing.assert_allclose(
        log_probabilities,
        log_probability_from_functional,
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
@given(
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_dense_neural_network_calculate_jacobian_single_observation(env_id: str, seed: int):
    """
    Test calculate_jacobian for a single state-action pair in the DenseNetworkPolicy.

    Args:
        env_id: The Gymnasium environment ID with a continuous action space.
        seed: Random seed for sampling observations, actions and generating network parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)

    hidden_space_dims = [4, 4]
    policy = DenseNetworkPolicy(env_id=env_id, hidden_space_dims=hidden_space_dims)

    observation = env.observation_space.sample()[..., np.newaxis]
    action = env.action_space.sample()

    jacobian = policy.calculate_jacobian(observation, action)

    eps = 0.001
    policy_dict = copy.deepcopy(policy.get_state())

    for parameter_name, parameter_value in policy_dict.items():
        param_shape = tuple(parameter_value.shape)
        df_finite_diffs = np.zeros(param_shape)

        for idx in np.ndindex(param_shape):
            new_policy_dict_plus = copy.deepcopy(policy_dict)
            new_policy_dict_minus = copy.deepcopy(policy_dict)

            new_policy_dict_plus[parameter_name][idx] += eps
            policy.set_state(new_policy_dict_plus)
            log_prob_plus = policy.calculate_log_probabilities(observation, action).item()

            new_policy_dict_minus[parameter_name][idx] -= eps
            policy.set_state(new_policy_dict_minus)
            log_prob_minus = policy.calculate_log_probabilities(observation, action).item()

            df_finite_diffs[idx] = 0.5 * (log_prob_plus - log_prob_minus) / eps

        np.testing.assert_almost_equal(
            jacobian[parameter_name].squeeze().detach().numpy(),
            df_finite_diffs.squeeze(),
            decimal=2,
        )


@pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
@given(
    n_observations=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_dense_neural_network_calculate_jacobian_multiple_observations(env_id: str, n_observations: int, seed: int):
    """
    Test calculate_jacobian for multiple state-action pairs in the DenseNetworkPolicy.

    Args:
        env_id: The Gymnasium environment ID with a continuous action space.
        n_observations: The number of observations to sample from the environment.
        seed: Random seed for sampling observations, actions and generating network parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)

    hidden_space_dims = [4, 4]
    policy = DenseNetworkPolicy(env_id=env_id, hidden_space_dims=hidden_space_dims)

    observations = np.concatenate(
        [env.observation_space.sample()[..., np.newaxis] for _ in range(n_observations)],
        axis=1,
    )
    actions = np.concatenate(
        [env.action_space.sample() for _ in range(n_observations)],
    )

    jacobian = policy.calculate_jacobian(observations, actions)

    eps = 0.01
    policy_dict = copy.deepcopy(policy.get_state())

    for parameter_name, parameter_value in policy_dict.items():
        param_shape = tuple(parameter_value.shape)
        df_finite_diffs = np.zeros((n_observations, *param_shape))

        for i in range(n_observations):
            obs_i = observations[:, i][..., np.newaxis]
            act_i = actions[i]

            for idx in np.ndindex(param_shape):
                new_policy_dict_plus = copy.deepcopy(policy_dict)
                new_policy_dict_minus = copy.deepcopy(policy_dict)

                new_policy_dict_plus[parameter_name][idx] += eps
                policy.set_state(new_policy_dict_plus)
                log_prob_plus = policy.calculate_log_probabilities(obs_i, act_i).item()

                new_policy_dict_minus[parameter_name][idx] -= eps
                policy.set_state(new_policy_dict_minus)
                log_prob_minus = policy.calculate_log_probabilities(obs_i, act_i).item()

                df_finite_diffs[(i, *idx)] = 0.5 * (log_prob_plus - log_prob_minus) / eps

        np.testing.assert_almost_equal(
            jacobian[parameter_name].squeeze().detach().numpy(),
            df_finite_diffs.squeeze(),
            decimal=2,
        )
