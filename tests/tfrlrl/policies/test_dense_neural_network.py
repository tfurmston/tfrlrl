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
        assert log_probabilities.shape == (n_observations, 1)
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
