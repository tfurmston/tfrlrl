import copy

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import (
    Tensor,
    sum,
)
from torch.optim import (
    SGD,
)

from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.linear_soft_max import LinearSoftMax


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_linear_softmax_policy_init_with_discrete_action_space(env_id: str):
    """
    Test that LinearSoftMax can be initialized with discrete action space environments.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.

    """
    env = gym.make(env_id)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, feature_fn)
    assert isinstance(policy, LinearSoftMax)
    assert policy._env.spec.id == env_id


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(observation=st.integers(min_value=0, max_value=47), seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_linear_softmax_policy_generate_action(env_id: str, observation: int, seed: int):
    """
    Test the generate_action function of the LinearSoftMax policy.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        observation: A valid observation (state) from the environment.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, feature_fn)
    action = policy.generate_action(np.array([observation]))

    assert isinstance(action, np.int64)
    assert action.size == 1
    assert action.flat[0] in [0, 1, 2, 3]


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(n_observations=st.integers(min_value=2, max_value=2), seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_linear_softmax_policy_calculate_log_probabilities(env_id: str, n_observations: int, seed: int):
    """
    Test the calculate_log_probabilities function of the LinearSoftMax policy.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        n_observations: The number of observations for which to calculate the log-probabilities.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    observations = np.array([np.random.randint(0, 47) for _ in range(n_observations)])[np.newaxis, :]
    actions = np.array([policy.generate_action(observations[0, i]) for i in range(n_observations)])

    log_probs = policy.calculate_log_probabilities(observations, actions)

    assert isinstance(log_probs, Tensor)
    assert log_probs.shape == (n_observations,)
    np.testing.assert_array_less(log_probs.detach().numpy(), 0)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(observation=st.integers(min_value=0, max_value=47), seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_linear_softmax_policy_action_probabilities_sum_to_one(env_id: str, observation: int, seed: int):
    """
    Test action probabilities sum to one for single observations and parameter values.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        observation: A valid observation (state) from the environment.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    observation = np.array([observation])

    action_probs = policy.calculate_action_distribution(observation).probs.detach().numpy()
    action_probs_from_network = policy.network(policy.construct_network_input(observation)).squeeze()

    # Check that probabilities are consistent with those obtained from the network
    np.testing.assert_allclose(
        action_probs,
        action_probs_from_network.detach().numpy(),
        rtol=1e-6,
        atol=1e-9,
    )

    # Check that probabilities sum to 1.0 (within numerical tolerance)
    np.testing.assert_allclose(np.sum(action_probs), 1.0, rtol=1e-6, atol=1e-9)

    # Check that all probabilities are non-negative
    assert np.all(action_probs >= 0.0)

    # Check that all probabilities are <= 1.0
    assert np.all(action_probs <= 1.0)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    n_observations=st.integers(min_value=2, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_linear_softmax_policy_action_probabilities_sum_to_one_multiple_observations(
    env_id: str, n_observations: int, seed: int
):
    """
    Test action probabilities sum to one for multiple observations and parameter values.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        n_observations: The number of observations for which to calculate the probabilities.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    observations = np.random.randint(low=0, high=47, size=(1, n_observations))

    action_probs = policy.calculate_action_distribution(observations).probs.detach().numpy()
    action_probs_from_network = policy.network(policy.construct_network_input(observations)).squeeze()

    # Check that probabilities are consistent with those obtained from the network
    np.testing.assert_allclose(
        action_probs,
        action_probs_from_network.detach().numpy(),
        rtol=1e-6,
        atol=1e-9,
    )

    # Check that probabilities sum to 1.0 (within numerical tolerance)
    np.testing.assert_allclose(np.sum(action_probs, axis=1), np.ones(shape=(n_observations)), rtol=1e-6, atol=1e-9)

    # Check that all probabilities are non-negative
    assert np.all(action_probs >= 0.0)

    # Check that all probabilities are <= 1.0
    assert np.all(action_probs <= 1.0)


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    observation=st.integers(min_value=0, max_value=47),
    action=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_linear_softmax_policy_log_probability(env_id: str, observation: int, action: int, seed: int):
    """
    Test calculate_log_probabilities returns log-probabilities of the given action for single observation.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        observation: A valid observation (state) from the environment.
        action: A valid action from the environment.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    action = np.array([action])
    observation = np.array([observation])

    action_probs = policy.calculate_action_distribution(observation).probs.detach().numpy()
    log_probability = policy.calculate_log_probabilities(observation, action).detach().numpy()

    np.testing.assert_allclose(
        np.log(action_probs[action]),
        log_probability,
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    n_observations=st.integers(min_value=2, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_linear_softmax_policy_log_probabilities(env_id: str, n_observations: int, seed: int):
    """
    Test that calculate_log_probabilities returns the log-probabilities for multiple observations.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        n_observations: The number of observations for which to calculate the log-probabilities.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    observations = np.random.randint(low=0, high=47, size=(1, n_observations))
    actions = np.random.randint(low=0, high=3, size=(1, n_observations))

    action_probs = policy.calculate_action_distribution(observations).probs.detach().numpy()
    log_probabilities = policy.calculate_log_probabilities(observations, actions).detach().numpy()

    for n in range(n_observations):
        np.testing.assert_allclose(
            np.log(action_probs[n, actions[0, n]]),
            log_probabilities[0, n],
            rtol=1e-6,
            atol=1e-9,
        )


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    observation=st.integers(min_value=0, max_value=47),
    action=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_make_log_prob_fn_single_observation(env_id: str, observation: int, action: int, seed: int):
    """
    Test make_log_prob_fn returns function that can be used to calculate log-probability of given state-action pair.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        observation: A valid observation (state) from the environment.
        action: A valid action from the environment.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    action = np.array([action])
    observation = np.array([observation])

    log_probability = policy.calculate_log_probabilities(observation, action).detach().numpy()
    log_prob_fn, params = policy.make_log_prob_fn(observation, action)
    log_probability_from_functional = log_prob_fn(params).detach().numpy()
    np.testing.assert_allclose(
        log_probability,
        log_probability_from_functional,
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    n_observations=st.integers(min_value=2, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_make_log_prob_fn_multiple_observations(env_id: str, n_observations: int, seed: int):
    """
    Test make_log_prob_fn returns function that can be used to calculate log-probabilities of given state-action pairs.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        n_observations: The number of observations for which to calculate the log-probabilities.
        seed: Random seed for generating softmax parameters.

    """
    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    observations = np.random.randint(low=0, high=47, size=(1, n_observations))
    actions = np.random.randint(low=0, high=3, size=(1, n_observations))

    log_probabilities = policy.calculate_log_probabilities(observations, actions).detach().numpy()
    log_prob_fn, params = policy.make_log_prob_fn(observations, actions)
    log_probability_from_functional = log_prob_fn(params).detach().numpy()

    for n in range(n_observations):
        np.testing.assert_allclose(
            log_probabilities[0, n],
            log_probability_from_functional[0, n],
            rtol=1e-6,
            atol=1e-9,
        )


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    observation=st.integers(min_value=0, max_value=47),
    action=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_linear_softmax_policy_log_probabilities_derivatives(env_id: str, observation: int, action: int, seed: int):
    """
    Test calculation of the derivativres of the log-probabilities of the policy.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        observation: A valid observation (state) from the environment.
        action: A valid action from the environment.
        seed: Random seed for generating softmax parameters.

    """
    eps = 0.01

    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    action = np.array([action])
    observation = np.array([observation])

    policy_dict = copy.deepcopy(policy.get_state())
    n_weights = policy_dict['network.linear.weight'].shape[1]

    df_log_probs_finite_diffs = np.zeros((1, n_weights))

    for n in range(n_weights):
        new_policy_dict_plus = copy.deepcopy(policy_dict)
        new_policy_dict_minus = copy.deepcopy(policy_dict)

        new_policy_dict_plus['network.linear.weight'][0, n] += eps
        policy.set_state(new_policy_dict_plus)
        log_probability_plus = policy.calculate_log_probabilities(observation, action).detach().numpy()

        new_policy_dict_minus['network.linear.weight'][0, n] -= eps
        policy.set_state(new_policy_dict_minus)
        log_probability_minus = policy.calculate_log_probabilities(observation, action).detach().numpy()

        df_log_probs_finite_diffs[0, n] = 0.5 * (log_probability_plus - log_probability_minus) / eps

    optimizer = SGD(policy.get_parameters())
    optimizer.zero_grad()
    loss = policy.calculate_log_probabilities(observation, action)
    loss.backward()

    df_log_probs = list(policy.get_parameters())[0].grad.detach().numpy()

    np.testing.assert_almost_equal(
        df_log_probs,
        df_log_probs_finite_diffs,
        decimal=2,
    )


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    n_observations=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(deadline=None)
def test_linear_softmax_policy_log_probabilities_derivatives_multiple_observations(
    env_id: str, n_observations: int, seed: int
):
    """
    Test calculation of the derivativres of the log-probabilities of the policy with multiple observations.

    Args:
        env_id: The Gymnasium environment ID with a discrete action space.
        n_observations: The number of observations to sample from the environment.
        seed: Random seed for generating softmax parameters.

    """
    eps = 0.01

    env = gym.make(env_id)
    np.random.seed(seed)
    feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
    policy = LinearSoftMax(env_id, feature_fn)

    observations = np.random.randint(low=0, high=47, size=(1, n_observations))
    actions = np.random.randint(low=0, high=3, size=(1, n_observations))

    policy_dict = copy.deepcopy(policy.get_state())
    n_weights = policy_dict['network.linear.weight'].shape[1]

    df_log_probs_finite_diffs = np.zeros((1, n_weights))

    for i in range(n_observations):
        for n in range(n_weights):
            new_policy_dict_plus = copy.deepcopy(policy_dict)
            new_policy_dict_minus = copy.deepcopy(policy_dict)

            new_policy_dict_plus['network.linear.weight'][0, n] += eps
            policy.set_state(new_policy_dict_plus)
            log_probability_plus = (
                policy.calculate_log_probabilities(
                    observations[:, i],
                    actions[:, i],
                )
                .detach()
                .numpy()
            )

            new_policy_dict_minus['network.linear.weight'][0, n] -= eps
            policy.set_state(new_policy_dict_minus)
            log_probability_minus = (
                policy.calculate_log_probabilities(
                    observations[:, i],
                    actions[:, i],
                )
                .detach()
                .numpy()
            )

            df_log_probs_finite_diffs[0, n] += 0.5 * (log_probability_plus - log_probability_minus) / eps

    optimizer = SGD(policy.get_parameters())
    optimizer.zero_grad()
    loss = sum(
        policy.calculate_log_probabilities(
            observations=observations,
            actions=actions,
        )
    )
    loss.backward()

    df_log_probs = list(policy.get_parameters())[0].grad.detach().numpy()

    np.testing.assert_almost_equal(
        df_log_probs,
        df_log_probs_finite_diffs,
        decimal=2,
    )
