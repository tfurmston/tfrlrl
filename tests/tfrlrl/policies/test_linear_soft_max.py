import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import construct_one_hot_feature_function
from tfrlrl.policies.linear_soft_max import LinearSoftMax


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
def test_init_with_discrete_action_space(env_id: str):
    """
    Test that LinearSoftMax can be initialized with discrete action space environments.

    :param env_id: The Gymnasium environment ID with a discrete action space.
    """
    env = gym.make(env_id)
    softmax_parameters = np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(env.observation_space.n * (env.action_space.n - 1), ),
    )
    feature_fn = construct_one_hot_feature_function(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, softmax_parameters, feature_fn)
    assert isinstance(policy, LinearSoftMax)
    assert policy._env.spec.id == env_id


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    observation=st.integers(min_value=0, max_value=47),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(deadline=None)
def test_action_probabilities_sum_to_one(env_id: str, observation: int, seed: int):
    """
    Test that action probabilities sum to one for various observations and parameter values.

    :param env_id: The Gymnasium environment ID with a discrete action space.
    :param observation: A valid observation (state) from the environment.
    :param seed: Random seed for generating softmax parameters.
    """
    env = gym.make(env_id)
    np.random.seed(seed)
    softmax_parameters = np.random.uniform(
        low=-10.0,
        high=10.0,
        size=(env.observation_space.n * (env.action_space.n - 1), ),
    )
    feature_fn = construct_one_hot_feature_function(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, softmax_parameters, feature_fn)
    action_probs = policy.calculate_action_probabilities(observation)

    # Check that probabilities sum to 1.0 (within numerical tolerance)
    np.testing.assert_allclose(np.sum(action_probs), 1.0, rtol=1e-6, atol=1e-9)

    # Check that all probabilities are non-negative
    assert np.all(action_probs >= 0.0), 'All action probabilities should be non-negative'

    # Check that all probabilities are <= 1.0
    assert np.all(action_probs <= 1.0), 'All action probabilities should be <= 1.0'


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    observation=st.integers(min_value=0, max_value=47),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(deadline=None)
def test_generate_action_returns_valid_actions(env_id: str, observation: int, seed: int):
    """
    Test that generate_action returns valid actions for various policy initializations.

    :param env_id: The Gymnasium environment ID with a discrete action space.
    :param observation: A valid observation (state) from the environment.
    :param seed: Random seed for generating softmax parameters.
    """
    env = gym.make(env_id)
    np.random.seed(seed)
    softmax_parameters = np.random.uniform(
        low=-10.0,
        high=10.0,
        size=(env.observation_space.n * (env.action_space.n - 1), ),
    )
    feature_fn = construct_one_hot_feature_function(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, softmax_parameters, feature_fn)

    # Generate multiple actions to test consistency
    for _ in range(10):
        action = policy.generate_action(observation)

        # Check that action is an integer
        assert isinstance(action, (int, np.integer)), f'Action should be an integer, got {type(action)}'

        # Check that action is within valid range
        assert 0 <= action < env.action_space.n, f'Action {action} is out of valid range [0, {env.action_space.n})'


@pytest.mark.parametrize('env_id', ['CliffWalking-v1'])
@given(
    observation=st.integers(min_value=0, max_value=47),
    action=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=1000)
)
@settings(deadline=None)
def test_calculate_log_derivative_finite_difference(env_id: str, observation: int, action: int, seed: int):
    """
    Test that calculate_log_derivative matches finite difference approximation.

    :param env_id: The Gymnasium environment ID with a discrete action space.
    :param observation: A valid observation (state) from the environment.
    :param action: A valid action from the action space.
    :param seed: Random seed for generating softmax parameters.
    """
    env = gym.make(env_id)
    np.random.seed(seed)
    softmax_parameters = np.random.uniform(
        low=-5.0,
        high=5.0,
        size=(env.observation_space.n * (env.action_space.n - 1), ),
    )
    feature_fn = construct_one_hot_feature_function(env.observation_space.n, env.action_space.n)

    policy = LinearSoftMax(env_id, softmax_parameters, feature_fn)

    # Calculate analytical gradient
    analytical_grad = policy.calculate_log_derivative(observation, action)

    # Calculate numerical gradient using finite differences
    epsilon = 1e-5
    numerical_grad = np.zeros_like(softmax_parameters)

    for i in range(len(softmax_parameters)):
        # Perturb parameter positively
        params_plus = softmax_parameters.copy()
        params_plus[i] += epsilon
        policy_plus = LinearSoftMax(env_id, params_plus, feature_fn)
        probs_plus = policy_plus.calculate_action_probabilities(observation)
        log_prob_plus = np.log(probs_plus[action] + 1e-10)

        # Perturb parameter negatively
        params_minus = softmax_parameters.copy()
        params_minus[i] -= epsilon
        policy_minus = LinearSoftMax(env_id, params_minus, feature_fn)
        probs_minus = policy_minus.calculate_action_probabilities(observation)
        log_prob_minus = np.log(probs_minus[action] + 1e-10)

        # Finite difference approximation
        numerical_grad[i] = (log_prob_plus - log_prob_minus) / (2 * epsilon)

    # Compare analytical and numerical gradients
    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-4,
                               err_msg='Analytical gradient does not match numerical gradient')
