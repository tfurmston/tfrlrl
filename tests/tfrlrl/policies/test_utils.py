import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.policies.utils import flatten_tensor_dict, unflatten_tensor_dict


@pytest.mark.parametrize('env_id', ['FrozenLake-v1', 'InvertedPendulum-v5'])
def test_flatten_tensor_dict(env_id: str):
    """
    Test flatten_tensor_dict function on policy parameters.

    This function tests that flatten_tensor_dict flattens policy parameters and maintains
    the parameter ordering.

    Args:
        env_id: The environment I.D. on which to run the test. This is used to determine the policy on which
        to perform the test.

    """
    if env_id == 'FrozenLake-v1':
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)
    elif env_id == 'InvertedPendulum-v5':
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[16, 32],
        )
    else:
        raise ValueError('Unexpected environment: %s', env_id)

    flattend_tensor = flatten_tensor_dict(
        {name: param for name, param in policy.network.named_parameters()},
    )

    indx = 0
    for _, param in policy.network.named_parameters():
        for p_indx in np.ndindex(param.shape):
            assert param[p_indx] == flattend_tensor[indx]
            indx += 1

    assert indx == flattend_tensor.shape[0]


@pytest.mark.parametrize('env_id', ['FrozenLake-v1', 'InvertedPendulum-v5'])
def test_unflatten_tensor_dict(env_id: str):
    """
    Test unflatten_tensor_dict function on policy parameters.

    This function tests that unflatten_tensor_dict is the inverse of flatten_tensor_dict, i.e. that
    flattening a dictionary of policy parameters and then unflattening the result recovers the
    original dictionary of tensors.

    Args:
        env_id: The environment I.D. on which to run the test. This is used to determine the policy on which
        to perform the test.

    """
    if env_id == 'FrozenLake-v1':
        env = gym.make(env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)
    elif env_id == 'InvertedPendulum-v5':
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[16, 32],
        )
    else:
        raise ValueError('Unexpected environment: %s', env_id)

    reference = {name: param for name, param in policy.network.named_parameters()}
    flattend_tensor = flatten_tensor_dict(reference)
    unflattened = unflatten_tensor_dict(flattend_tensor, reference)

    assert list(unflattened.keys()) == list(reference.keys())
    for name, param in reference.items():
        assert unflattened[name].shape == param.shape
        assert (unflattened[name] == param).all()


@pytest.mark.parametrize(
    'env_id, extend_actions',
    [
        (
            'CliffWalking-v1',
            False,
        ),
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
    n_observations=st.integers(min_value=5, max_value=10),
)
@settings(deadline=None)
def test_flatten_tensor_dict_jacobian(env_id: str, extend_actions, n_observations: int):
    """
    Test flatten_tensor_dict function on the policy Jacobian.

    This function tests that flatten_tensor_dict flattens policy Jacobian and maintains
    the parameter ordering.

    Args:
        env_id: The environment I.D. on which to run the test. This is used to determine the policy on which
        to perform the test.
        extend_actions: A Boolean indicating whether to extend the action dimensions. This is used to capture
        the case of a 1-dimensional conintuous action space, in which the actions could be stored as either
        a (n_observations) or an (1, n_observations) array.
        n_observations: The number of observations.

    """
    env = gym.make(env_id)

    if env_id == 'CliffWalking-v1':
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)
        observations = np.random.randint(low=0, high=47, size=(1, n_observations))
        actions = np.random.randint(low=0, high=3, size=(1, n_observations))
    elif env_id == 'InvertedPendulum-v5':
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[16, 32],
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
    else:
        raise ValueError('Unexpected environment: %s', env_id)

    jacobian = policy.calculate_jacobian(observations, actions)
    flattend_jacobian = flatten_tensor_dict(
        jacobian,
        dim=actions.ndim,
    )

    indx = 0
    for _, jac_param in jacobian.items():
        param_shape = tuple(jac_param.shape[actions.ndim :])
        for p_indx in np.ndindex(param_shape):
            for i in range(n_observations):
                if actions.ndim == 1:
                    assert jac_param[(i,) + p_indx] == flattend_jacobian[i, indx]
                else:
                    assert jac_param[(0, i) + p_indx] == flattend_jacobian[0, i, indx]
            indx += 1
    assert indx == flattend_jacobian.shape[actions.ndim]


@pytest.mark.parametrize(
    'env_id, extend_actions',
    [
        (
            'CliffWalking-v1',
            False,
        ),
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
    n_observations=st.integers(min_value=5, max_value=10),
)
@settings(deadline=None)
def test_unflatten_tensor_dict_jacobian(env_id: str, extend_actions, n_observations: int):
    """
    Test unflatten_tensor_dict function on the policy Jacobian.

    This function tests that unflatten_tensor_dict is the inverse of flatten_tensor_dict, i.e. that
    flattening the policy Jacobian and then unflattening the result recovers the original dictionary
    of Jacobian tensors.

    Args:
        env_id: The environment I.D. on which to run the test. This is used to determine the policy on which
        to perform the test.
        extend_actions: A Boolean indicating whether to extend the action dimensions. This is used to capture
        the case of a 1-dimensional conintuous action space, in which the actions could be stored as either
        a (n_observations) or an (1, n_observations) array.
        n_observations: The number of observations.

    """
    env = gym.make(env_id)

    if env_id == 'CliffWalking-v1':
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(env_id, feature_fn)
        observations = np.random.randint(low=0, high=47, size=(1, n_observations))
        actions = np.random.randint(low=0, high=3, size=(1, n_observations))
    elif env_id == 'InvertedPendulum-v5':
        policy = DenseNetworkPolicy(
            env_id=env_id,
            hidden_space_dims=[16, 32],
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
    else:
        raise ValueError('Unexpected environment: %s', env_id)

    jacobian = policy.calculate_jacobian(observations, actions)
    flattend_jacobian = flatten_tensor_dict(
        jacobian,
        dim=actions.ndim,
    )
    unflattened_jacobian = unflatten_tensor_dict(
        flattend_jacobian,
        jacobian,
        dim=actions.ndim,
    )

    assert list(unflattened_jacobian.keys()) == list(jacobian.keys())
    for name, jac_param in jacobian.items():
        assert unflattened_jacobian[name].shape == jac_param.shape
        assert (unflattened_jacobian[name] == jac_param).all()
