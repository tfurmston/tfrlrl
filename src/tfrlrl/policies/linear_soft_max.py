
from typing import Callable, Tuple, Union

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from tfrlrl.policies.base import BaseDifferentiablePolicy, PolicyException


class LinearSoftMax(BaseDifferentiablePolicy):
    """Linear softmax policy for discrete action spaces.

    This policy computes action probabilities using a linear softmax parameterization with
    feature functions. The policy is differentiable with respect to its parameters, enabling
    gradient-based optimization.

    Args:
        env_id: The Gymnasium environment ID (e.g., 'CliffWalking-v0').
        softmax_parameters: The parameters of the softmax policy.
        feature_fn: A function that maps observations to feature representations.

    Raises:
        PolicyException: If the environment does not have a discrete action space.
    """

    def __init__(self, env_id: str, softmax_parameters: NDArray, feature_fn: Callable[[NDArray], NDArray]):
        """
        Initialize the LinearSoftMax policy.

        :param env_id: The Gymnasium environment ID.
        :param softmax_parameters: The parameters of the softmax policy.
        :param feature_fn: A function that maps observations to feature representations.
        :raises PolicyException: If the environment does not have a discrete action space.
        """
        super().__init__()
        self._env = gym.make(env_id)
        if not isinstance(self._env.action_space, gym.spaces.Discrete):
            raise PolicyException('The LinearSoftMax is applicable to discrete action spaces only.')

        self._softmax_parameters = softmax_parameters
        self._feature_fn = feature_fn

    def calculate_action_probabilities(self, observation: NDArray):
        """
        Calculate the probability distribution over actions for a given observation.

        :param observation: The current state observation from the environment.
        :return: A probability distribution over actions.
        """
        scores = np.sum(np.multiply(self._feature_fn(observation), self._softmax_parameters), axis=1)
        scores -= np.max(scores)
        return np.exp(scores) / np.sum(np.exp(scores))

    def generate_action(self, observation: NDArray) -> Tuple[Union[int, float, NDArray]]:
        """
        Generate an action by sampling from the softmax probability distribution.

        :param observation: The current state observation from the environment.
        :return: A sampled action from the discrete action space.
        """
        return self._env.action_space.sample(probability=self.calculate_action_probabilities(observation))

    def calculate_log_derivative(self, observation: NDArray, action: Tuple[Union[int, float, NDArray]]) -> NDArray:
        """
        Calculate the log derivative of the policy with respect to its parameters.

        :param observation: The state observation from the environment.
        :param action: The action taken in the given observation state.
        :return: The log derivative (gradient) of the policy parameters.
        :raises NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError
