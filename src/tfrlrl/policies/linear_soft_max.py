from collections import OrderedDict

import gymnasium as gym
import numpy.typing as npt
from numpy.typing import NDArray
from torch import (
    Tensor,
    nn,
    tensor,
)
from torch.distributions.multinomial import Categorical

from tfrlrl.features.base import FeatureFunction
from tfrlrl.policies.base import (
    BasePyTorchPolicy,
    PolicyException,
)


class LinearSoftMaxNetwork(nn.Module):
    """
    PyTorch module class for a linear soft-max network.

    The linear soft-max neural network of this class is used to select from a range of discrete actions.
    """

    def __init__(self, n_features: int):
        """
        Initialise linear soft-max neural network.

        param: n_features: The number of features in the feature space.
        """
        super().__init__()

        # Define sequential network for policy
        self.network = nn.Sequential(
            OrderedDict(
                [
                    ('linear', nn.Linear(n_features, 1)),
                    ('softmax', nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(self, x: Tensor) -> tuple[Tensor]:
        """
        Perform a forward pass of the network on the given tensor.

        param: x: An input tensor over which to perform the forward pass.
        return: A Tensor for the action probabilities.
        """
        return self.network(x.float())


class LinearSoftMax(BasePyTorchPolicy):
    """
    Linear softmax policy for discrete action spaces.

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

    def __init__(self, env_id: str, feature_fn: FeatureFunction):
        """
        Initialize the LinearSoftMax policy.

        Args:
            env_id: The Gymnasium environment ID.
            feature_fn: An instance of a feature function.

        Raises:
            :raises PolicyException: If the environment does not have a discrete action space.

        """
        self._env = gym.make(env_id)
        if not isinstance(self._env.action_space, gym.spaces.Discrete):
            raise PolicyException('The LinearSoftMax is applicable to discrete action spaces only.')
        super().__init__(
            network=LinearSoftMaxNetwork(feature_fn.n_features),
        )
        self._feature_fn = feature_fn

    def generate_action(self, observation: NDArray) -> Tensor:
        """
        Generate an action by sampling from the softmax probability distribution.

        Args:
            observation: The current state observation from the environment.

        Returns:
            :return: A sampled action from the discrete action space.

        """
        dist = Categorical(probs=self.network(tensor(self._feature_fn(observation)).T).squeeze())
        return dist.sample().numpy()

    def calculate_log_probabilities(self, observations: npt.NDArray, actions: npt.NDArray) -> Tensor:
        """
        Calculate the log-probailities of the for the given (observation, action) pairs.

        param: observations: The state observations from the environment.
        param: actions: The actions taken in the given observation state.
        return: The log-probabilities of the policy for the given observation-action pairs.
        """
        dist = Categorical(probs=self.network(tensor(self._feature_fn(observations)).T).squeeze())
        return dist.log_prob(tensor(actions))
