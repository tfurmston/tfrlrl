from collections import OrderedDict
from typing import List

import gymnasium as gym
import numpy.typing as npt
from torch import (
    Tensor,
    exp,
    log,
    nn,
    tensor,
)
from torch.distributions.normal import Normal

from tfrlrl.policies.base import BasePyTorchPolicy, PolicyException


class DensePolicyNetwork(nn.Module):
    """
    A dense neural network for calculating the mean and standard deviation of a Gaussian distribution.

    The dense neural network of this class is used to construct the mean and standard deviation of a Gausssian
    policy. This is then used to sample actions in an environment with continuous actions.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int, hidden_space_dims: List[int]):
        """
        Initialise dense neural network for calculating the mean and standard deviation of a Gaussian policy.

        Args:
            obs_space_dims: The number of dimensions in the observation space.
            action_space_dims: The number of dimensions in the action space.
            hidden_space_dims: A list of the number of dimensions for the hidden layers in the network.

        """
        super().__init__()

        # Define the layer dimensions for all the hidden layers.
        layer_dims = [
            (obs_space_dims if n == 0 else hidden_space_dims[n - 1], hidden_space_dims[n])
            for n in range(len(hidden_space_dims))
        ]
        # Define the linear layers for all of the hidden layers.
        lin_layers = [
            (f'lin_{n}', nn.Linear(layer_dims[n][0], layer_dims[n][1])) for n in range(len(hidden_space_dims))
        ]
        # Define the activation layers for all of the hidden layers.
        act_layers = [(f'actn_{n}', nn.Tanh()) for n in range(len(hidden_space_dims))]
        # Define sequential network for policy
        self.shared_net = nn.Sequential(
            OrderedDict(
                [lin_layers[n // 2] if n % 2 == 0 else act_layers[n // 2] for n in range(2 * len(hidden_space_dims))]
            )
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(nn.Linear(hidden_space_dims[-1], action_space_dims))

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(nn.Linear(hidden_space_dims[-1], action_space_dims))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass of the network on the given tensor.

        Args:
            x: An input tensor over which to perform the forward pass.

        Returns:
            A tuple of (Tensor, Tensor) for the mean and standard deviation of the policy.

        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = log(1 + exp(self.policy_stddev_net(shared_features)))

        return action_means, action_stddevs


class DenseNetworkPolicy(BasePyTorchPolicy):
    """Policy class that uses a dense neural network for constructing the mean and standard deviation of a Gaussian."""

    def __init__(self, env_id: str, hidden_space_dims: List[int]):
        """Initialise dense network policy."""
        self._env = gym.make(env_id)
        if not isinstance(self._env.action_space, gym.spaces.Box):
            raise PolicyException('The DenseNetworkPolicy is applicable to continuous action spaces only.')
        if not isinstance(self._env.observation_space, gym.spaces.Box):
            raise PolicyException('The DenseNetworkPolicy is applicable to continuous observation spaces only.')
        super().__init__(
            network=DensePolicyNetwork(
                self._env.observation_space.shape[0],
                self._env.action_space.shape[0],
                hidden_space_dims,
            )
        )

        self.eps = 1e-6

    def generate_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        """
        Generate a random action sampled from the policy.

        This function samples an action from a Gaussian in which the mean and standard deviation are constructed
        from the dense neural network.

        Args:
            observation: The current state observation.

        Returns:
            A NumPy array containing a randomly sampled continuous action from the environment's continuous
            action space.

        """
        action_mean, action_stddev = self.network(tensor(observation).T)
        dist = Normal(action_mean + self.eps, action_stddev + self.eps)
        return dist.sample().numpy()

    def calculate_log_probabilities(self, observations: npt.NDArray, actions: npt.NDArray) -> Tensor:
        """
        Calculate the log-probailities of the for the given (observation, action) pairs.

        Args:
            observations: The state observations from the environment.
            actions: The actions taken in the given observation state.

        Returns:
            A Tensor containing the log-probabilities of the policy for the given observation-action pairs.

        """
        action_means, action_stddevs = self.network(tensor(observations).T)
        return (
            Normal(
                action_means[:, 0] + self.eps,
                action_stddevs[:, 0] + self.eps,
            )
            .log_prob(tensor(actions))
            .T
        )
