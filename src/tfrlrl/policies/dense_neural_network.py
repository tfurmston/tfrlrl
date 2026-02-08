from collections import OrderedDict
from typing import List

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from torch import (
    Tensor,
    exp,
    log,
    nn,
    tensor,
)
from torch.distributions.normal import Normal

from tfrlrl.policies.base import BaseDifferentiablePolicy


class DensePolicyNetwork(nn.Module):
    """
    PyTorch module class for a dense neural network.

    The dense neural network of this class is used to construct the mean and standard deviation of a Gausssian
    policy.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int, hidden_space_dims: List[int]):
        """
        Initialise dense neural network for calculating the mean and standard deviation of a Gaussian policy.

        param: obs_space_dims: The number of dimensions in the observation space.
        param: action_space_dims: The number of dimensions in the action space.
        param: hidden_space_dims: A list of the number of dimensions for the hidden layers in the network.
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

        param: x: An input tensor over which to perform the forward pass.
        return: A tuple of (Tensor, Tensor) for the mean and standard deviation of the policy.
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = log(1 + exp(self.policy_stddev_net(shared_features)))

        return action_means, action_stddevs


class DenseNetworkPolicy(BaseDifferentiablePolicy):
    """Policy class that uses a dense neural network for constructing the mean and standard deviation of a Gaussian."""

    def __init__(self, env_id: str, hidden_space_dims: List[int]):
        """Initialise dense network policy."""
        super().__init__()
        self._env = gym.make(env_id)
        # if not isinstance(self._env.action_space, gym.spaces.Discrete):
        #     raise PolicyException('The LinearSoftMax is applicable to discrete action spaces only.')

        self.network = DensePolicyNetwork(
            self._env.observation_space.shape[0],
            self._env.action_space.shape[0],
            hidden_space_dims,
        )
        self.eps = 1e-6

    def generate_action(self, observation: npt.ArrayLike) -> npt.ArrayLike:
        """
        Generate a random action sampled from the policy.

        This function samples an action from a Gaussian in which the mean and standard deviation are constructed
        from the dense neural network.

        param: observation: The current state observation.
        return: A randomly sampled continuous action from the environment's continuous action space.
        """
        action_mean, action_stddev = self.network(tensor(observation))
        dist = Normal(action_mean[0] + self.eps, action_stddev[0] + self.eps)
        action = dist.sample().numpy()
        action = action[..., np.newaxis]
        return action

    def calculate_log_derivative(self):
        """
        Calculate the log derivative of the policy with respect to its parameters.

        This method computes the gradient of the log probability of taking the given action
        in the given observation state with respect to the policy's parameters. This is used
        in policy gradient methods like REINFORCE, Actor-Critic, and PPO.

        param: observation: The state observation from the environment.
        param: action: The action taken in the given observation state.
        return: The log derivative (gradient) of the policy parameters for the given observation-action pair.
        """
        raise NotImplementedError

    def get_parameters(self):
        """
        Get the current policy parameters.

        return: The current parameters of the policy as a numpy array.
        """
        raise NotImplementedError

    def set_parameters(self):
        """
        Set new policy parameters.

        param: parameters: The new parameters to set for the policy.
        """
        raise NotImplementedError
