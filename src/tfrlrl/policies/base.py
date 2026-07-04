from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, Tuple, Union

import gymnasium as gym
import numpy as np
from torch import (
    Tensor,
    nn,
)
from torch.func import jacfwd


class PolicyException(Exception):
    """Custom exception to encompass errors raised by the policies."""

    pass


class BasePolicy(ABC):
    """
    Abstract base class for reinforcement learning policies.

    Policies are responsible for selecting actions given environment observations.
    Subclasses must implement the generate_action method to define action selection behavior.
    """

    @abstractmethod
    def generate_action(self, observation: np.ndarray) -> Union[int, float, np.ndarray]:
        """
        Generate an action for the given state observation.

        Args:
            observation: The current state observation from the environment.

        Returns:
            An action appropriate for the environment's action space. Can be an integer
            for discrete action spaces, or a float/array for continuous action spaces.

        """
        ...

    @abstractmethod
    def update(self, state_dict: Dict[str, Any]) -> None:
        """
        Update the policy using the keyword arguments provided.

        Args:
            state_dict: The state dictionary for the policy.

        """
        ...


class UniformActionSamplingPolicy(BasePolicy):
    """
    Policy that uniformly samples actions from a discrete action space.

    This policy ignores the observation and randomly selects actions with equal probability
    from the environment's discrete action space. Useful as a baseline or for exploration.

    Args:
        env_id: The Gymnasium environment ID (e.g., "CartPole-v1").

    Raises:
        PolicyException: If the environment does not have a discrete action space.

    """

    def __init__(self, env_id: str):
        """
        Initialize the UniformActionSamplingPolicy.

        :param env_id: The Gymnasium environment ID.
        :raises PolicyException: If the environment does not have a discrete action space.
        """
        super().__init__()
        self._env = gym.make(env_id)
        if not isinstance(self._env.action_space, gym.spaces.Discrete):
            raise PolicyException('The UniformActionSamplingPolicy is applicable to discrete action spaces only.')

    def generate_action(self, observation: np.ndarray) -> Union[int, float, np.ndarray]:
        """
        Generate a random action uniformly sampled from the discrete action space.

        Args:
            observation: The current state observation (ignored by this policy).

        Returns:
            A randomly sampled integer action from the environment's discrete action space.

        """
        return self._env.action_space.sample()

    def update(self, _: Dict[str, Any]) -> None:
        """
        Update the policy.

        This is a dummy function, as there is nothing to update in this policy.

        Args:
            state_dict: The state dictionary for the policy.

        """
        pass


class BasePyTorchPolicy(BasePolicy):
    """
    Abstract base class for PyTorch reinforcement learning policies.

    This class extends BasePolicy to support policies which are parameterised through PyTorch,
    enabling gradient-based policy optimization methods such as policy gradient algorithms.
    """

    def __init__(self, network: nn.Module):
        """
        Initialise dense network policy.

        Initialise the PyTorch policy, including setting the network.

        Args:
            network: An instance of a PyTorch Module that will be used within the policy.

        """
        super().__init__()
        self.network = network

    def get_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """
        Get the current policy parameters.

        Return the parameters of the PyTorch neural network. This function can be used to set the
        parameters in a PyTorch optimisation algorithm.

        Returns:
            The current parameters of the policy as a generator PyTorch parameters.

        """
        return self.network.parameters()

    def get_state(self) -> dict[str, Any]:
        """
        Get the state dictionary of the Pytorch network.

        Return the state dictionary of the policy's Pytorch network.

        Returns:
            The current state dictionary of the policy's Pytorch network.

        """
        return self.network.state_dict()

    def set_state(self, state_dict: dict[str, Any]) -> None:
        """
        Set the state dictionary of the Pytorch network.

        Args:
            state_dict: The state dictionary to be assigned to the policy's Pytorch network.

        """
        self.network.load_state_dict(state_dict)

    def update(self, state_dict) -> None:
        """
        Update the policy.

        Args:
            state_dict: The state dictionary to be assigned to the policy's Pytorch network.

        """
        self.set_state(state_dict)

    @abstractmethod
    def calculate_log_probabilities(self, observations: np.ndarray, actions: np.ndarray) -> Tensor:
        """
        Calculate the log-probabilities of the given actions for the corresponding observations.

        Calculate the log-probabilities for the given actions for the corresponding observations.
        This function is expected to be used in batch.

        Args:
            observations: A NumPy array of the observations for which to calculate the log-probabilities.
            actions: A NumPy array of the actions for which to calculate the log-probabilities (of the corresponding
            observations).

        Returns:
            A PyTorch Tensor containing the log-probabilities of the given (observation, action) pairs. The shape of
            the output is expected to be either (n_observation) or (1, n_observations).

        """
        ...

    @abstractmethod
    def make_log_prob_fn(self, observations: np.ndarray, actions: np.ndarray) -> Tuple[Callable[[Dict], Tensor], Dict]:
        """
        Construct a PyTorch functional to calculate the log-probabilities of the given state-action pairs.

        Construct a PyTorch functional that takes the policy parameters as inputs and returns the log-probabilites
        of the given state-action pairs. This functional can be used in various functionality, such as in the
        construction of the Jacboian of the policy.

        Args:
            observations: A NumPy array of the observations for which to calculate the log-probabilities.
            actions: A NumPy array of the actions for which to calculate the log-probabilities (of the corresponding
            observations).

        Returns:
            A tuple consisting of a function that takes the policy parameters as inputs and returns the
            log-probabilities of the given state-action pairs and a dictionary of policy parameters.

        """
        ...

    def calculate_jacobian(self, observations: np.ndarray, actions: np.ndarray) -> Dict[str, Tensor]:
        """
        Calculate the Jacobian of the log-probabilites at the given state-action pairs.

        This function calculates the Jacobian of the log-probabilities of the policy for the given
        state-action pairs. The Jacobian is given in the form of a dictionary, with the keys corresponding
        to the different parameters of the policy (network).

        Args:
            observations: A NumPy array of the observations for which to calculate the log-probabilities.
            actions: A NumPy array of the actions for which to calculate the log-probabilities (of the corresponding
            observations).

        Returns:
            A dictionary mapping the parameter name to the (rows of the) Jacobian corresponding to that parameter.

        """
        log_prob_fn, params = self.make_log_prob_fn(observations, actions)
        return jacfwd(log_prob_fn)(params)
