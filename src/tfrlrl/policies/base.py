from abc import ABC, abstractmethod
from typing import Any, Generator, Tuple, Union

import gymnasium as gym
from numpy.typing import NDArray
from torch import (
    Tensor,
    nn,
)


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
    def generate_action(self, observation: NDArray) -> Tuple[Union[int, float, NDArray]]:
        """
        Generate an action for the given state observation.

        Args:
            observation: The current state observation from the environment.

        Returns:
            An action appropriate for the environment's action space. Can be an integer
            for discrete action spaces, or a float/array for continuous action spaces.

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

    def generate_action(self, observation: NDArray) -> Tuple[Union[int, float, NDArray]]:
        """
        Generate a random action uniformly sampled from the discrete action space.

        Args:
            observation: The current state observation (ignored by this policy).

        Returns:
            A randomly sampled integer action from the environment's discrete action space.

        """
        return self._env.action_space.sample()


class BaseDifferentiablePolicy(BasePolicy):
    """
    Abstract base class for differentiable parameterized reinforcement learning policies.

    This class extends BasePolicy to support policies with differentiable parameters,
    enabling gradient-based policy optimization methods such as policy gradient algorithms.
    Subclasses must implement both generate_action and calculate_log_derivative methods.
    """

    @abstractmethod
    def calculate_log_derivative(self, observation: NDArray, action: Tuple[Union[int, float, NDArray]]) -> NDArray:
        """
        Calculate the log derivative of the policy with respect to its parameters.

        This method computes the gradient of the log probability of taking the given action
        in the given observation state with respect to the policy's parameters. This is used
        in policy gradient methods like REINFORCE, Actor-Critic, and PPO.

        Args:
            observation: The state observation from the environment.
            action: The action taken in the given observation state.

        Returns:
            The log derivative (gradient) of the policy parameters for the given observation-action pair.

        """
        ...

    @abstractmethod
    def get_parameters(self) -> NDArray:
        """
        Get the current policy parameters.

        Returns:
            The current parameters of the policy as a numpy array.

        """
        ...

    @abstractmethod
    def set_parameters(self, parameters: NDArray) -> None:
        """
        Set new policy parameters.

        Args:
            parameters: The new parameters to set for the policy.

        """
        ...


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

    def get_parameters(self) -> Generator[nn.parameter.Parameter, None, None]:
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

    @abstractmethod
    def calculate_log_probabilities(self, observations: NDArray, actions: NDArray) -> Tensor:
        """
        Calculate the log-probabilities of the given actions for the corresponding observations.

        Calculate the log-probabilities for the given actions for the corresponding observations.
        This function is expected to be used in batch.

        Args:
            observations: A NumPy array of the observations for which to calculate the log-probabilities.
            actions: A NumPy array of the actions for which to calculate the log-probabilities (of the corresponding
            observations).

        Returns:
            A PyTorch Tensor containing the log-probabilities of the given (observation, action) pairs.

        """
        ...
