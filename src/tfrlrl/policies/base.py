from abc import ABC, abstractmethod
from typing import Tuple, Union

import gymnasium as gym
from numpy.typing import NDArray


class PolicyException(Exception):
    """Custom exception to encompass errors raised by the policies."""

    pass


class BasePolicy(ABC):
    """Abstract base class for reinforcement learning policies.

    Policies are responsible for selecting actions given environment observations.
    Subclasses must implement the generate_action method to define action selection behavior.
    """

    @abstractmethod
    def generate_action(self, observation: NDArray) -> Tuple[Union[int, float, NDArray]]:
        """Generate an action for the given state observation.

        Args:
            observation: The current state observation from the environment.

        Returns:
            An action appropriate for the environment's action space. Can be an integer
            for discrete action spaces, or a float/array for continuous action spaces.
        """
        ...


class UniformActionSamplingPolicy(BasePolicy):
    """Policy that uniformly samples actions from a discrete action space.

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
        """Generate a random action uniformly sampled from the discrete action space.

        Args:
            observation: The current state observation (ignored by this policy).

        Returns:
            A randomly sampled integer action from the environment's discrete action space.
        """
        return self._env.action_space.sample()


class BaseDifferentiablePolicy(BasePolicy):
    """Abstract base class for differentiable parameterized reinforcement learning policies.

    This class extends BasePolicy to support policies with differentiable parameters,
    enabling gradient-based policy optimization methods such as policy gradient algorithms.
    Subclasses must implement both generate_action and calculate_log_derivative methods.
    """

    @abstractmethod
    def calculate_log_derivative(self, observation: NDArray, action: Tuple[Union[int, float, NDArray]]) -> NDArray:
        """Calculate the log derivative of the policy with respect to its parameters.

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
