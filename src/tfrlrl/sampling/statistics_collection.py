from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tfrlrl.policies.base import BaseDifferentiablePolicy, BasePolicy


class BaseStatisticsCollector(ABC):
    """Base class for collecting statistics during sampling."""

    @abstractmethod
    def reset(self):
        """Reset the statistics in the collector."""
        ...

    @abstractmethod
    def update_policy(self, new_policy: BasePolicy) -> None:
        """Update the policy of the statistics collector."""
        ...

    @abstractmethod
    def collect_step_statistics(self, sample):
        """Collect statistics from a sample step."""
        ...

    @abstractmethod
    def aggregate_statistics(self):
        """Aggregate the statistics collected by the collector."""
        ...


class EpisocidPolicyGradientStatisticsCollector(BaseStatisticsCollector):
    """Class for collecting policy statistics during episodes."""

    def __init__(self, policy: Optional[BaseDifferentiablePolicy]):
        """Initialise the policy-gradients statistics collector."""
        self._policy = policy
        self.rewards = []
        self.log_pol_grads = []

    def reset(self):
        """Reset the collected statistics to empty lists."""
        self.rewards = []
        self.log_pol_grads = []

    def update_policy(self, new_policy: BaseDifferentiablePolicy) -> None:
        """Update the policy of the statistics collector."""
        self._policy = new_policy

    def collect_step_statistics(self, sample):
        """
        Collect statistics for the given step sample.

        This function collects the statistics for the given step sample. This consists of the gradient of the
        log-polify and the reward.

        :param: sample: The step sample from which to calculate the statistics.
        """
        self.log_pol_grads.append(
            self._policy.calculate_log_derivative(
                sample.observation,
                sample.action,
            )
        )
        self.rewards.append(sample.reward)

    def aggregate_statistics(self) -> Tuple[NDArray, NDArray]:
        """
        Aggregate the statistics collected to date.

        This function calculates the policy gradient from the collected episode. It then returns this
        gradient and the rewards.
        :return: The policy gradient and step rewards of the episode.
        """
        log_pol_grads = np.array(self.log_pol_grads)
        rewards = np.array(self.rewards)
        T = rewards.size
        episode_gradient = np.matmul(np.matmul(rewards, np.tril(np.ones(T))), log_pol_grads) / T
        return rewards, episode_gradient
