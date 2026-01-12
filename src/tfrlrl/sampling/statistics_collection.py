from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tfrlrl.data_models.statistics import BaseStatistics
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
    def collect_step_statistics(self, sample) -> None:
        """Collect statistics from a sample step."""
        ...

    @abstractmethod
    def aggregate_statistics(self) -> BaseStatistics:
        """Aggregate the statistics collected by the collector."""
        ...

    @classmethod
    @abstractmethod
    def merge_statistics(cls) -> BaseStatistics:
        """Merge the statistics collected by different aggregations of statistics collector(s)."""
        ...


@dataclass
class EpisodePolicyGradientStatistics(BaseStatistics):
    """Dataclass for the statistics collected in episodic policy gradients."""

    total_reward: float
    episode_gradient: np.ndarray


class EpisocidPolicyGradientStatisticsCollector(BaseStatisticsCollector):
    """Class for collecting policy statistics during episodes."""

    def __init__(self, policy: Optional[BaseDifferentiablePolicy]):
        """Initialise the policy-gradients statistics collector."""
        self._policy = policy
        self.rewards = []
        self.log_pol_grads = []

    def reset(self) -> None:
        """Reset the collected statistics to empty lists."""
        self.rewards = []
        self.log_pol_grads = []

    def update_policy(self, new_policy: BaseDifferentiablePolicy) -> None:
        """Update the policy of the statistics collector."""
        self._policy = new_policy

    def collect_step_statistics(self, sample) -> None:
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

    def aggregate_statistics(self) -> EpisodePolicyGradientStatistics:
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
        episode_gradient = episode_gradient[..., np.newaxis]
        return EpisodePolicyGradientStatistics(
            total_reward=np.sum(rewards),
            episode_gradient=episode_gradient,
        )

    @classmethod
    def merge_statistics(cls, statistics: List[EpisodePolicyGradientStatistics]) -> EpisodePolicyGradientStatistics:
        """Merge statistics across different episodes."""
        return EpisodePolicyGradientStatistics(
            total_reward=np.array([x.total_reward for x in statistics]),
            episode_gradient=np.concatenate([x.episode_gradient for x in statistics], axis=1),
        )
