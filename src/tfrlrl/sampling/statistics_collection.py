from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from tfrlrl.data_models.statistics import BaseStatistics


class BaseStatisticsCollector(ABC):
    """Base class for collecting statistics during sampling."""

    @abstractmethod
    def reset(self):
        """Reset the statistics in the collector."""
        ...

    # @abstractmethod
    # def update(self) -> None:
    #     """Update the statistics collector."""
    #     ...

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
    observations: np.ndarray
    actions: np.ndarray
    total_expected_rewards: np.ndarray


class EpisocidPolicyGradientStatisticsCollector(BaseStatisticsCollector):
    """Class for collecting policy statistics during episodes."""

    def __init__(self):
        """Initialise the policy-gradients statistics collector."""
        self.observations = []
        self.actions = []
        self.rewards = []

    def reset(self) -> None:
        """Reset the collected statistics to empty lists."""
        self.observations = []
        self.actions = []
        self.rewards = []

    def collect_step_statistics(self, sample) -> None:
        """
        Collect statistics for the given step sample.

        This function collects the statistics for the given step sample.

        Args:
            sample: The step sample from which to calculate the statistics.

        """
        # TODO: Fix the actions to be a consistent shape.
        # This should be done in the samplers, not here.
        if isinstance(sample.action, int):
            self.actions.append(sample.action * np.ones(1))
        elif isinstance(sample.action, np.integer):
            self.actions.append(sample.action[..., np.newaxis][..., np.newaxis])
        else:
            self.actions.append(sample.action)

        self.observations.append(sample.observation)
        self.rewards.append(sample.reward)

    def aggregate_statistics(self) -> EpisodePolicyGradientStatistics:
        """
        Aggregate the statistics collected to date.

        This function calculates the policy gradient from the collected episode. It then returns this
        gradient and the rewards.

        Returns:
            An instance of the EpisodePolicyGradientStatistics dataclass.

        """
        observations = np.concatenate(self.observations, axis=1)
        actions = np.concatenate(self.actions, axis=1)
        rewards = np.array(self.rewards)

        T = rewards.size
        total_expected_rewards = np.matmul(rewards, np.tril(np.ones(T))) / T

        return EpisodePolicyGradientStatistics(
            total_reward=np.sum(rewards),
            observations=observations,
            actions=actions,
            total_expected_rewards=total_expected_rewards,
        )

    @classmethod
    def merge_statistics(cls, statistics: List[EpisodePolicyGradientStatistics]) -> EpisodePolicyGradientStatistics:
        """
        Merge statistics across different episodes.

        This merges statistics across different instances of the EpisodePolicyGradientStatistics data class.

        Returns:
            An instance of the EpisodePolicyGradientStatistics dataclass.

        """
        return EpisodePolicyGradientStatistics(
            total_reward=np.array([x.total_reward for x in statistics]),
            observations=np.concatenate([x.observations for x in statistics], axis=1),
            actions=np.concatenate([x.actions for x in statistics], axis=1),
            total_expected_rewards=np.concatenate([x.total_expected_rewards for x in statistics]),
        )
