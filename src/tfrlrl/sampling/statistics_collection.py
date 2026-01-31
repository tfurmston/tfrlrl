import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from tfrlrl.baselines.linear import Baseline
from tfrlrl.data_models.statistics import BaseStatistics, StatisticsException
from tfrlrl.data_models.step import construct_step_dataclasses

logger = logging.getLogger(__name__)


class BaseStatisticsCollector(ABC):
    """Base class for collecting statistics during sampling."""

    @abstractmethod
    def reset(self):
        """Reset the statistics in the collector."""
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
    observations: np.ndarray
    actions: np.ndarray
    total_expected_rewards: np.ndarray
    baseline_features: np.ndarray = None
    baseline_targets: np.ndarray = None


class EpisocidPolicyGradientStatisticsCollector(BaseStatisticsCollector):
    """
    Statistics collector for episodic-level statistics collection.

    This class can be used to collect statistics for episodes sampled from environments. In
    particular, it collects the actions, observations and total future rewards for all of the
    steps in an episode.

    Attributes:
        steps_cls: A dataclass for aggregating a collection of steps.
        steps: The collection of steps collected to date.

    """

    def __init__(self, env_id: str):
        """
        Initialise the policy-gradients statistics collector.

        Args:
            env_id: The I.D. of the environment from which to collect statistics.

        """
        _, _, self.steps_cls = construct_step_dataclasses(
            env_id,
        )
        self.steps = []

    def reset(self) -> None:
        """Reset the collected statistics to empty lists."""
        self.steps = []

    def collect_step_statistics(self, sample) -> None:
        """
        Collect statistics for the given step sample.

        This function collects the statistics for the given step sample.

        Args:
            sample: The step sample from which to calculate the statistics.

        """
        self.steps.append(sample)

    def aggregate_statistics(self) -> EpisodePolicyGradientStatistics:
        """
        Aggregate the statistics collected to date.

        This function calculates the policy gradient from the collected episode. It then returns this
        gradient and the rewards.

        Returns:
            An instance of the EpisodePolicyGradientStatistics dataclass.

        """
        steps = self.steps_cls(sample_steps=self.steps)

        T = steps.rewards.size
        total_expected_rewards = np.matmul(steps.rewards, np.tril(np.ones(T))) / T

        baseline_features = None
        if self.baseline is not None:
            baseline_features = self.baseline.calculate_features(steps.observations, np.arange(T))
            logger.debug('Substracting baseline from total expected rewards.')
            total_expected_rewards -= self.baseline.calculate_baseline(
                steps.observations,
                np.arange(T),
                feature_matrix=baseline_features,
            )

        return EpisodePolicyGradientStatistics(
            total_reward=np.sum(steps.rewards),
            observations=steps.observations,
            actions=steps.actions,
            total_expected_rewards=total_expected_rewards,
            baseline_features=baseline_features,
            baseline_targets=None if self.baseline is None else total_expected_rewards,
        )

    @classmethod
    def merge_statistics(cls, statistics: List[EpisodePolicyGradientStatistics]) -> EpisodePolicyGradientStatistics:
        """
        Merge statistics across different episodes.

        This merges statistics across different instances of the EpisodePolicyGradientStatistics data class.

        Returns:
            An instance of the EpisodePolicyGradientStatistics dataclass.

        """
        baseline_features = [x.baseline_features for x in statistics]
        baseline_targets = [x.baseline_targets for x in statistics]

        if any([x is None for x in baseline_features]) and any([x is not None for x in baseline_features]):
            raise StatisticsException('All baseline features should either be None or a NumPy array.')
        if any([x is None for x in baseline_targets]) and any([x is not None for x in baseline_targets]):
            raise StatisticsException('All baseline features should either be None or a NumPy array.')

        return EpisodePolicyGradientStatistics(
            total_reward=np.array([x.total_reward for x in statistics]),
            observations=np.concatenate([x.observations for x in statistics], axis=-1),
            actions=np.concatenate([x.actions for x in statistics], axis=-1),
            total_expected_rewards=np.concatenate([x.total_expected_rewards for x in statistics]),
            baseline_features=baseline_features or None,
            baseline_targets=baseline_targets or None,
        )
