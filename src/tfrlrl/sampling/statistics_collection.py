import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from tfrlrl.baselines.linear import Baseline
from tfrlrl.data_models.statistics import BaseStatistics
from tfrlrl.data_models.step import construct_step_dataclasses
from tfrlrl.sampling.utils import merge_optional_statistics

logger = logging.getLogger(__name__)


@dataclass
class EpisodePolicyGradientStatistics(BaseStatistics):
    """Dataclass for the statistics collected in episodic policy gradients."""

    total_reward: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    total_expected_rewards: np.ndarray
    baseline_features: Optional[np.ndarray] = None
    baseline_targets: Optional[np.ndarray] = None


class BaseStatisticsCollector(ABC):
    """
    Base class for collecting statistics during sampling.

    Attributes:
        steps_cls: A dataclass for aggregating a collection of steps.
        baseline: An instance of a baseline class, if one was given during class construction.

    """

    def __init__(self, env_id: str, baseline: Optional[Baseline] = None):
        """
        Initialise the base statistics collector.

        Args:
            env_id: The I.D. of the environment from which to collect statistics.
            baseline: If given, an instance of the baseline class, which will be used for
            variance reduction when estimating policy gradients.

        """
        _, _, self.steps_cls = construct_step_dataclasses(
            env_id,
        )
        self.baseline = baseline

    def update(self, baseline_state_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the the statistic collector, e.g., the baseline calculations.

        Args:
            baseline_state_dict: The state dictionary of the baseline.

        """
        if self.baseline is not None and baseline_state_dict is not None:
            self.baseline.update(state_dict=baseline_state_dict)

    @abstractmethod
    def reset(self) -> None:
        """Reset the statistics in the collector."""
        ...

    @abstractmethod
    def collect_step_statistics(self, sample) -> None:
        """
        Collect statistics for the given step sample.

        This function collects the statistics for the given step sample.

        Args:
            sample: The step sample from which to calculate the statistics.

        """
        ...

    @abstractmethod
    def aggregate_statistics(self) -> BaseStatistics:
        """
        Aggregate the statistics collected to date.

        This function calculates the total future rewards of the observed examples. If a baseline
        is specified, then the baselines will be calculated and subtracted from the total future
        rewards.

        Returns:
            An instance of the EpisodePolicyGradientStatistics dataclass.

        """
        ...

    @classmethod
    @abstractmethod
    def merge_statistics(cls, statistics: List[BaseStatistics]) -> BaseStatistics:
        """Merge the statistics collected by different aggregations of statistics collector(s)."""
        ...


class EpisocidPolicyGradientStatisticsCollector(BaseStatisticsCollector):
    """
    Statistics collector for episodic-level statistics collection.

    This class can be used to collect statistics for episodes sampled from environments. In
    particular, it collects the actions, observations and total future rewards for all of the
    steps in an episode.

    Attributes:
        steps_cls: A dataclass for aggregating a collection of steps.
        baseline: An instance of a baseline class, if one was given during class construction.
        steps: The collection of steps collected to date.

    """

    def __init__(self, env_id: str, baseline: Optional[Baseline] = None):
        """
        Initialise the policy-gradients statistics collector.

        Args:
            env_id: The I.D. of the environment from which to collect statistics.
            baseline: If given, an instance of the baseline class, which will be used for
            variance reduction when estimating policy gradients.

        """
        super().__init__(
            env_id=env_id,
            baseline=baseline,
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

        This function calculates the total future rewards of the observed examples. If a baseline
        is specified, then the baselines will be calculated and subtracted from the total future
        rewards.

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
        return EpisodePolicyGradientStatistics(
            total_reward=np.array([x.total_reward for x in statistics]),
            observations=np.concatenate([x.observations for x in statistics], axis=-1),
            actions=np.concatenate([x.actions for x in statistics], axis=-1),
            total_expected_rewards=np.concatenate([x.total_expected_rewards for x in statistics]),
            baseline_features=merge_optional_statistics(statistics, 'baseline_features', 1),
            baseline_targets=merge_optional_statistics(statistics, 'baseline_targets', 0),
        )
