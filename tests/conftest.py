import os
from collections import ChainMap, defaultdict
from dataclasses import dataclass
from typing import List

from pytest import fixture

from tfrlrl.policies.base import BasePolicy
from tfrlrl.sampling.statistics_collection import BaseStatisticsCollector


@dataclass
class DummyStatistics:
    """Dataclass for the statistics collected test sample episodes."""

    samples: dict[str, list]  # A map from the episode ID to the list of steps in the episode.


class DummyStatisticsCollector(BaseStatisticsCollector):
    """Test class for collecting statistics during sampling."""

    def __init__(self):
        """Initialise statistics collector."""
        self._samples = defaultdict(list)

    def reset(self):
        """Reset the statistics in the collector."""
        self._samples = defaultdict(list)

    def update_policy(self, new_policy: BasePolicy) -> None:
        """Update the policy of the statistics collector."""
        pass

    def collect_step_statistics(self, sample):
        """Collect statistics from a sample step."""
        self._samples[sample.env_id].append(sample)

    def aggregate_statistics(self):
        """Aggregate the statistics collected by the collector."""
        return DummyStatistics(samples=self._samples)

    @classmethod
    def merge_statistics(cls, statistics: List[DummyStatistics]):
        """Aggregate the statistics collected by the collector."""
        return DummyStatistics(
            samples=dict(ChainMap(*[x.samples for x in statistics])),
        )


@fixture(scope='session')
def test_ray_cluster(num_cpus: int = 2):
    """Pytest fixture for construct a test Ray cluster."""
    os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'

    import ray

    ray.init(num_cpus=num_cpus)
    yield
    ray.shutdown()
    del os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO']
