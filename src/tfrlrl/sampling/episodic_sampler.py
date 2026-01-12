from typing import Optional

import ray

from tfrlrl.data_models.statistics import BaseStatistics
from tfrlrl.policies.base import BasePolicy
from tfrlrl.sampling.sampler import Sampler
from tfrlrl.sampling.statistics_collection import BaseStatisticsCollector


class EpisodicSampler:
    """
    Class that provides functionality to sample episodes from a given Gym environment.

    The class is single-threaded, i.e., it samples from a single instance of the environment using a single thread. The
    class provides iterable support, see https://docs.python.org/3/library/stdtypes.html#typeiter.
    """

    def __init__(
        self,
        env_id: str,
        statistics_collector: Optional[BaseStatisticsCollector],
        n_episodes: int = None,
        policy: Optional[BasePolicy] = None,
        **kwargs,
    ):
        """
        Initialise instance of EpisodicSampler, which entails initialising the environment and setting member variables.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param statistics_collector: Optional instance of a statistics collector.
        :param n_episodes: If given, the number of episodes to sample from the environment. If not given, then there is
          no limit on the number of sampled episodes.
        :param policy: Optional policy instance for action selection. If not provided, defaults to
        UniformActionSamplingPolicy.
        :param kwargs: Optional keyword-arguments for the environment.
        """
        self._sampler = Sampler(env_id, policy=policy, **kwargs)
        self._statistics_collector = statistics_collector
        self._n_episodes = n_episodes
        self._n_episodes_taken = 0

    def __iter__(self):
        """Ensure that the EpisodicSampler class supports the iterable protocol."""
        return self

    def __next__(self) -> BaseStatistics:
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        if self._n_episodes is not None and self._n_episodes_taken >= self._n_episodes:
            raise StopIteration

        self._statistics_collector.reset()
        for sample in self._sampler:
            self._statistics_collector.collect_step_statistics(sample)
            if sample.done:
                self._n_episodes_taken += 1
                break
        return self._statistics_collector.aggregate_statistics()

    def reset(self) -> None:
        """Reset the iterator so that a new iterable can be created."""
        self._statistics_collector.reset()
        self._n_episodes_taken = 0

    def update_policy(self, new_policy: BasePolicy) -> None:
        """
        Update the policy used for action selection.

        :param new_policy: New policy instance to use for sampling.
        """
        self._sampler._policy = new_policy
        self._statistics_collector.update_policy(new_policy)

    def sample(self) -> BaseStatistics:
        """Sample all episodes from the sampler and merge the statistics."""
        return self._statistics_collector.merge_statistics([x for x in self])


RemoteEpisodicSampler = ray.remote(EpisodicSampler)


class RayEpisodicSampler:
    """
    Class that provides functionality to sample episodes from a given Gym environment in a parallel manner.

    The class uses Ray to sample multiple episodes concurrently from different workers.
    """

    def __init__(
        self,
        n_samplers: int,
        env_id: str,
        statistics_collector: Optional[BaseStatisticsCollector],
        n_episodes: int = None,
        policy: Optional[BasePolicy] = None,
        **kwargs,
    ):
        """
        Initialise instance of RayEpisodicSampler, which entails initialising multiple samplers to be used by Ray.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param statistics_collector: Optional instance of a statistics collector.
        :param n_episodes: If given, the number of episodes to sample from the environment (in each of the Ray
        workers). If not given, then there is no limit on the number of sampled episodes.
        :param policy: Optional policy instance for action selection. If not provided, defaults to
        UniformActionSamplingPolicy.
        :param kwargs: Optional keyword-arguments for the environment.
        """
        self.statistics_collector = statistics_collector
        self.samplers = [
            RemoteEpisodicSampler.remote(
                env_id=env_id,
                statistics_collector=statistics_collector,
                n_episodes=n_episodes // n_samplers,
                policy=policy,
                **kwargs,
            )
            for _ in range(n_samplers)
        ]

    def __iter__(self):
        """Ensure that the RayEpisodicSampler class supports the iterable protocol."""
        return self

    def __next__(self) -> BaseStatistics:
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        return self.statistics_collector.merge_statistics(
            ray.get([sampler.__next__.remote() for sampler in self.samplers])
        )

    def reset(self) -> None:
        """Reset all samplers."""
        ray.get([env.reset.remote() for env in self.samplers])

    def update_policy(self, new_policy: BasePolicy) -> None:
        """
        Update the policy across all samplers.

        :param new_policy: New policy instance to use for sampling across all samplers.
        """
        ray.get([env.update_policy.remote(new_policy) for env in self.samplers])

    def sample(self) -> BaseStatistics:
        """Sample all episodes from the sampler and merge the statistics."""
        return self.statistics_collector.merge_statistics(
            ray.get([sampler.sample.remote() for sampler in self.samplers])
        )
