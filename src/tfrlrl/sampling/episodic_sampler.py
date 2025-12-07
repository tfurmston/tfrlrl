from typing import Optional, Tuple

from numpy.typing import NDArray

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

    def __next__(self) -> Tuple[NDArray, NDArray]:
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
        self._n_episodes_taken = 0

    def update_policy(self, new_policy: BasePolicy) -> None:
        """
        Update the policy used for action selection.

        :param new_policy: New policy instance to use for sampling.
        """
        self._sampler._policy = new_policy
        self._statistics_collector.update_policy(new_policy)
