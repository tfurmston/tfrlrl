import uuid
from typing import Dict, Union

import gymnasium as gym
import ray
from numpy.typing import NDArray

from tfrlrl.data_models.step import Step, Steps


@ray.remote
class Sampler:
    """
    This class provides functionality to sample from a given Gym environment.

    The class is single-threaded, i.e., it samples from a single instance of the environment using a single thread. The
    class provides iterable support, see https://docs.python.org/3/library/stdtypes.html#typeiter.
    """

    def __init__(self, env_id: str, n_steps: int = None):
        """
        Initialise the instance of the Sampler. This entails initialising the environment and setting member variables.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: If given, the number of steps to sample from the environment. If not given, then there is no
        limit on the number of sampled steps.
        """
        self._env = gym.make(env_id)
        self._env_id = str(uuid.uuid4())
        self._n_steps = n_steps
        self._n_steps_taken = 0
        self._n_env_steps_taken = 0
        self._observation = None
        self._next_observation = None
        self._action = None
        self._reward = None
        self._terminated = True
        self._truncated = True
        self._info = None

    def __iter__(self):
        """Ensure that the Sampler class supports the iterable protocol."""
        return self

    def __next__(self) -> (str, int, NDArray, Union[int, float, NDArray], NDArray, float, bool, Dict):
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        if self._n_steps is not None and self._n_steps_taken >= self._n_steps:
            raise StopIteration

        if self._terminated or self._truncated:
            self._observation, self._info = self._env.reset()
            self._env_id = str(uuid.uuid4())
            self._n_env_steps_taken = 0
        else:
            self._observation = self._next_observation

        self._action = self._env.action_space.sample()
        self._next_observation, self._reward, self._terminated, self._truncated, self._info = self._env.step(
            self._action,
        )
        self._n_steps_taken += 1
        self._n_env_steps_taken += 1
        return Step(
            env_id=self._env_id,
            time_step=self._n_env_steps_taken,
            observation=self._observation,
            action=self._action,
            next_observation=self._next_observation,
            reward=self._reward,
            done=self._terminated or self._truncated,
            info=self._info,
        )


class RaySampler:
    """
    This class provides functionality to sample from multiple instances of a given Gym environment through Ray.

    The class uses Ray to distribute the samplimng across the different environments.
    """

    def __init__(self, env_id: str, n_envs: int, n_steps: int = None):
        """
        Initialise instance of the RaySampler. This entails initialising the environment and setting member variables.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_envs: The number of environments from which to sample.
        :param n_steps: If given, the number of steps to sample from the environment. If not given, then there is no
        limit on the number of sampled steps.
        """
        self._envs = [Sampler.remote(env_id=env_id, n_steps=n_steps) for _ in range(n_envs)]

    def __iter__(self):
        """Ensure that the RaySampler class supports the iterable protocol."""
        return self

    def __next__(self) -> (str, int, NDArray, Union[int, float, NDArray], NDArray, float, bool, Dict):
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        return Steps(sample_steps=ray.get([env.__next__.remote() for env in self._envs]))
