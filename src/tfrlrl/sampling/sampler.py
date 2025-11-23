import uuid
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import ray
from numpy.typing import NDArray

from tfrlrl.data_models.step import construct_step_dataclasses
from tfrlrl.policies.base import BasePolicy, UniformActionSamplingPolicy


@ray.remote
class Sampler:
    """
    This class provides functionality to sample from a given Gym environment.

    The class is single-threaded, i.e., it samples from a single instance of the environment using a single thread. The
    class provides iterable support, see https://docs.python.org/3/library/stdtypes.html#typeiter.
    """

    def __init__(self, env_id: str, n_steps: int = None, policy: Optional[BasePolicy] = None):
        """
        Initialise the instance of the Sampler. This entails initialising the environment and setting member variables.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: If given, the number of steps to sample from the environment. If not given, then there is no
        limit on the number of sampled steps.
        :param policy: Optional policy instance for action selection. If not provided, defaults to
        UniformActionSamplingPolicy.
        """
        self.step_cls, self.steps_cls = construct_step_dataclasses(env_id)
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
        self._policy = policy if policy is not None else UniformActionSamplingPolicy(env_id)

    def __iter__(self):
        """Ensure that the Sampler class supports the iterable protocol."""
        return self

    def __next__(self) -> Tuple[str, int, NDArray, Union[int, float, NDArray], NDArray, float, bool, Dict]:
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        if self._n_steps is not None and self._n_steps_taken >= self._n_steps:
            raise StopIteration

        if self._terminated or self._truncated:
            self._observation, self._info = self._env.reset()
            if isinstance(self._observation, float):
                self._observation = self._observation * np.ones(1)
            elif isinstance(self._observation, int):
                self._observation = self._observation * np.ones(1, dtype=np.int64)
            self._env_id = str(uuid.uuid4())
            self._n_env_steps_taken = 0
        else:
            self._observation = self._next_observation

        self._action = self._policy.generate_action(self._observation)
        self._next_observation, self._reward, self._terminated, self._truncated, self._info = self._env.step(
            self._action,
        )
        if isinstance(self._next_observation, float):
            self._next_observation = self._next_observation * np.ones(1)
        elif isinstance(self._next_observation, int):
            self._next_observation = self._next_observation * np.ones(1, dtype=np.int64)

        self._n_steps_taken += 1
        self._n_env_steps_taken += 1
        return self.step_cls(
            env_id=self._env_id,
            time_step=self._n_env_steps_taken,
            observation=self._observation,
            action=self._action,
            next_observation=self._next_observation,
            reward=self._reward,
            done=self._terminated or self._truncated,
            info=self._info,
        )

    def update_policy(self, new_policy: BasePolicy) -> None:
        """
        Update the policy used for action selection.

        :param new_policy: New policy instance to use for sampling.
        """
        self._policy = new_policy


class RaySampler:
    """
    This class provides functionality to sample from multiple instances of a given Gym environment through Ray.

    The class uses Ray to distribute the samplimng across the different environments.
    """

    def __init__(self, env_id: str, n_envs: int, n_steps: int = None, policy: Optional[BasePolicy] = None):
        """
        Initialise instance of the RaySampler. This entails initialising the environment and setting member variables.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_envs: The number of environments from which to sample.
        :param n_steps: If given, the number of steps to sample from the environment. If not given, then there is no
        limit on the number of sampled steps.
        :param policy: Optional policy instance for action selection. If not provided, defaults to
        UniformActionSamplingPolicy in each Sampler.
        """
        self.step_cls, self.steps_cls = construct_step_dataclasses(env_id)
        self._envs = [Sampler.remote(env_id=env_id, n_steps=n_steps, policy=policy) for _ in range(n_envs)]

    def __iter__(self):
        """Ensure that the RaySampler class supports the iterable protocol."""
        return self

    def __next__(self) -> Tuple[str, int, NDArray, Union[int, float, NDArray], NDArray, float, bool, Dict]:
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        return self.steps_cls(sample_steps=ray.get([env.__next__.remote() for env in self._envs]))

    def update_policy(self, new_policy: BasePolicy) -> None:
        """
        Update the policy across all sampler actors.

        :param new_policy: New policy instance to use for sampling in all environments.
        """
        ray.get([env.update_policy.remote(new_policy) for env in self._envs])
