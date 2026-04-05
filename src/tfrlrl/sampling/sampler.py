import uuid
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
from numpy.typing import NDArray

from tfrlrl.data_models.step import construct_step_dataclasses
from tfrlrl.policies.base import BasePolicy, UniformActionSamplingPolicy


class Sampler:
    """
    Class that provides functionality to sample from a given Gym environment.

    The class is single-threaded, i.e., it samples from a single instance of the environment using a single thread. The
    class provides iterable support, see https://docs.python.org/3/library/stdtypes.html#typeiter.
    """

    def __init__(self, env_id: str, n_steps: Optional[int] = None, policy: Optional[BasePolicy] = None, **kwargs):
        """
        Initialise instance of Sampler, which entails initialising the environment and setting member variables.

        Args:
            env_id: The Gym environment ID to be used in the sampling.
            n_steps: If given, the number of steps to sample from the environment. If not given, then there is no
            limit on the number of sampled steps.
            policy: Optional policy instance for action selection. If not provided, defaults to
            UniformActionSamplingPolicy.
            kwargs: Optional keyword-arguments for the environment.

        """
        self.obs_cls, self.step_cls, self.steps_cls = construct_step_dataclasses(
            env_id,
        )
        self._env = gym.make(env_id, **kwargs)
        self._env_id = str(uuid.uuid4())
        self._n_steps = n_steps
        self._n_steps_taken = 0
        self._n_env_steps_taken = 0
        self.step = None
        self._policy = policy if policy is not None else UniformActionSamplingPolicy(env_id)

    def __iter__(self):
        """Ensure that the Sampler class supports the iterable protocol."""
        return self

    def __next__(self) -> Tuple[str, int, NDArray, Union[int, float, NDArray], NDArray, float, bool, Dict]:
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        if self._n_steps is not None and self._n_steps_taken >= self._n_steps:
            raise StopIteration

        if self.step is None or self.step.done:
            initial_observation, info = self._env.reset()
            observation = self.obs_cls(observation=initial_observation).observation
            self._env_id = str(uuid.uuid4())
            self._n_env_steps_taken = 0
        else:
            observation = self.step.next_observation

        action = self._policy.generate_action(observation[..., 0])
        next_observation, reward, terminated, truncated, info = self._env.step(
            action,
        )
        self.step = self.step_cls(
            env_id=self._env_id,
            time_step=self._n_env_steps_taken,
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=terminated or truncated,
            info=info,
        )

        self._n_steps_taken += 1
        self._n_env_steps_taken += 1
        return self.step

    def reset(self) -> None:
        """Reset the iterator so that a new iterable can be created."""
        if self._n_steps_taken is not None:
            self._n_steps_taken = 0

    def update(self, policy_state_dict: Dict[str, Any]) -> None:
        """
        Update the sampler, e.g. the policy used for action selection.

        Args:
            policy_state_dict: The state dictionary for the policy.

        """
        self._policy.update(
            state_dict=policy_state_dict,
        )
