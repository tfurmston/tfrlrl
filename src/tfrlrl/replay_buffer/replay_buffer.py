import logging
from typing import List

import gymnasium as gym
import numpy as np

from tfrlrl.data_models.replay_buffer import ReplayBufferSample

logger = logging.getLogger(__name__)


class ReplayBufferException(Exception):
    """Custom exception to encompass errors raised by the replay buffer class."""

    pass


class ReplayBuffer:
    """Experience replay buffer class."""

    def __init__(self, env_id: str, buffer_size: int = 100):
        """Initialise experience replay buffer."""
        env = gym.make(env_id)
        self._observations = np.ndarray(
            env.observation_space.shape + (buffer_size, ),
        )
        self._next_observations = np.ndarray(
            env.observation_space.shape + (buffer_size, ),
        )
        if isinstance(env.action_space, gym.spaces.Discrete):
            self._actions = np.ndarray(
                (1, buffer_size),
            )
        elif isinstance(env.action_space, gym.spaces.Box):
            self._actions = np.ndarray(
                env.action_space.shape + (buffer_size, ),
            )
        else:
            raise ReplayBufferException('Action space must be Discrete or Box.')
        self._rewards = np.ndarray(
            (1, buffer_size),
        )
        self._dones = np.ndarray(
            (1, buffer_size),
        )

        self._indx = 0
        self._buffer_full = False
        self._buffer_size = buffer_size

    def _increment_index(self, sample_size):
        if self._indx + sample_size >= self._buffer_size:
            self._buffer_full = True
        self._indx = (self._indx + sample_size) % self._buffer_size

    def _calculate_sample_indices(self, sample_size) -> List[int]:
        return [x % self._buffer_size for x in range(self._indx, self._indx + sample_size)]

    def _calculate_sample_range(self, n_steps: int) -> int:
        if self._buffer_full:
            indx = self._buffer_size
        else:
            indx = self._indx

        if indx < n_steps:
            raise ReplayBufferException(
                'There are insufficient samples in the replay buffer to sample %s steps', n_steps)
        return indx

    def _update_buffer(self, inds, observations, next_observations, actions, rewards, dones):
        self._observations[:, inds] = observations
        self._next_observations[:, inds] = next_observations
        self._actions[:, inds] = actions
        self._rewards[0, inds] = rewards
        self._dones[0, inds] = dones

    def add_step(self, step):
        """Add step to the replay buffer."""
        inds = self._calculate_sample_indices(1)
        self._update_buffer(
            inds,
            step.observation,
            step.next_observation,
            step.action,
            step.reward,
            step.done,
        )
        self._increment_index(1)

    def add_steps(self, steps):
        """Add steps to the replay buffer."""
        inds = self._calculate_sample_indices(steps.n_steps)
        self._update_buffer(
            inds,
            steps.observations,
            steps.next_observations,
            steps.actions,
            steps.rewards,
            steps.dones,
        )
        self._increment_index(steps.n_steps)

    def sample(self, n_steps: int) -> ReplayBufferSample:
        """Sample the given number of steps from the replay buffer."""
        inds = np.random.permutation(self._calculate_sample_range(n_steps))[:n_steps]
        return ReplayBufferSample(
            observations=self._observations[:, inds],
            next_observations=self._next_observations[:, inds],
            actions=self._actions[:, inds],
            rewards=self._rewards[:, inds],
            dones=self._dones[:, inds],
        )
