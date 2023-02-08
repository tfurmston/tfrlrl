import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class ReplayBuffer:

    def __init__(self, d_obs: int, d_action: int, buffer_size : int = 100):

        self._observations = np.ndarray(
            (d_obs, buffer_size),
        )
        self._next_observations = np.ndarray(
            (d_obs, buffer_size),
        )
        self._actions = np.ndarray(
            (d_action, buffer_size),
        )
        self._rewards = np.ndarray(
            (1, buffer_size),
        )
        self._dones = np.ndarray(
            (1, buffer_size),
        )

        self._indx = 0
        self._buffer_size = buffer_size

    def _increment_index(self, sample_size):
        self._indx = (self._indx + sample_size) % self._buffer_size

    def _calculate_sample_indices(self, sample_size) -> List[int]:
        return [x % self._buffer_size for x in range(self._indx, self._indx + sample_size)]

    def _update_buffer(self, inds, observations, next_observations, actions, rewards, dones):
        self._observations[:, inds] = observations
        self._next_observations[:, inds] = next_observations
        self._actions[:, inds] = actions
        self._rewards[0, inds] = rewards
        self._dones[0, inds] = dones

    def add_sample(self, observation: np.ndarray, next_observation: np.ndarray, action: int, reward: float, done: bool):
        inds = self._calculate_sample_indices(1)
        self._update_buffer(
            inds,
            observation,
            next_observation,
            action,
            reward,
            done,
        )
        self._increment_index(1)

    def add_samples(self, observations: np.ndarray, next_observations: np.ndarray, actions: int, rewards: float,
                    dones: bool):
        sample_size = 2
        inds = self._calculate_sample_indices(sample_size)
        self._update_buffer(
            inds,
            observations,
            next_observations,
            actions,
            rewards,
            dones,
        )
        self._increment_index(sample_size)

    def sample(self):
        raise NotImplementedError()
