"""Reward model dataclasses for different return estimation paradigms."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AverageEpisodicReward:
    """Average episodic return: G_t = (r_t + r_{t+1} + ... + r_{T-1}) / T."""

    def compute(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute the average episodic return for each step.

        Args:
            rewards: A 1-D array of per-step rewards of length T.

        Returns:
            A 1-D array of length T where entry t is the sum of future rewards divided by T.

        """
        T = rewards.size
        return np.matmul(rewards, np.tril(np.ones(T))) / T


@dataclass
class DiscountedReward:
    """Discounted return: G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^(T-1-t) * r_{T-1}."""

    gamma: float

    def __post_init__(self):
        """Validate that gamma is strictly within the open interval (0, 1)."""
        if not isinstance(self.gamma, float):
            raise TypeError(f'gamma must be a float, got {type(self.gamma).__name__!r}')
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(f'gamma must be in the open interval (0, 1), got {self.gamma}')

    def compute(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute the discounted return for each step.

        Args:
            rewards: A 1-D array of per-step rewards of length T.

        Returns:
            A 1-D array of length T where entry t is the discounted sum of future rewards.

        """
        T = rewards.size
        i_idx = np.arange(T)[:, None]
        j_idx = np.arange(T)[None, :]
        discount_matrix = np.where(i_idx >= j_idx, np.power(self.gamma, i_idx - j_idx), 0.0)
        return np.matmul(rewards, discount_matrix)
