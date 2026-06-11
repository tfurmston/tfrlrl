"""Reward model dataclasses for different ways of estimating total expected reward."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AverageEpisodicReward:
    """Average episodic return: G_t = (r_t + r_{t+1} + ... + r_{T-1}) / T."""

    def compute(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute the average episodic return for each step.

        Args:
            rewards: 1-D array of rewards of length T.

        Returns:
            1-D array of length T where each element is the average future return from that step.

        """
        T = rewards.size
        return np.matmul(rewards, np.tril(np.ones(T))) / T


@dataclass
class DiscountedReward:
    """Discounted return: G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(T-1-t)·r_{T-1}."""

    gamma: float

    def compute(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute the discounted return for each step.

        Args:
            rewards: 1-D array of rewards of length T.

        Returns:
            1-D array of length T where each element is the discounted future return from that step.

        """
        T = rewards.size
        i_idx = np.arange(T)[:, None]
        j_idx = np.arange(T)[None, :]
        discount_matrix = np.where(i_idx >= j_idx, np.power(self.gamma, i_idx - j_idx), 0.0)
        return np.matmul(rewards, discount_matrix)
