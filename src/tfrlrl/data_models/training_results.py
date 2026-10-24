"""Dataclasses for the results returned by training algorithms."""

from dataclasses import dataclass

import numpy as np

from tfrlrl.policies.base import BasePyTorchPolicy


@dataclass
class PolicyGradientTrainingResult:
    """
    The result of training a policy with a policy-gradient algorithm.

    Args:
        policy: The trained policy.
        total_rewards: A 1-D array of length n_iterations, in which entry n is the mean total episodic reward
        observed during the n-th iteration of training.

    """

    policy: BasePyTorchPolicy
    total_rewards: np.ndarray
