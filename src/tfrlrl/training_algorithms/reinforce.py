"""Stochastic Gradient Descent training algorithm."""

import logging
from typing import Optional, Union

import numpy as np
import ray
from torch import (
    no_grad,
    sum,
    tensor,
)
from torch.optim import (
    Optimizer,
)

from tfrlrl import settings
from tfrlrl.baselines.linear import Baseline
from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward
from tfrlrl.policies.base import BasePyTorchPolicy
from tfrlrl.sampling.episodic_sampler import (
    EpisodicSampler,
    RayEpisodicSampler,
)
from tfrlrl.sampling.statistics_collection import EpisocidPolicyGradientStatisticsCollector

logger = logging.getLogger(__name__)


def train_policy_gradient(
    env_id: str,
    policy: BasePyTorchPolicy,
    n_iterations: int,
    n_episodes: int,
    optimizer: Optimizer,
    n_samplers: int = 1,
    baseline: Optional[Baseline] = None,
    reward_model: Optional[Union[AverageEpisodicReward, DiscountedReward]] = None,
    n_iteration_logging: int = 10,
    **kwargs,
) -> BasePyTorchPolicy:
    """
    Train a policy using stochastic gradient ascent on the policy gradient.

    Args:
        env_id: Gymnasium environment ID (e.g., CartPole-v1, MountainCar-v0).
        policy: The policy to train. Must have get_parameters() and set_parameters() methods.
        n_iterations: The number of policy updates to perform.
        n_episodes: The number of episodes to sample during each policy update.
        optimizer: An instance of a PyTorch optimizer class that will be used to optimise the policy.
        n_samplers: The number of samplers to used to sample from the environment.
        baseline: An instance of a baseline class, if one is given.
        reward_model: The reward model to use when computing total expected rewards. Defaults to
        AverageEpisodicReward if not specified.
        n_iteration_logging: The number of algorithm iterations between logging algorithm performance.
        kwargs: Additional keyword arguments to pass to the EpisodicSampler (e.g., is_slippery).

    Returns:
        The trained policy.

    """
    statistics_collector = EpisocidPolicyGradientStatisticsCollector(
        env_id,
        baseline=baseline,
        reward_model=reward_model,
    )

    if n_samplers > 1:
        if not ray.is_initialized():
            ray.init(
                num_cpus=settings.ray_cpu,
                ignore_reinit_error=True,
            )
            logger.info('Ray initialized for %s CPUs', settings.ray_cpu)
        sampler: Union[RayEpisodicSampler, EpisodicSampler] = RayEpisodicSampler(
            n_samplers=n_samplers,
            env_id=env_id,
            n_episodes=n_episodes,
            policy=policy,
            statistics_collector=statistics_collector,
            **kwargs,
        )
    else:
        sampler = EpisodicSampler(
            env_id=env_id,
            n_episodes=n_episodes,
            policy=policy,
            statistics_collector=statistics_collector,
            **kwargs,
        )

    for n in range(n_iterations):
        with no_grad():
            statistics = sampler.sample()

        optimizer.zero_grad()
        log_probabilities = policy.calculate_log_probabilities(
            observations=statistics.observations,
            actions=statistics.actions,
        )
        loss = -sum(log_probabilities * tensor(statistics.total_expected_rewards))
        loss.backward()
        optimizer.step()

        if n % n_iteration_logging == 0:
            logger.info('Policy update: %s', n)
            logger.info('Average total episodic reward: %s', np.average(statistics.total_reward))

        if baseline:
            baseline.fit(statistics.baseline_features, statistics.baseline_targets)

        sampler.reset()
        sampler.update(
            policy_state_dict=policy.get_state(),
            baseline_state_dict=baseline.get_state() if baseline else None,
        )

    return policy
