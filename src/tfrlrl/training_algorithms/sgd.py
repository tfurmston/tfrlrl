"""Stochastic Gradient Descent training algorithm."""

import logging

import numpy as np
import ray

from tfrlrl import settings
from tfrlrl.policies.base import BasePolicy
from tfrlrl.sampling.episodic_sampler import (
    EpisodicSampler,
    RayEpisodicSampler,
)
from tfrlrl.sampling.statistics_collection import EpisocidPolicyGradientStatisticsCollector

logger = logging.getLogger(__name__)


def train_policy_gradient(
    env_id: str,
    policy: BasePolicy,
    n_iterations: int,
    n_episodes: int,
    alpha: float,
    n_samplers: int = 1,
    **kwargs,
) -> BasePolicy:
    """
    Train a policy using stochastic gradient ascent on the policy gradient.

    :param env_id: Gymnasium environment ID (e.g., CartPole-v1, MountainCar-v0).
    :param policy: The policy to train. Must have get_parameters() and set_parameters() methods.
    :param n_iterations: The number of policy updates to perform.
    :param n_episodes: The number of episodes to sample during each policy update.
    :param alpha: The initial step size to take in stochastic gradient ascent.
    :param n_samplers: The number of samplers to used to sample from the environment.
    :param kwargs: Additional keyword arguments to pass to the EpisodicSampler (e.g., is_slippery).
    :return: The trained policy.
    """
    statistics_collector = EpisocidPolicyGradientStatisticsCollector(policy)

    if n_samplers > 1:
        if not ray.is_initialized():
            ray.init(
                num_cpus=settings.ray_cpu,
                ignore_reinit_error=True,
            )
            logger.info('Ray initialized for %s CPUs', settings.ray_cpu)
        sampler = RayEpisodicSampler(
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
        statistics = sampler.sample()
        policy_gradient = np.average(np.array(statistics.episode_gradient), axis=1)
        policy.set_parameters(policy.get_parameters() + (alpha / (n + 1)) * policy_gradient)

        if n % 10 == 0:
            logger.info('Policy update: %s', n)
            logger.info('Average total episodic reward: %s', np.average(statistics.total_reward))
            logger.info('Policy gradient magnitude: %s', np.sum(np.abs(policy_gradient)))

        sampler.reset()
        sampler.update_policy(policy)

    return policy
