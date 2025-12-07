"""Stochastic Gradient Descent training algorithm."""

import logging

import numpy as np

from tfrlrl.policies.base import BasePolicy
from tfrlrl.sampling.episodic_sampler import EpisodicSampler
from tfrlrl.sampling.statistics_collection import EpisocidPolicyGradientStatisticsCollector

logger = logging.getLogger(__name__)


def train_policy_gradient(
    env_id: str,
    policy: BasePolicy,
    n_iterations: int,
    n_episodes: int,
    alpha: float,
    **kwargs,
) -> BasePolicy:
    """
    Train a policy using stochastic gradient ascent on the policy gradient.

    :param env_id: Gymnasium environment ID (e.g., CartPole-v1, MountainCar-v0).
    :param policy: The policy to train. Must have get_parameters() and set_parameters() methods.
    :param n_iterations: The number of policy updates to perform.
    :param n_episodes: The number of episodes to sample during each policy update.
    :param alpha: The initial step size to take in stochastic gradient ascent.
    :param kwargs: Additional keyword arguments to pass to the EpisodicSampler (e.g., is_slippery).
    :return: The trained policy.
    """
    statistics_collector = EpisocidPolicyGradientStatisticsCollector(policy)

    sampler = EpisodicSampler(
        env_id=env_id,
        n_episodes=n_episodes,
        policy=policy,
        statistics_collector=statistics_collector,
        **kwargs,
    )

    for n in range(n_iterations):
        stats = [x for x in sampler]
        total_rewards = [np.sum(x[0]) for x in stats]
        policy_gradients = [x[1] for x in stats]

        policy_gradient = np.average(np.array(policy_gradients), axis=0)
        policy.set_parameters(policy.get_parameters() + (alpha / (n + 1)) * policy_gradient)

        if n % 10 == 0:
            logger.info('Policy update: %s', n)
            logger.info('Average total episodic reward: %s', np.average(np.array(total_rewards)))
            logger.info('Policy gradient magnitude: %s', np.sum(np.abs(policy_gradient)))

        sampler.reset()
        sampler.update_policy(policy)

    return policy
