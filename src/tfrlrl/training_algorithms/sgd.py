"""Stochastic Gradient Descent training algorithm."""

import logging

import numpy as np
import ray
from torch import (
    sum,
    tensor,
)
from torch.optim import (
    AdamW,
)

from tfrlrl import settings
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
    alpha: float,
    n_samplers: int = 1,
    **kwargs,
) -> BasePyTorchPolicy:
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
    statistics_collector = EpisocidPolicyGradientStatisticsCollector(env_id)
    optimizer = AdamW(policy.get_parameters(), lr=alpha)

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

        actions = statistics.actions
        actions = actions[np.newaxis, :]

        # Update the policy network
        optimizer.zero_grad()

        log_probabilities = policy.calculate_log_probabilities(
            observations=statistics.observations,
            actions=actions,
        )
        loss = -sum(log_probabilities * tensor(statistics.total_expected_rewards))

        loss.backward()
        optimizer.step()

        print(f'iteration: {n}')
        # print(statistics.observations.shape)
        # print(actions.shape)
        # print(log_probabilities.shape)
        # print(tensor(statistics.total_expected_rewards).shape)
        print(loss)

        if n % 10 == 0:
            logger.info('Policy update: %s', n)
            logger.info('Average total episodic reward: %s', np.average(statistics.total_reward))

        sampler.reset()
        sampler.update(state_dict=policy.get_state())

    return policy
