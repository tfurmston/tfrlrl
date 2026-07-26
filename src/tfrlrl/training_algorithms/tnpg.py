"""Stochastic Gradient Descent training algorithm."""

import logging
from typing import Callable, Optional, Union

import numpy as np
import ray
from torch import no_grad, sum, tensor
from torch.optim import (
    SGD,
    Optimizer,
)
from torch.optim.lr_scheduler import LRScheduler

from tfrlrl import settings
from tfrlrl.baselines.linear import Baseline
from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward
from tfrlrl.optimisation.conjugate_gradients import calculate_conjugate_gradient
from tfrlrl.policies.base import BasePyTorchPolicy
from tfrlrl.policies.utils import flatten_tensor_dict, unflatten_tensor_dict
from tfrlrl.sampling.episodic_sampler import (
    EpisodicSampler,
    RayEpisodicSampler,
)
from tfrlrl.sampling.statistics_collection import (
    EpisocidPolicyGradientStatisticsCollector,
    EpisodePolicyGradientStatistics,
)

logger = logging.getLogger(__name__)


def calculate_steepest_gradient_direction(
    policy: BasePyTorchPolicy, statistics: EpisodePolicyGradientStatistics, optimizer: Optimizer
) -> np.ndarray:
    """
    Calculate the direction of steepest gradient ascent of the policy.

    This function calculates the steepest gradient ascent direction, i.e., standard policy gradients, and returns
    it in the form of a vector (i.e. NumPy array).

    Args:
        policy: The policy for which the Fisher Infromation matrix is to be calculated.
        statistics: The statistics over which the expectation within the Fisher Information matrix calculation
        is to be performed.
        optimizer: A PyTorch optimizer class that will be used to calculate the policy gradient.

    Returns:
        A NumPy array containing the policy gradient.

    """
    optimizer.zero_grad()
    log_probabilities = policy.calculate_log_probabilities(
        observations=statistics.observations,
        actions=statistics.actions,
    )
    loss = -sum(log_probabilities * tensor(statistics.total_expected_rewards))
    loss.backward()

    return (
        flatten_tensor_dict(
            {name: param.grad for name, param in policy.network.named_parameters() if param.grad is not None},
        )
        .detach()
        .numpy()
    )


def construct_fim_vector_product_fn(
    policy: BasePyTorchPolicy, statistics: EpisodePolicyGradientStatistics
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct function for calculating the product of the Fisher Information matrix with a vector.

    This function constructs a function that when called with a vector (NumPy array) will return the
    product of the Fisher Information matrix with that vector. The Fisher Information matrix is calculated
    w.r.t. the given policy, while the expetation in Fisher Information matrix is calculated over the sample
      state-action pairs in the given statistics.

    Args:
        policy: The policy for which the Fisher Infromation matrix is to be calculated.
        statistics: The statistics over which the expectation within the Fisher Information matrix calculation
        is to be performed.

    Returns:
        A callable that takes an input vector as an argument and returns the product of the Fisher Information
        matrix with that vector.

    """
    jacobian = policy.calculate_jacobian(
        observations=statistics.observations,
        actions=statistics.actions,
    )
    jacobian_matrix = (
        flatten_tensor_dict(
            jacobian,
            dim=statistics.actions.ndim,
        )
        .detach()
        .numpy()
    )

    if jacobian_matrix.ndim > 3 or (jacobian_matrix.ndim > 2 and jacobian_matrix.shape[0] > 1):
        raise RuntimeError('Unsupported shape of Jacobian matrix: %s', jacobian_matrix.shape)
    elif jacobian_matrix.ndim > 2:
        jacobian_matrix = jacobian_matrix.squeeze()

    def calculate_fim_vector_product(v: np.ndarray):
        return np.matmul(jacobian_matrix.T, np.matmul(jacobian_matrix, v))

    return calculate_fim_vector_product


def train_policy_gradient(
    env_id: str,
    policy: BasePyTorchPolicy,
    n_iterations: int,
    n_episodes: int,
    lr: float,
    lr_scheduler_fn: Optional[Callable[[Optimizer], LRScheduler]] = None,
    n_samplers: int = 1,
    baseline: Optional[Baseline] = None,
    reward_model: Optional[Union[AverageEpisodicReward, DiscountedReward]] = None,
    n_iteration_logging: int = 10,
    **kwargs,
) -> BasePyTorchPolicy:
    """
    Train a policy using truncated natural policy gradient ascent.

    Args:
        env_id: Gymnasium environment ID (e.g., CartPole-v1, MountainCar-v0).
        policy: The policy to train. Must have get_parameters() and set_parameters() methods.
        n_iterations: The number of policy updates to perform.
        n_episodes: The number of episodes to sample during each policy update.
        lr: The base learning rate for the SGD optimizer used to apply the natural policy gradient.
        lr_scheduler_fn: An optional factory that, given the SGD optimizer instantiated internally,
        returns an LRScheduler wrapping it. When not given, the learning rate stays constant at lr.
        n_samplers: The number of samplers to used to sample from the environment.
        baseline: An instance of a baseline class, if one is given.
        reward_model: The reward model to use when computing total expected rewards. Defaults to
        AverageEpisodicReward if not specified.
        n_iteration_logging: The number of algorithm iterations between logging algorithm performance.
        kwargs: Additional keyword arguments to pass to the EpisodicSampler (e.g., is_slippery).

    Returns:
        The trained policy.

    """
    optimizer = SGD(policy.network.parameters(), lr=lr, maximize=True)
    lr_scheduler = lr_scheduler_fn(optimizer) if lr_scheduler_fn is not None else None

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

        logger.debug('Calculate truncated-natural policy gradient.')
        sgd = calculate_steepest_gradient_direction(
            policy=policy,
            statistics=statistics,
            optimizer=optimizer,
        )
        tngd = calculate_conjugate_gradient(
            mat_v_mult_fn=construct_fim_vector_product_fn(
                policy=policy,
                statistics=statistics,
            ),
            b=sgd,
            n_iters=1,
        )
        print(tngd)
        logger.debug('Update policy parameters.')
        tngd_dict = unflatten_tensor_dict(
            tensor(tngd),
            reference={name: param for name, param in policy.network.named_parameters()},
            dim=0,
        )
        for name, param in policy.network.named_parameters():
            param.grad = tngd_dict[name]
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

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
