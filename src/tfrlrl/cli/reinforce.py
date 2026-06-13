import argparse
import json
import logging

import gymnasium as gym
from torch.optim import (
    AdamW,
)

from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward
from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.training_algorithms.reinforce import train_policy_gradient

logging.basicConfig(format='%(asctime)s %(message)s', force=True)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """
    Parse command line arguments for the sampling CLI.

    :param args: Command line arguments to parse. If None, uses sys.argv.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Sample steps from a Gymnasium environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--env-id',
        type=str,
        required=True,
        help='Gymnasium environment ID (e.g., CartPole-v1, MountainCar-v0)',
    )
    parser.add_argument(
        '--n-iterations',
        type=int,
        default=100,
        help='The number of policy updates to perform.',
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100,
        help='The number of episodes to sample during each policy update.',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=100.0,
        help='The initial step size to take in stochastic gradient ascent.',
    )
    parser.add_argument(
        '--n-samplers',
        type=int,
        default=1,
        help='The number of samplers to use when sampling episodes.',
    )
    parser.add_argument(
        '--env-kwargs',
        type=str,
        default='{}',
        help='Environment-specific keyword arguments as a JSON string (e.g., \'{"is_slippery": false}\').',
    )
    parser.add_argument(
        '--policy-class',
        type=str,
        required=True,
        choices=['linear', 'dense'],
        help='The type of policy class to use in the stochastic gradient algorithm.',
    )
    parser.add_argument(
        '--n-hidden',
        type=int,
        nargs='+',
        default=[16, 32],
        help='The number of hidden dimensions to use in a dense policy network.',
    )
    parser.add_argument(
        '--reward-model',
        type=str,
        default='average-episodic',
        choices=['average-episodic', 'discounted'],
        help='Reward model to use when computing returns.',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=None,
        help='Discount factor for the discounted reward model, must be in (0, 1). '
        'Required when --reward-model=discounted.',
    )
    return parser.parse_args(args)


def main(args=None):
    """
    Entry point for the sampling CLI.

    :param args: Command line arguments. If None, uses sys.argv.
    :return: Exit code (0 for success, 1 for failure).
    """
    parsed_args = parse_args(args)

    # Parse environment kwargs from JSON string
    try:
        env_kwargs = json.loads(parsed_args.env_kwargs)
    except json.JSONDecodeError as e:
        logger.error('Failed to parse --env-kwargs as JSON: %s', e)
        return 1

    if not isinstance(env_kwargs, dict):
        logger.error('--env-kwargs must be a JSON object (dictionary), got: %s', type(env_kwargs).__name__)
        return 1
    if env_kwargs is not None:
        logger.info('Environment Arguments: %s', env_kwargs)

    if parsed_args.reward_model == 'discounted':
        if parsed_args.gamma is None:
            logger.error('--gamma is required when --reward-model=discounted')
            return 1
        try:
            reward_model = DiscountedReward(gamma=parsed_args.gamma)
        except (TypeError, ValueError) as e:
            logger.error('Invalid --gamma value: %s', e)
            return 1
    else:
        reward_model = AverageEpisodicReward()

    if parsed_args.policy_class == 'linear':
        logger.info('Using a linear policy with a one-hot feature encoding.')
        env = gym.make(parsed_args.env_id)
        feature_fn = OneHotFeatureFunction(env.observation_space.n, env.action_space.n)
        policy = LinearSoftMax(parsed_args.env_id, feature_fn)
    else:
        logger.info('Using a dense policy with hidden dimensions: %s', parsed_args.n_hidden)
        policy = DenseNetworkPolicy(
            env_id=parsed_args.env_id,
            hidden_space_dims=parsed_args.n_hidden,
        )

    optimizer = AdamW(policy.get_parameters(), lr=parsed_args.alpha)

    train_policy_gradient(
        env_id=parsed_args.env_id,
        policy=policy,
        n_iterations=parsed_args.n_iterations,
        n_episodes=parsed_args.n_episodes,
        optimizer=optimizer,
        n_samplers=parsed_args.n_samplers,
        reward_model=reward_model,
        **env_kwargs,
    )
