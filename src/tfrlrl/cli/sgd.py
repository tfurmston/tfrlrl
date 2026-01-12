import argparse
import json
import logging

import gymnasium as gym
import numpy as np

from tfrlrl.features.onehot import construct_one_hot_feature_function
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.training_algorithms.sgd import train_policy_gradient

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

    env = gym.make(parsed_args.env_id)
    S = env.observation_space.n
    A = env.action_space.n

    feature_fn = construct_one_hot_feature_function(S=S, A=A)
    softmax_parameters = np.random.random(size=S * (A - 1))
    pol = LinearSoftMax(
        parsed_args.env_id,
        softmax_parameters,
        feature_fn,
    )
    train_policy_gradient(
        env_id=parsed_args.env_id,
        policy=pol,
        n_iterations=parsed_args.n_iterations,
        n_episodes=parsed_args.n_episodes,
        alpha=parsed_args.alpha,
        n_samplers=parsed_args.n_samplers,
        **env_kwargs,
    )
