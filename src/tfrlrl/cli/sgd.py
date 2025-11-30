import argparse
import logging

import gymnasium as gym
import numpy as np

from tfrlrl.features.onehot import construct_one_hot_feature_function
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.sampling.episodic_sampler import EpisodicSampler

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
    return parser.parse_args(args)


def main(args=None):
    """
    Entry point for the sampling CLI.

    :param args: Command line arguments. If None, uses sys.argv.
    :return: Exit code (0 for success, 1 for failure).
    """
    parsed_args = parse_args(args)

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

    for n in range(parsed_args.n_iterations):
        sampler = EpisodicSampler(
            env_id=parsed_args.env_id,
            n_episodes=parsed_args.n_episodes,
            policy=pol,
            is_slippery=False,
        )

        x = [(rewards, episode_pol_gradient) for rewards, episode_pol_gradient in sampler]

        policy_gradients = [y[1] for y in x]
        total_rewatds = [np.sum(y[0]) for y in x]

        policy_gradient = np.average(np.array(policy_gradients), axis=0)
        pol.set_parameters(pol.get_parameters() + (parsed_args.alpha / (n + 1)) * policy_gradient)

        if n % 10 == 0:
            logger.info('Policy update: %s', n)
            logger.info('Average total episodic reward: %s', np.average(np.array(total_rewatds)))
            logger.info('Policy gradient magnitude: %s', np.sum(np.abs(policy_gradient)))
