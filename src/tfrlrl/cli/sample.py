"""CLI for sampling steps from Gymnasium environments."""

import argparse
import logging

import numpy as np

from tfrlrl.sampling.sampler import Sampler

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
        '--n-steps',
        type=int,
        required=True,
        help='Total number of steps to sample',
    )
    return parser.parse_args(args)


def compute_statistics(samples):
    """
    Compute and return statistics about collected samples.

    :param samples: List of step or steps samples.
    :return: Dictionary of statistics.
    """
    total_steps = len(samples)
    all_rewards = np.array([s.reward for s in samples])
    all_dones = np.array([s.done for s in samples])

    n_episodes = int(np.sum(all_dones))
    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    min_reward = float(np.min(all_rewards))
    max_reward = float(np.max(all_rewards))

    return {
        'total_steps': total_steps,
        'n_episodes': n_episodes,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
    }


def main(args=None):
    """
    Entry point for the sampling CLI.

    :param args: Command line arguments. If None, uses sys.argv.
    :return: Exit code (0 for success, 1 for failure).
    """
    parsed_args = parse_args(args)

    logger.info('Sampling %s steps from %s', parsed_args.n_steps, parsed_args.env_id)

    # Collect samples
    logger.info('Sampling %s steps from environment', parsed_args.n_steps)
    sampler = Sampler(env_id=parsed_args.env_id, n_steps=parsed_args.n_steps)

    samples = [sample for sample in sampler]

    # Compute statistics
    stats = compute_statistics(samples)

    # Log summary
    logger.info('=' * 60)
    logger.info('SAMPLING SUMMARY')
    logger.info('=' * 60)
    logger.info('Environment:        %s', parsed_args.env_id)
    logger.info('Total steps:        %d', stats['total_steps'])
    logger.info('Episodes completed: %d', stats['n_episodes'])
    logger.info('Mean reward:        %.4f ± %.4f', stats['mean_reward'], stats['std_reward'])
    logger.info('Reward range:       [%.4f, %.4f]', stats['min_reward'], stats['max_reward'])
    logger.info('=' * 60)

    return 0
