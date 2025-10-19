"""CLI for sampling steps from Gymnasium environments."""
import argparse
import logging

import numpy as np
import ray

from tfrlrl.sampling.sampler import RaySampler, Sampler

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
    parser.add_argument(
        '--n-envs',
        type=int,
        default=1,
        help='Number of parallel environments to use for sampling',
    )
    return parser.parse_args(args)


def collect_samples_single_env(env_id: str, n_steps: int):
    """
    Collect samples from a single environment without Ray.

    :param env_id: The Gymnasium environment ID.
    :param n_steps: Number of steps to sample.
    :return: List of step samples.
    """
    sampler = Sampler.remote(env_id=env_id, n_steps=n_steps)
    samples = []
    for _ in range(n_steps):
        sample = ray.get(sampler.__next__.remote())
        samples.append(sample)
    return samples


def collect_samples_parallel(env_id: str, n_steps: int, n_envs: int):
    """
    Collect samples from multiple parallel environments using Ray.

    :param env_id: The Gymnasium environment ID.
    :param n_steps: Number of steps to sample per environment.
    :param n_envs: Number of parallel environments.
    :return: List of batched step samples.
    """
    ray_sampler = RaySampler(env_id=env_id, n_envs=n_envs, n_steps=n_steps)
    samples = []
    for steps in ray_sampler:
        samples.append(steps)
    return samples


def compute_statistics(samples, is_parallel: bool):
    """
    Compute and return statistics about collected samples.

    :param samples: List of step or steps samples.
    :param is_parallel: Whether samples are from parallel environments (Steps objects).
    :return: Dictionary of statistics.
    """
    if is_parallel:
        total_steps = sum(s.n_steps for s in samples)
        all_rewards = np.concatenate([s.rewards for s in samples])
        all_dones = np.concatenate([s.dones for s in samples])
    else:
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
    logger.info('Using %s parallel environment(s)', parsed_args.n_envs)

    try:
        # Initialize Ray (required for both single and parallel sampling)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        logger.info('Ray initialized')

        # Collect samples
        if parsed_args.n_envs > 1:
            # Calculate steps per environment
            steps_per_env = parsed_args.n_steps // parsed_args.n_envs
            logger.info('Sampling %s steps per environment across %s environments', steps_per_env, parsed_args.n_envs)

            samples = collect_samples_parallel(
                parsed_args.env_id,
                steps_per_env,
                parsed_args.n_envs,
            )
            is_parallel = True
        else:
            logger.info('Sampling %s steps from single environment', parsed_args.n_steps)
            samples = collect_samples_single_env(
                parsed_args.env_id,
                parsed_args.n_steps,
            )
            is_parallel = False

        # Compute statistics
        stats = compute_statistics(samples, is_parallel)

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

    except Exception:  # noqa: B902
        logger.exception('Error during sampling')
        return 1

    finally:
        # Shutdown Ray if it was initialized
        if ray.is_initialized():
            ray.shutdown()
            logger.info('Ray shutdown')
