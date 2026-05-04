"""
Script provides a recreation of the results in the tutorial, https://gymnasium.farama.org/tutorials/training_agents/mujoco_reinforce/.

The hyperparameters are as given in the above tutorial. The script saves video evaluations of the trained policy to the
specified path.
"""

import argparse
import logging
import random

import numpy as np
import torch
from torch.optim import (
    AdamW,
)

from tfrlrl.evaluation.video import construct_video_evaluation
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy
from tfrlrl.training_algorithms.reinforce import train_policy_gradient

logging.basicConfig(format='%(asctime)s %(message)s', force=True)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """
    Parse command line arguments for the evaluation of REINFORCE on the Inverted Pendulem problem.

    :param args: Command line arguments to parse. If None, uses sys.argv.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Sample steps from a Gymnasium environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--video-directory',
        type=str,
        required=True,
        help='The directory to which evaluation videos should be saved.',
    )
    parser.add_argument(
        '--video-prefix',
        type=str,
        required=True,
        help='The prefix of the evaluation video files.',
    )
    return parser.parse_args(args)


def main(args=None):
    """
    Entry point for the evaluation of REINFORCE on the inverted pendulum environment.

    :param args: Command line arguments. If None, uses sys.argv.
    :return: Exit code (0 for success, 1 for failure).
    """
    parsed_args = parse_args(args)

    environment_id = 'InvertedPendulum-v5'
    hidden_space_dims = [16, 32]

    n_iterations = 25000
    n_episodes = 1
    n_samplers = 1
    alpha = 1e-4
    n_eval_episodes = 4

    for i, seed in enumerate([1]):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        policy = DenseNetworkPolicy(
            env_id=environment_id,
            hidden_space_dims=hidden_space_dims,
        )
        optimizer = AdamW(policy.get_parameters(), lr=alpha)

        policy = train_policy_gradient(
            env_id=environment_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            optimizer=optimizer,
            n_samplers=n_samplers,
            n_iteration_logging=1000,
        )

        construct_video_evaluation(
            environment_id,
            policy,
            n_eval_episodes,
            '/'.join([parsed_args.video_directory, f'training_run_{i}']),
            parsed_args.video_prefix,
        )


if __name__ == '__main__':
    main()
