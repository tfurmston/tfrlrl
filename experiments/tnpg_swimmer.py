import argparse
import logging
import random

import numpy as np
import torch

from tfrlrl.baselines.linear import LinearBaseline
from tfrlrl.data_models.reward_models import DiscountedReward
from tfrlrl.evaluation.video import construct_video_evaluation
from tfrlrl.optimisation.trust_region import TrustRegionConfig
from tfrlrl.policies.dense_neural_network import (
    DenseNetworkPolicy,
    DenseNetworkPolicyConfig,
)
from tfrlrl.training_algorithms.tnpg import (
    TNPGConjugateGradientConfig,
    TNPGStepSizeConfig,
    train_policy_gradient,
)

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
    Entry point for the evaluation of TNPG on the swimmer environment.

    :param args: Command line arguments. If None, uses sys.argv.
    :return: Exit code (0 for success, 1 for failure).
    """
    parsed_args = parse_args(args)

    environment_id = 'Swimmer-v5'
    reward_model = DiscountedReward(gamma=0.999999)
    hidden_space_dims = [100, 50, 25]

    n_iterations = 500
    n_episodes = 100
    n_samplers = 1
    n_eval_episodes = 4
    cg_config = TNPGConjugateGradientConfig(
        n_iters=10,
        n_samples_fim=1000,
    )
    step_size_config = TNPGStepSizeConfig(
        trust_region_config=TrustRegionConfig(
            delta=0.01,
        ),
    )

    for i, seed in enumerate([1]):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        policy_config = DenseNetworkPolicyConfig(
            hidden_space_dims=hidden_space_dims,
            std_parameterisation='global',
        )
        policy = DenseNetworkPolicy(
            env_id=environment_id,
            config=policy_config,
        )

        baseline = LinearBaseline(environment_id)

        construct_video_evaluation(
            environment_id,
            policy,
            n_eval_episodes,
            '/'.join([parsed_args.video_directory, f'initial_policy_{i}']),
            parsed_args.video_prefix,
        )

        policy = train_policy_gradient(
            env_id=environment_id,
            policy=policy,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            baseline=baseline,
            step_size_config=step_size_config,
            reward_model=reward_model,
            n_samplers=n_samplers,
            cg_config=cg_config,
            n_iteration_logging=10,
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
