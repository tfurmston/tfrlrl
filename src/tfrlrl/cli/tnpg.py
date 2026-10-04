import argparse
import json
import logging

import gymnasium as gym

from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward
from tfrlrl.features.onehot import OneHotFeatureFunction
from tfrlrl.optimisation.trust_region import TrustRegionConfig
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy, DenseNetworkPolicyConfig
from tfrlrl.policies.linear_soft_max import LinearSoftMax
from tfrlrl.training_algorithms.tnpg import TNPGStepSizeConfig, train_policy_gradient

logger = logging.getLogger(__name__)


def parse_args(args=None):
    """
    Parse command line arguments for the truncated natural policy gradient CLI.

    :param args: Command line arguments to parse. If None, uses sys.argv.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train a policy using truncated natural policy gradient ascent',
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
        help='The base learning rate for the SGD optimizer used to apply the natural policy gradient.',
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
        help='The type of policy class to use in the truncated natural policy gradient algorithm.',
    )
    parser.add_argument(
        '--n-hidden',
        type=int,
        nargs='+',
        default=[16, 32],
        help='The number of hidden dimensions to use in a dense policy network.',
    )
    parser.add_argument(
        '--std-parameterisation',
        type=str,
        default='global',
        choices=['network', 'global'],
        help="How the standard deviation of a dense policy is parameterised: 'network' for a standard deviation "
        "that is a function of the observation, or 'global' for a single state-independent learned vector.",
    )
    parser.add_argument(
        '--init-std',
        type=float,
        default=1.0,
        help='The initial standard deviation of a dense policy when --std-parameterisation=global. Ignored otherwise.',
    )
    parser.add_argument(
        '--learn-std',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether the standard deviation of a dense policy is trainable when --std-parameterisation=global. '
        'Ignored otherwise.',
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
    parser.add_argument(
        '--n-iters-cg',
        type=int,
        default=None,
        help='The maximum number of conjugate-gradient iterations to perform when calculating the truncated '
        'natural policy gradient direction. Defaults to the default of calculate_conjugate_gradient.',
    )
    parser.add_argument(
        '--n-samples-fim',
        type=int,
        default=None,
        help='The number of state-action pairs to randomly subsample when calculating the Fisher Information '
        'matrix-vector product. When not given, all sampled state-action pairs are used.',
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=None,
        help='The Kullback-Leibler divergence budget for the trust-region step size calculation. When given, '
        'the step size used to update policy parameters is calculated from this trust-region formula at each '
        'iteration, with --alpha used only as a fallback value. When not given (the default), --alpha is used '
        'directly as a fixed step size.',
    )
    parser.add_argument(
        '--fim-reg-coeff',
        type=float,
        default=1e-5,
        help='A damping coefficient added to the diagonal of the Fisher Information matrix for numerical '
        'stability. Only used when --delta is given.',
    )
    parsed = parser.parse_args(args)
    if parsed.reward_model == 'discounted' and parsed.gamma is None:
        parser.error('--gamma is required when --reward-model=discounted')
    return parsed


def main(args=None):
    """
    Entry point for the truncated natural policy gradient CLI.

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
        policy_config = DenseNetworkPolicyConfig(
            hidden_space_dims=parsed_args.n_hidden,
            std_parameterisation=parsed_args.std_parameterisation,
            init_std=parsed_args.init_std,
            learn_std=parsed_args.learn_std,
        )
        policy = DenseNetworkPolicy(
            env_id=parsed_args.env_id,
            config=policy_config,
        )

    step_size_config = TNPGStepSizeConfig(
        lr=parsed_args.alpha,
        trust_region_config=TrustRegionConfig(delta=parsed_args.delta, reg_coeff=parsed_args.fim_reg_coeff)
        if parsed_args.delta is not None
        else None,
    )

    train_policy_gradient(
        env_id=parsed_args.env_id,
        policy=policy,
        n_iterations=parsed_args.n_iterations,
        n_episodes=parsed_args.n_episodes,
        step_size_config=step_size_config,
        n_samplers=parsed_args.n_samplers,
        reward_model=reward_model,
        n_iters_cg=parsed_args.n_iters_cg,
        n_samples_fim=parsed_args.n_samples_fim,
        **env_kwargs,
    )
