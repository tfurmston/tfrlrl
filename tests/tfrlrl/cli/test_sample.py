"""Tests for the sampling CLI."""

import gymnasium as gym
import pytest

from tfrlrl.cli.sample import compute_statistics, main, parse_args
from tfrlrl.sampling.sampler import Sampler


@pytest.mark.parametrize(
    'args,expected',
    [
        (
            ['--env-id', 'CartPole-v1', '--n-steps', '100'],
            {'env_id': 'CartPole-v1', 'n_steps': 100},
        ),
        (
            ['--env-id', 'MountainCar-v0', '--n-steps', '500'],
            {'env_id': 'MountainCar-v0', 'n_steps': 500},
        ),
    ],
)
def test_parse_args(args, expected):
    """Test parsing command line arguments with various combinations."""
    parsed = parse_args(args)
    assert parsed.env_id == expected['env_id']
    assert parsed.n_steps == expected['n_steps']


@pytest.mark.parametrize(
    'args',
    [
        ['--env-id', 'CartPole-v1'],  # Missing n-steps
        ['--n-steps', '100'],  # Missing env-id
    ],
)
def test_parse_args_missing_required(args):
    """Test that missing required arguments raises SystemExit."""
    with pytest.raises(SystemExit):
        parse_args(args)


@pytest.mark.parametrize(
    'env_id,n_steps',
    [
        ('CartPole-v1', 20),
        ('CartPole-v1', 10),
    ],
)
def test_compute_statistics(env_id, n_steps):
    """Test computing statistics for single environment samples."""
    sampler = Sampler(env_id=env_id, n_steps=n_steps)
    samples = [sample for sample in sampler]
    expected_total_steps = n_steps

    stats = compute_statistics(samples)

    assert stats['total_steps'] == expected_total_steps
    assert 'n_episodes' in stats
    assert 'mean_reward' in stats
    assert 'std_reward' in stats
    assert 'min_reward' in stats
    assert 'max_reward' in stats
    assert isinstance(stats['n_episodes'], int)
    assert isinstance(stats['mean_reward'], float)


@pytest.mark.parametrize(
    'env_id,n_steps',
    [
        ('CartPole-v1', 10),
        ('CartPole-v1', 20),
    ],
)
def test_main(env_id, n_steps, caplog):
    """Test main function with various configurations."""
    args = ['--env-id', env_id, '--n-steps', str(n_steps)]
    exit_code = main(args)

    assert exit_code == 0


def test_main_invalid_env():
    """Test main function with invalid environment ID."""
    args = ['--env-id', 'InvalidEnv-v999', '--n-steps', '10']

    with pytest.raises(gym.error.NameNotFound):
        main(args)
