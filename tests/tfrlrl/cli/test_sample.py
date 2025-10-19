"""Tests for the sampling CLI."""
import pytest

from tfrlrl.cli.sample import collect_samples_parallel, collect_samples_single_env, compute_statistics, main, parse_args


@pytest.mark.parametrize(
    'args,expected',
    [
        (
            ['--env-id', 'CartPole-v1', '--n-steps', '100'],
            {'env_id': 'CartPole-v1', 'n_steps': 100, 'n_envs': 1},
        ),
        (
            ['--env-id', 'MountainCar-v0', '--n-steps', '500', '--n-envs', '4'],
            {'env_id': 'MountainCar-v0', 'n_steps': 500, 'n_envs': 4},
        ),
    ],
)
def test_parse_args(args, expected):
    """Test parsing command line arguments with various combinations."""
    parsed = parse_args(args)
    assert parsed.env_id == expected['env_id']
    assert parsed.n_steps == expected['n_steps']
    assert parsed.n_envs == expected['n_envs']


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


def test_collect_samples_single_env(test_ray_cluster):
    """Test collecting samples from a single environment."""
    env_id = 'CartPole-v1'
    n_steps = 10

    samples = collect_samples_single_env(env_id, n_steps)

    assert len(samples) == n_steps
    for sample in samples:
        assert hasattr(sample, 'env_id')
        assert hasattr(sample, 'observation')
        assert hasattr(sample, 'action')
        assert hasattr(sample, 'reward')
        assert hasattr(sample, 'done')


def test_collect_samples_parallel(test_ray_cluster):
    """Test collecting samples from parallel environments."""
    env_id = 'CartPole-v1'
    n_steps = 5
    n_envs = 2

    samples = collect_samples_parallel(env_id, n_steps, n_envs)

    assert len(samples) == n_steps
    for batch in samples:
        assert hasattr(batch, 'n_steps')
        assert batch.n_steps == n_envs
        assert hasattr(batch, 'observations')
        assert hasattr(batch, 'actions')
        assert hasattr(batch, 'rewards')


@pytest.mark.parametrize(
    'env_id,n_steps,is_parallel',
    [
        ('CartPole-v1', 20, False),
        ('CartPole-v1', 10, True),
    ],
)
def test_compute_statistics(env_id, n_steps, is_parallel, test_ray_cluster):
    """Test computing statistics for both single and parallel environment samples."""
    if is_parallel:
        n_envs = 2
        samples = collect_samples_parallel(env_id, n_steps, n_envs)
        expected_total_steps = n_steps * n_envs
    else:
        samples = collect_samples_single_env(env_id, n_steps)
        expected_total_steps = n_steps

    stats = compute_statistics(samples, is_parallel=is_parallel)

    assert stats['total_steps'] == expected_total_steps
    assert 'n_episodes' in stats
    assert 'mean_reward' in stats
    assert 'std_reward' in stats
    assert 'min_reward' in stats
    assert 'max_reward' in stats
    assert isinstance(stats['n_episodes'], int)
    assert isinstance(stats['mean_reward'], float)


@pytest.mark.parametrize(
    'env_id,n_steps,n_envs',
    [
        ('CartPole-v1', 10, 1),
        ('CartPole-v1', 20, 2),
    ],
)
def test_main(env_id, n_steps, n_envs, test_ray_cluster, caplog):
    """Test main function with various configurations."""
    args = ['--env-id', env_id, '--n-steps', str(n_steps)]
    if n_envs > 1:
        args.extend(['--n-envs', str(n_envs)])

    exit_code = main(args)

    assert exit_code == 0


def test_main_invalid_env(test_ray_cluster):
    """Test main function with invalid environment ID."""
    args = ['--env-id', 'InvalidEnv-v999', '--n-steps', '10']

    exit_code = main(args)

    assert exit_code == 1
