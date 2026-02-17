"""Tests for the SGD training CLI."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.cli.sgd import main, parse_args


class TestParseArgs:
    """Tests for the parse_args function."""

    @pytest.mark.parametrize(
        'args,expected',
        [
            (
                ['--env-id', 'FrozenLake-v1', '--n-iterations', '10', '--n-episodes', '20', '--alpha', '1.0'],
                {'env_id': 'FrozenLake-v1', 'n_iterations': 10, 'n_episodes': 20, 'alpha': 1.0, 'env_kwargs': '{}'},
            ),
            (
                ['--env-id', 'FrozenLake-v1', '--env-kwargs', '{"is_slippery": false}'],
                {
                    'env_id': 'FrozenLake-v1',
                    'n_iterations': 100,
                    'n_episodes': 100,
                    'alpha': 100.0,
                    'env_kwargs': '{"is_slippery": false}',
                },
            ),
        ],
    )
    def test_parse_args(self, args, expected):
        """Test parsing command line arguments with various combinations."""
        parsed = parse_args(args)
        assert parsed.env_id == expected['env_id']
        assert parsed.n_iterations == expected['n_iterations']
        assert parsed.n_episodes == expected['n_episodes']
        assert parsed.alpha == expected['alpha']
        assert parsed.env_kwargs == expected['env_kwargs']

    def test_parse_args_missing_required(self):
        """Test that missing required arguments raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(['--n-iterations', '10'])


class TestMain:
    """Tests for the main function."""

    @pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
    @given(
        n_iterations=st.integers(min_value=1, max_value=3),
        n_episodes=st.integers(min_value=5, max_value=15),
        alpha=st.floats(min_value=0.0001, max_value=0.001),
    )
    @settings(deadline=10000)
    def test_main_with_different_parameters(self, env_id: str, n_iterations: int, n_episodes: int, alpha: float):
        """
        Test main function with different parameter values.

        :param env_id: The Gym environment ID to be used in training.
        :param n_iterations: The number of policy updates to perform.
        :param n_episodes: The number of episodes to sample during each policy update.
        :param alpha: The initial step size for stochastic gradient ascent.
        """
        args = [
            '--env-id',
            env_id,
            '--n-iterations',
            str(n_iterations),
            '--n-episodes',
            str(n_episodes),
            '--alpha',
            str(alpha),
        ]

        exit_code = main(args)

        assert exit_code is None or exit_code == 0

    @pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
    def test_main_with_env_kwargs(self, env_id: str):
        """
        Test main function with env-kwargs set.

        :param env_id: The Gym environment ID to be used in training.
        """
        args = [
            '--env-id',
            env_id,
            '--n-iterations',
            '2',
            '--n-episodes',
            '5',
            '--alpha',
            '1.0',
            '--env-kwargs',
            '{"reset_noise_scale": 0.001}',
        ]

        exit_code = main(args)

        assert exit_code is None or exit_code == 0

    @pytest.mark.parametrize('env_id', ['InvertedPendulum-v5'])
    def test_main_with_default_env_kwargs(self, env_id: str):
        """
        Test main function with default (empty) env-kwargs.

        :param env_id: The Gym environment ID to be used in training.
        """
        args = [
            '--env-id',
            env_id,
            '--n-iterations',
            '2',
            '--n-episodes',
            '5',
            '--alpha',
            '1.0',
        ]

        exit_code = main(args)

        assert exit_code is None or exit_code == 0

    @pytest.mark.parametrize(
        'malformed_json',
        [
            '{"is_slippery": false',  # Missing closing brace
            '{is_slippery: false}',  # Missing quotes around key
            '{"is_slippery": False}',  # Python bool instead of JSON bool
            'not valid json',  # Completely invalid
            '["is_slippery", false]',  # JSON array instead of object
            '123',  # JSON number instead of object
            '"string"',  # JSON string instead of object
        ],
    )
    def test_main_with_malformed_env_kwargs(self, malformed_json: str):
        """
        Test main function with malformed env-kwargs returns exit code 1.

        :param malformed_json: Malformed JSON string to test.
        """
        args = [
            '--env-id',
            'FrozenLake-v1',
            '--n-iterations',
            '1',
            '--n-episodes',
            '2',
            '--alpha',
            '1.0',
            '--env-kwargs',
            malformed_json,
        ]

        exit_code = main(args)

        assert exit_code == 1
