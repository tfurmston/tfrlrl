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
                [
                    '--env-id',
                    'FrozenLake-v1',
                    '--n-iterations',
                    '10',
                    '--n-episodes',
                    '20',
                    '--alpha',
                    '1.0',
                    '--policy-class',
                    'dense',
                ],
                {
                    'env_id': 'FrozenLake-v1',
                    'n_iterations': 10,
                    'n_episodes': 20,
                    'alpha': 1.0,
                    'env_kwargs': '{}',
                    'policy_class': 'dense',
                },
            ),
            (
                ['--env-id', 'FrozenLake-v1', '--env-kwargs', '{"is_slippery": false}', '--policy-class', 'dense'],
                {
                    'env_id': 'FrozenLake-v1',
                    'n_iterations': 100,
                    'n_episodes': 100,
                    'alpha': 100.0,
                    'env_kwargs': '{"is_slippery": false}',
                    'policy_class': 'dense',
                },
            ),
            (
                [
                    '--env-id',
                    'FrozenLake-v1',
                    '--n-iterations',
                    '10',
                    '--n-episodes',
                    '20',
                    '--alpha',
                    '1.0',
                    '--policy-class',
                    'linear',
                ],
                {
                    'env_id': 'FrozenLake-v1',
                    'n_iterations': 10,
                    'n_episodes': 20,
                    'alpha': 1.0,
                    'env_kwargs': '{}',
                    'policy_class': 'linear',
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

    @pytest.mark.parametrize(
        'env_id, policy_class',
        [
            (
                'InvertedPendulum-v5',
                'dense',
            ),
            (
                'FrozenLake-v1',
                'linear',
            ),
        ],
    )
    @given(
        n_iterations=st.integers(min_value=1, max_value=3),
        n_episodes=st.integers(min_value=5, max_value=15),
        alpha=st.floats(min_value=0.0001, max_value=0.001),
    )
    @settings(deadline=10000)
    def test_main_with_different_parameters(
        self, env_id: str, policy_class: str, n_iterations: int, n_episodes: int, alpha: float
    ):
        """
        Test main function with different parameter values.

        Args:
            env_id: The Gym environment ID to be used in training.
            policy_class: The policy class to use in SGD.
            n_iterations: The number of policy updates to perform.
            n_episodes: The number of episodes to sample during each policy update.
            alpha: The initial step size for stochastic gradient ascent.

        """
        args = [
            '--env-id',
            env_id,
            '--policy-class',
            policy_class,
            '--n-iterations',
            str(n_iterations),
            '--n-episodes',
            str(n_episodes),
            '--alpha',
            str(alpha),
        ]

        exit_code = main(args)

        assert exit_code is None or exit_code == 0

    @pytest.mark.parametrize(
        'env_id, policy_class, env_kwargs',
        [
            ('InvertedPendulum-v5', 'dense', '{"reset_noise_scale": 0.001}'),
            (
                'FrozenLake-v1',
                'linear',
                '{"is_slippery": false}',
            ),
        ],
    )
    def test_main_with_env_kwargs(self, env_id: str, policy_class: str, env_kwargs: str):
        """
        Test main function with env-kwargs set.

        Args:
            env_id: The Gym environment ID to be used in training.
            policy_class: The policy class to use in SGD.
            env_kwargs: Any key-words passed to the environment.

        """
        args = [
            '--env-id',
            env_id,
            '--policy-class',
            policy_class,
            '--n-iterations',
            '2',
            '--n-episodes',
            '5',
            '--alpha',
            '1.0',
            '--env-kwargs',
            env_kwargs,
        ]

        exit_code = main(args)

        assert exit_code is None or exit_code == 0

    @pytest.mark.parametrize(
        'env_id, policy_class',
        [
            (
                'InvertedPendulum-v5',
                'dense',
            ),
            (
                'FrozenLake-v1',
                'linear',
            ),
        ],
    )
    def test_main_with_default_env_kwargs(self, env_id: str, policy_class: str):
        """
        Test main function with default (empty) env-kwargs.

        Args:
            env_id: The Gym environment ID to be used in training.
            policy_class: The policy class to use in SGD.

        """
        args = [
            '--env-id',
            env_id,
            '--policy-class',
            policy_class,
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

        Args:
            malformed_json: Malformed JSON string to test.

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
            '--policy-class',
            'dense',
        ]

        exit_code = main(args)

        assert exit_code == 1
