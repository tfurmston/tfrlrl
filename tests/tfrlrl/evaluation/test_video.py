import logging
import os
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.evaluation.video import construct_video_evaluation


def _make_mock_env(n_steps_per_episode: int = 1):
    """
    Create a mock gymnasium environment that terminates after n_steps_per_episode steps.

    Args:
        n_steps_per_episode: The number of steps before the episode terminates.

    Returns:
        A MagicMock that mimics a gymnasium environment.

    """
    mock_env = MagicMock()
    mock_env.reset.return_value = (np.zeros(4), {})
    mock_env.action_space = gym.spaces.Discrete(2)

    step_state = {'count': 0}

    def step_fn(action):
        step_state['count'] += 1
        terminated = step_state['count'] % n_steps_per_episode == 0
        if terminated:
            step_state['count'] = 0
        return (np.zeros(4), 1.0, terminated, False, {})

    mock_env.step.side_effect = step_fn
    return mock_env


class TestConstructVideoEvaluation:
    """Unit tests for the construct_video_evaluation function."""

    @pytest.mark.parametrize(
        'env_id, n_episodes, name_prefix',
        [
            ('CartPole-v1', 1, 'test'),
            ('CartPole-v1', 3, 'eval'),
        ],
    )
    def test_construct_video_evaluation_creates_video_folder(
        self,
        env_id: str,
        n_episodes: int,
        name_prefix: str,
        tmp_path,
    ):
        """
        Test that construct_video_evaluation creates the video folder if it does not exist.

        Args:
            env_id: The Gym environment ID.
            n_episodes: The number of evaluation episodes.
            name_prefix: The prefix for video file names.
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env()
        policy = MagicMock()
        policy.generate_action.return_value = 0

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env),
        ):
            construct_video_evaluation(
                env_id=env_id,
                policy=policy,
                n_episodes=n_episodes,
                video_folder=video_folder,
                name_prefix=name_prefix,
            )

        assert os.path.isdir(video_folder)

    @pytest.mark.parametrize('env_id', ['CartPole-v1'])
    @given(n_episodes=st.integers(min_value=1, max_value=5))
    @settings(deadline=2000)
    def test_construct_video_evaluation_calls_policy_once_per_episode(
        self,
        env_id: str,
        n_episodes: int,
    ):
        """
        Test that construct_video_evaluation calls policy.generate_action once per episode.

        With a mock environment that terminates in a single step, generate_action should
        be called exactly n_episodes times.

        Args:
            env_id: The Gym environment ID.
            n_episodes: The number of evaluation episodes.

        """
        mock_env = _make_mock_env(n_steps_per_episode=1)
        policy = MagicMock()
        policy.generate_action.return_value = 0

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env),
        ):
            construct_video_evaluation(
                env_id=env_id,
                policy=policy,
                n_episodes=n_episodes,
                video_folder='/tmp/videos',
                name_prefix='test',
            )

        assert policy.generate_action.call_count == n_episodes

    @pytest.mark.parametrize(
        'env_id, n_episodes',
        [
            ('CartPole-v1', 1),
            ('CartPole-v1', 3),
            ('CartPole-v1', 5),
        ],
    )
    def test_construct_video_evaluation_logs_one_record_per_episode_plus_summary(
        self,
        env_id: str,
        n_episodes: int,
        tmp_path,
        caplog,
    ):
        """
        Test that construct_video_evaluation emits one INFO log record per episode, plus a summary record.

        Args:
            env_id: The Gym environment ID.
            n_episodes: The number of evaluation episodes.
            tmp_path: Pytest temporary directory fixture.
            caplog: Pytest log capture fixture.

        """
        mock_env = _make_mock_env()
        policy = MagicMock()
        policy.generate_action.return_value = 0

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env),
        ):
            with caplog.at_level(logging.INFO, logger='tfrlrl.evaluation.video'):
                construct_video_evaluation(
                    env_id=env_id,
                    policy=policy,
                    n_episodes=n_episodes,
                    video_folder=str(tmp_path / 'videos'),
                    name_prefix='test',
                )

        assert len(caplog.records) == n_episodes + 1
        for record in caplog.records[:-1]:
            assert record.levelno == logging.INFO
            assert 'steps' in record.message
            assert 'reward' in record.message
        assert 'summary' in caplog.records[-1].message.lower()

    @pytest.mark.parametrize(
        'n_steps_per_episode, expected_reward',
        [
            (1, 1.0),
            (3, 3.0),
            (5, 5.0),
        ],
    )
    def test_construct_video_evaluation_logs_accumulated_episode_reward(
        self,
        n_steps_per_episode: int,
        expected_reward: float,
        tmp_path,
        caplog,
    ):
        """
        Test that construct_video_evaluation logs the correctly accumulated episode reward.

        The mock environment returns a reward of 1.0 per step, so the logged total reward
        should equal n_steps_per_episode.

        Args:
            n_steps_per_episode: The number of steps in the single evaluation episode.
            expected_reward: The expected accumulated reward for the episode.
            tmp_path: Pytest temporary directory fixture.
            caplog: Pytest log capture fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env(n_steps_per_episode=n_steps_per_episode)
        policy = MagicMock()
        policy.generate_action.return_value = 0

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env),
        ):
            with caplog.at_level(logging.INFO, logger='tfrlrl.evaluation.video'):
                construct_video_evaluation(
                    env_id='CartPole-v1',
                    policy=policy,
                    n_episodes=1,
                    video_folder=video_folder,
                    name_prefix='test',
                )

        assert len(caplog.records) == 2
        assert str(expected_reward) in caplog.records[0].message

    def test_construct_video_evaluation_closes_env(self, tmp_path):
        """
        Test that construct_video_evaluation closes the environment after all episodes complete.

        Args:
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env()
        policy = MagicMock()
        policy.generate_action.return_value = 0

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env),
        ):
            construct_video_evaluation(
                env_id='CartPole-v1',
                policy=policy,
                n_episodes=2,
                video_folder=video_folder,
                name_prefix='test',
            )

        mock_env.close.assert_called_once()

    @pytest.mark.parametrize(
        'env_id, name_prefix',
        [
            ('CartPole-v1', 'my-eval'),
            ('CartPole-v1', 'run-01'),
        ],
    )
    def test_construct_video_evaluation_passes_correct_args_to_record_video(
        self,
        env_id: str,
        name_prefix: str,
        tmp_path,
    ):
        """
        Test that construct_video_evaluation passes the correct arguments to RecordVideo.

        Args:
            env_id: The Gym environment ID.
            name_prefix: The prefix for video file names.
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env()
        policy = MagicMock()
        policy.generate_action.return_value = 0

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env) as mock_make,
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env) as mock_record_video,
        ):
            construct_video_evaluation(
                env_id=env_id,
                policy=policy,
                n_episodes=1,
                video_folder=video_folder,
                name_prefix=name_prefix,
            )

        mock_make.assert_any_call(env_id, render_mode='rgb_array')
        mock_record_video.assert_called_once()
        _, kwargs = mock_record_video.call_args
        assert kwargs['video_folder'] == video_folder
        assert kwargs['name_prefix'] == name_prefix
