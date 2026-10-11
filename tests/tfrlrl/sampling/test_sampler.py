import os
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.policies.base import UniformActionSamplingPolicy
from tfrlrl.sampling.sampler import Sampler, VideoRecordingConfig


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


class TestSampler:
    """Class that encapsulates the unit tests for the Sampler class."""

    @pytest.mark.parametrize('env_id', ['CartPole-v1', 'CliffWalking-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=2000)
    def test_sample_n_steps_without_limit(self, env_id: str, n_steps: int):
        """
        Test that n-steps can be sampled from the environment and that the outputs follow the expected format.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        sampler = Sampler(env_id)
        for _ in range(n_steps):
            sample = sampler.__next__()
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)

    @pytest.mark.parametrize('env_id', ['CartPole-v1', 'CliffWalking-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=2000)
    def test_sample_n_steps_with_policy(self, env_id: str, n_steps: int):
        """
        Test that n-steps can be sampled from the environment with a custom policy.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment.
        """
        policy = UniformActionSamplingPolicy(env_id)
        sampler = Sampler(env_id, n_steps=n_steps, policy=policy)
        for sample in sampler:
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)
            # Verify action is valid for the environment
            assert isinstance(sample.action, (int, np.integer))
            if env_id == 'CartPole-v1':
                assert 0 <= sample.action < 2  # CartPole has 2 actions
            elif env_id == 'CliffWalking-v1':
                assert 0 <= sample.action < 4  # Cliff Walking has 4 actions

    @given(n_steps=st.integers(min_value=10, max_value=100))
    @settings(deadline=2000)
    def test_sample_with_env_kwargs(self, n_steps: int):
        """
        Test that environment kwargs are correctly passed through to the environment construction.

        Uses FrozenLake-v1 with is_slippery parameter to verify kwargs functionality.

        :param n_steps: The number of steps to sample from the environment.
        """
        env_id = 'FrozenLake-v1'
        sampler = Sampler(env_id, n_steps=n_steps, is_slippery=False)
        for sample in sampler:
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)
            # Verify action is valid for FrozenLake (4 actions: 0, 1, 2, 3)
            assert isinstance(sample.action, (int, np.integer))
            assert 0 <= sample.action < 4

    @pytest.mark.parametrize('env_id', ['CartPole-v1', 'CliffWalking-v1'])
    @given(n_steps=st.integers(min_value=10, max_value=50))
    @settings(deadline=4000)
    def test_reset_allows_multiple_iterations(self, env_id: str, n_steps: int):
        """
        Test that the reset method allows the sampler to be iterated over multiple times.

        This test verifies that after exhausting the iterator, calling reset() allows
        the sampler to be used again for a fresh iteration.

        :param env_id: The Gym environment ID to be used in the sampling.
        :param n_steps: The number of steps to sample from the environment per iteration.
        """
        sampler = Sampler(env_id, n_steps=n_steps)

        # First iteration: collect all samples
        first_iteration_samples = list(sampler)
        assert len(first_iteration_samples) == n_steps

        # Verify that the iterator is exhausted
        with pytest.raises(StopIteration):
            next(sampler)

        # Reset the sampler
        sampler.reset()

        # Second iteration: should be able to iterate again
        second_iteration_samples = list(sampler)
        assert len(second_iteration_samples) == n_steps

        # Verify that both iterations produced valid samples
        for sample in first_iteration_samples + second_iteration_samples:
            assert isinstance(sample.env_id, str)
            assert isinstance(sample.time_step, int)
            assert isinstance(sample.observation, np.ndarray)
            assert isinstance(sample.next_observation, np.ndarray)
            assert isinstance(sample.reward, float) or isinstance(sample.reward, int)
            assert isinstance(sample.done, bool)
            assert isinstance(sample.info, dict)


class TestSamplerVideoRecording:
    """Class that encapsulates the unit tests for video recording support in the Sampler class."""

    def test_video_config_creates_video_folder(self, tmp_path):
        """
        Test that a video_config creates the video folder if it does not exist.

        Args:
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env()
        video_config = VideoRecordingConfig(video_folder=video_folder, name_prefix='test')

        assert not os.path.isdir(video_folder)

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env),
        ):
            Sampler('CartPole-v1', video_config=video_config)

        assert os.path.isdir(video_folder)

    def test_video_config_forces_rgb_array_render_mode(self, tmp_path):
        """
        Test that a video_config forces render_mode='rgb_array' on environment construction.

        Args:
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env()
        video_config = VideoRecordingConfig(video_folder=video_folder, name_prefix='test')

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env) as mock_make,
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env),
        ):
            Sampler('CartPole-v1', video_config=video_config)

        mock_make.assert_any_call('CartPole-v1', render_mode='rgb_array')

    def test_video_config_wraps_env_with_record_video(self, tmp_path):
        """
        Test that a video_config wraps the environment with RecordVideo using the given parameters.

        Args:
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env()
        video_config = VideoRecordingConfig(video_folder=video_folder, name_prefix='test-prefix')

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env) as mock_record_video,
        ):
            Sampler('CartPole-v1', video_config=video_config)

        mock_record_video.assert_called_once()
        _, kwargs = mock_record_video.call_args
        assert kwargs['video_folder'] == video_folder
        assert kwargs['name_prefix'] == 'test-prefix'
        assert kwargs['episode_trigger'](0) is True
        assert kwargs['episode_trigger'](5) is True

    def test_no_video_config_does_not_wrap_env(self):
        """Test that the environment is not wrapped with RecordVideo when no video_config is given."""
        mock_env = _make_mock_env()

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env) as mock_make,
            patch('tfrlrl.sampling.sampler.RecordVideo') as mock_record_video,
        ):
            Sampler('CartPole-v1')

        assert all('render_mode' not in call.kwargs for call in mock_make.call_args_list)
        mock_record_video.assert_not_called()

    def test_video_config_respects_custom_episode_trigger(self, tmp_path):
        """
        Test that a custom episode_trigger on video_config is passed through to RecordVideo unchanged.

        Args:
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')
        mock_env = _make_mock_env()

        def episode_trigger(episode_id):
            return episode_id % 2 == 0

        video_config = VideoRecordingConfig(
            video_folder=video_folder,
            name_prefix='test',
            episode_trigger=episode_trigger,
        )

        with (
            patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env),
            patch('tfrlrl.sampling.sampler.RecordVideo', return_value=mock_env) as mock_record_video,
        ):
            Sampler('CartPole-v1', video_config=video_config)

        _, kwargs = mock_record_video.call_args
        assert kwargs['episode_trigger'] is episode_trigger

    def test_close_closes_env(self):
        """Test that close() closes the underlying environment."""
        mock_env = _make_mock_env()

        with patch('tfrlrl.sampling.sampler.gym.make', return_value=mock_env):
            sampler = Sampler('CartPole-v1')
            sampler.close()

        mock_env.close.assert_called_once()
