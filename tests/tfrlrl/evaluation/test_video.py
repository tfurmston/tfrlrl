import logging
import os
import re

import pytest

from tfrlrl.evaluation.video import construct_video_evaluation
from tfrlrl.policies.dense_neural_network import DenseNetworkPolicy, DenseNetworkPolicyConfig

ENV_ID = 'InvertedPendulum-v5'


def _make_policy():
    """Construct a real policy for the (Box action space) environment used in these tests."""
    return DenseNetworkPolicy(env_id=ENV_ID, config=DenseNetworkPolicyConfig(hidden_space_dims=[8]))


class TestConstructVideoEvaluation:
    """Unit tests for the construct_video_evaluation function."""

    @pytest.mark.parametrize('n_episodes', [1, 2, 3])
    def test_construct_video_evaluation_creates_video_folder(self, n_episodes: int, tmp_path):
        """
        Test that construct_video_evaluation creates the video folder if it does not exist.

        Args:
            n_episodes: The number of evaluation episodes.
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')

        assert not os.path.isdir(video_folder)

        construct_video_evaluation(
            env_id=ENV_ID,
            policy=_make_policy(),
            n_episodes=n_episodes,
            video_folder=video_folder,
            name_prefix='test',
        )

        assert os.path.isdir(video_folder)

    @pytest.mark.parametrize('n_episodes', [1, 2, 3])
    def test_construct_video_evaluation_records_one_video_per_episode(self, n_episodes: int, tmp_path):
        """
        Test that construct_video_evaluation records exactly one video file per episode.

        Args:
            n_episodes: The number of evaluation episodes.
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')

        construct_video_evaluation(
            env_id=ENV_ID,
            policy=_make_policy(),
            n_episodes=n_episodes,
            video_folder=video_folder,
            name_prefix='test',
        )

        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        assert len(video_files) == n_episodes

    @pytest.mark.parametrize('name_prefix', ['my-eval', 'run-01'])
    def test_construct_video_evaluation_uses_name_prefix(self, name_prefix: str, tmp_path):
        """
        Test that construct_video_evaluation names video files using the given prefix.

        Args:
            name_prefix: The prefix for video file names.
            tmp_path: Pytest temporary directory fixture.

        """
        video_folder = str(tmp_path / 'videos')

        construct_video_evaluation(
            env_id=ENV_ID,
            policy=_make_policy(),
            n_episodes=1,
            video_folder=video_folder,
            name_prefix=name_prefix,
        )

        video_files = os.listdir(video_folder)
        assert video_files
        assert all(f.startswith(name_prefix) for f in video_files)

    @pytest.mark.parametrize('n_episodes', [1, 2, 3])
    def test_construct_video_evaluation_logs_one_record_per_episode_plus_summary(
        self,
        n_episodes: int,
        tmp_path,
        caplog,
    ):
        """
        Test that construct_video_evaluation emits one INFO log record per episode, plus a summary record.

        Args:
            n_episodes: The number of evaluation episodes.
            tmp_path: Pytest temporary directory fixture.
            caplog: Pytest log capture fixture.

        """
        with caplog.at_level(logging.INFO, logger='tfrlrl.evaluation.video'):
            construct_video_evaluation(
                env_id=ENV_ID,
                policy=_make_policy(),
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

    def test_construct_video_evaluation_logs_accumulated_episode_reward(self, tmp_path, caplog):
        """
        Test that construct_video_evaluation logs the correctly accumulated episode reward.

        InvertedPendulum-v5 gives a reward of 1.0 for every step the episode survives, and a reward of 0.0 on the
        terminal step where the pole falls. With an untrained policy, episodes reliably end this way well before
        the environment's default step limit, so the logged total reward should equal (step count - 1).

        Args:
            tmp_path: Pytest temporary directory fixture.
            caplog: Pytest log capture fixture.

        """
        with caplog.at_level(logging.INFO, logger='tfrlrl.evaluation.video'):
            construct_video_evaluation(
                env_id=ENV_ID,
                policy=_make_policy(),
                n_episodes=3,
                video_folder=str(tmp_path / 'videos'),
                name_prefix='test',
            )

        for record in caplog.records[:-1]:
            match = re.match(r'Episode \d+: (\d+) steps, reward = ([\d.]+)', record.message)
            assert match is not None
            n_steps, reward = int(match.group(1)), float(match.group(2))
            assert reward == float(n_steps - 1)
