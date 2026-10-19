import os

import numpy as np
import pytest

from tfrlrl.evaluation.plotting import plot_training_rewards


class TestPlotTrainingRewards:
    """Unit tests for the plot_training_rewards function."""

    @pytest.mark.parametrize('n_iterations', [1, 2, 10])
    def test_plot_training_rewards_creates_file(self, n_iterations: int, tmp_path):
        """
        Test that plot_training_rewards creates a file at the given output path.

        Args:
            n_iterations: The number of entries in the total_rewards array.
            tmp_path: Pytest temporary directory fixture.

        """
        output_path = str(tmp_path / 'training_rewards.png')
        total_rewards = np.arange(n_iterations, dtype=float)

        assert not os.path.isfile(output_path)

        plot_training_rewards(total_rewards, output_path)

        assert os.path.isfile(output_path)

    def test_plot_training_rewards_accepts_title(self, tmp_path):
        """
        Test that plot_training_rewards does not raise when a title is given.

        Args:
            tmp_path: Pytest temporary directory fixture.

        """
        output_path = str(tmp_path / 'training_rewards.png')
        total_rewards = np.arange(5, dtype=float)

        plot_training_rewards(total_rewards, output_path, title='Training Rewards')

        assert os.path.isfile(output_path)
