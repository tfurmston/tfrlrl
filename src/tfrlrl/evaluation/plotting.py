from typing import Optional

import numpy as np
from matplotlib.figure import Figure


def plot_training_rewards(total_rewards: np.ndarray, output_path: str, title: Optional[str] = None) -> None:
    """
    Plot the mean total episodic reward observed at each training iteration, and save it to file.

    Args:
        total_rewards: A 1-D array of length n_iterations, in which entry n is the mean total episodic
        reward observed during the n-th iteration of training.
        output_path: The file path to which the plot will be saved.
        title: An optional title for the plot.

    """
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.plot(np.arange(total_rewards.shape[0]), total_rewards)
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Mean total episodic reward')
    if title is not None:
        axes.set_title(title)
    figure.savefig(output_path)
