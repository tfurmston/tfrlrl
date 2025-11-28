from typing import Callable

import numpy as np
from numpy.typing import NDArray


class EnvironmentException(Exception):
    """Exception class to capture environment issues, such as incompatible state/action dimensions."""

    pass


def construct_one_hot_feature_function(S: int, A: int) -> Callable[[NDArray], NDArray]:
    """
    Construct a one-hot feature function for discrete state-action spaces.

    This function creates a feature function that maps state observations to one-hot encoded features,
    excluding one action per state to avoid linear dependency (common in softmax parameterization).

    :param S: The total number of states in the state space.
    :param A: The number of discrete actions in the action space.
    :return: A feature function that takes an observation and returns the corresponding feature matrix slice.
    """
    f = np.zeros([S * A, S * (A - 1)])
    inds = np.delete(np.arange(S * A), np.arange(S * A, step=A))
    f[inds, :] = np.eye(S * (A - 1))

    def feature_fn(observation: NDArray) -> NDArray:
        if observation.size != 1:
            raise EnvironmentException('Unsupported observation dimensions for one-hot feature function.')
        return f[observation.flat[0] : (observation.flat[0] + A), :]

    return feature_fn
