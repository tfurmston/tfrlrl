import numpy as np
from numpy.typing import NDArray

from tfrlrl.features.base import (
    EnvironmentException,
    FeatureFunction,
)


class OneHotFeatureFunction(FeatureFunction):
    """
    A one-hot feature function.

    A one-hot feature function that can be used in toy discrete domains to represent a full
    parameterised policy.
    """

    def __init__(self, S: int, A: int):
        """
        Initialise one-hot feature function for discrete state-action spaces.

        A feature function that maps state observations to one-hot encoded features, excluding one
        action per state to avoid linear dependency (common in softmax parameterization).

        Args:
            S: The total number of states in the state space.
            A: The number of discrete actions in the action space.

        """
        self.S = S
        self.A = A

        self.f = np.zeros([S * (A - 1), S * A])
        inds = np.delete(np.arange(S * A), np.arange(S * A, step=A))
        self.f[:, inds] = np.eye(S * (A - 1))

    @property
    def n_features(self) -> int:
        """The number of features in the feature function."""
        return self.S * (self.A - 1)

    def __call__(self, observations: NDArray) -> NDArray:
        """
        Construct a one-hot features for the given observations.

        This function creates one-hot features for the given observations. If a single observation is
        given, in the form of either a NumPy integer or a NumPy array of size 1, then the functon will
        return a NumPy array of size, [n_features, n_actions, 1], in which n_features is the number of
        features and n_actions is the number of actions. If multiple observations are given, then it
        must be in the form of a two-dimensional NumPy array in which the first diemsnion is of size
        one. This dimension represents the index of the state. In this case, the return type is a
        three-dimension NumPy array of size, [n_features, n_actions, n_observations], in which
        n_observations is the number of input observatiuons.

        Args:
            observations: The observations over which to calculate the features.

        Returns:
            The feature vectors for the given observations.

        """
        if observations.size == 1:
            return self.f[:, observations.flat[0] * self.A : (observations.flat[0] + 1) * self.A][:, :, np.newaxis]

        if observations.shape[0] > 1 or len(observations.shape) > 2:
            raise EnvironmentException(
                f'Unsupported observation dimensions for one-hot feature function: {observations.shape}'
            )
        return np.concatenate(
            [
                self.f[:, observations[0, i] * self.A : (observations[0, i] + 1) * self.A][:, :, np.newaxis]
                for i in range(observations.shape[1])
            ],
            axis=2,
        )
