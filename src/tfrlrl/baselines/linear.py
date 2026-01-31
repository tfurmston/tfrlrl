from abc import abstractmethod

import numpy as np
import numpy.typing as npt


class Baseline:
    """A base class for baseline algorithms, which are used to reduce the variance of policy graident methods."""

    @abstractmethod
    def calculate_baseline(self):
        """Calculate the baseline for the given examples."""
        ...

    @abstractmethod
    def fit(self):
        """Fit the baseline on the given examples."""
        ...


class LinearBaseline(Baseline):
    """
    A linear baseline class.

    This class uses a linear function for the baseline calculation. It uses the same features are given in
    ......
    """

    def __init__(self, reg_coeff=1e-5, coeffs: npt.ArrayLike = None):
        """Class constructor."""
        self.reg_coeff = reg_coeff
        self._coeffs = coeffs

    def calculate_features(self, observation_matrix: npt.ArrayLike, time_steps: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculate baseline features for the given observations.

        param: observation_matrix: An [n_obs, n_steps] NumPy matrix of observations.
        param: time_steps: An [n_steps] NumPy matrix of the time steps corresponding to the given observations.
        """
        o = np.clip(observation_matrix, -10, 10)
        al = time_steps.reshape(1, -1) / 100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((1, time_steps.shape[0]))], axis=0)

    def calculate_baseline(
        self, observation_matrix: npt.ArrayLike, time_steps: npt.ArrayLike, feature_matrix: npt.ArrayLike = None
    ) -> npt.ArrayLike:
        """
        Calculate the baseline for the given examples.

        param: observation_matrix: An [n_obs, n_steps] NumPy matrix of observations.
        param: time_steps: An [n_steps] NumPy matrix of the time steps corresponding to the given observations.
        param: feature_matrix: A pre-calculated [2 * n_obs + 4, n_steps] feature matrix. If provided the features
        will not be calculated again.
        """
        if self._coeffs is None:
            return np.zeros(time_steps.shape[0])
        if feature_matrix is None:
            feature_matrix = self.calculate_features(observation_matrix, time_steps)
        return np.dot(self._coeffs, feature_matrix)

    def fit(self, feature_matrix: npt.ArrayLike, regressand: npt.ArrayLike) -> None:
        """
        Fit the baseline on the given examples.

        param: feature_matrix: A pre-calculated [2 * n_obs + 4, n_steps] feature matrix. If provided the features
        will not be calculated again.
        param: regressand: A pre-calculated [n_steps] vector of the regressand.
        """
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                np.dot(feature_matrix, feature_matrix.T) + reg_coeff * np.identity(feature_matrix.shape[1]),
                np.dot(feature_matrix, regressand.T),
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10
