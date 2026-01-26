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


class LinearBaseline:
    """
    A linear baseline class.

    This class uses a linear function for the baseline calculation. It uses the same features are given in
    ......
    """

    def __init__(self, reg_coeff=1e-5):
        """Class constructor."""
        self.reg_coeff = reg_coeff
        self._coeffs = None

    def calculate_features(self, observation_matrix: npt.ArrayLike, time_steps: npt.ArrayLike) -> npt.ArrayLike:
        """Calculate baseline features for the given observations."""
        o = np.clip(observation_matrix, -10, 10)
        al = time_steps.reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((time_steps.shape[0], 1))], axis=1)

    def calculate_baseline(
        self, observation_matrix: npt.ArrayLike, time_steps: npt.ArrayLike, feature_matrix: npt.ArrayLike = None
    ) -> npt.ArrayLike:
        """Calculate the baseline for the given examples."""
        if self._coeffs is None:
            return np.zeros(time_steps.shape[0])
        if feature_matrix is None:
            feature_matrix = self.calculate_features(observation_matrix, time_steps)
        return np.dot(feature_matrix, self._coeffs)

    def fit(self, feature_matrix: npt.ArrayLike, regressand: npt.ArrayLike) -> None:
        """Fit the baseline on the given examples."""
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                np.dot(feature_matrix.T, feature_matrix) + reg_coeff * np.identity(feature_matrix.shape[1]),
                np.dot(feature_matrix.T, regressand),
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10
