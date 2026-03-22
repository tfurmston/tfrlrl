from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class Baseline:
    """A base class for baseline algorithms, which are used to reduce the variance of policy graident methods."""

    @abstractmethod
    def calculate_baseline(self, observation_matrix: npt.ArrayLike, time_steps: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculate the baseline for the given examples.

        Args:
            observation_matrix: An [n_obs, n_steps] NumPy matrix of observations.
            time_steps: An [n_steps] NumPy matrix of the time steps corresponding to the given observations.

        Returns:
            The baselines for the given observations.

        """
        ...

    @abstractmethod
    def fit(self, feature_matrix: npt.ArrayLike, regressand: npt.ArrayLike) -> None:
        """
        Fit the baseline on the given examples.

        Args:
            feature_matrix: A pre-calculated [2 * n_obs + 4, n_steps] feature matrix. If provided the features
            will not be calculated again.
            regressand: A pre-calculated [n_steps] vector of the regressand.

        """
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Get the state dictionary of the baseline clase.

        Return the state dictionary of the baseline. This follows the same terminology as PyTorch, though baselines
        as not implemented through PyTorch. This is just a convention. The state dictionary returned by this function
        should contain the information necessary to reinstantiate the baseline class.

        Returns:
            The current state dictionary of the baseline class.

        """
        ...

    @abstractmethod
    def set_state(self, state_dict: dict[str, Any]) -> None:
        """
        Set the state dictionary of the baseline clase.

        Set the state dictionary of the baseline. This follows the same terminology as PyTorch, though baselines
        as not implemented through PyTorch. This is just a convention. The state dictionary returned by this function
        should contain the information necessary to reinstantiate the baseline class.

        Args:
            state_dict: The state dictionary to be assigned to the baseline.

        """
        ...

    @abstractmethod
    def update(self, state_dict, **kwargs) -> None:
        """
        Update the baseline.

        Args:
            state_dict: The state dictionary to be assigned to the baseline.
            kwargs: Optional keyword-arguments for the policy update.

        """
        ...


class LinearBaseline(Baseline):
    """
    A linear baseline class.

    This class uses a linear function for the baseline calculation. It uses the same features are given in
    ......
    """

    def __init__(self, reg_coeff=1e-5, coeffs: npt.ArrayLike = None):
        """Class constructor."""
        self._reg_coeff = reg_coeff
        self._coeffs = coeffs

    def get_state(self) -> dict[str, Any]:
        """
        Get the state dictionary of the baseline clase.

        Return the state dictionary of the baseline. This follows the same terminology as PyTorch, though baselines
        as not implemented through PyTorch. This is just a convention. The state dictionary returned by this function
        should contain the information necessary to reinstantiate the baseline class.

        Returns:
            The current state dictionary of the baseline class.

        """
        return {
            'reg_coeff': self._reg_coeff,
            'coeffs': self._coeffs,
        }

    @abstractmethod
    def set_state(self, state_dict: dict[str, Any]) -> None:
        """
        Set the state dictionary of the baseline clase.

        Set the state dictionary of the baseline. This follows the same terminology as PyTorch, though baselines
        as not implemented through PyTorch. This is just a convention. The state dictionary returned by this function
        should contain the information necessary to reinstantiate the baseline class.

        Args:
            state_dict: The state dictionary to be assigned to the baseline.

        """
        self._reg_coeff = state_dict['reg_coeff']
        self._coeffs = state_dict['coeffs']

    @abstractmethod
    def update(self, state_dict, **kwargs) -> None:
        """
        Update the baseline.

        Args:
            state_dict: The state dictionary to be assigned to the baseline.
            kwargs: Optional keyword-arguments for the policy update.

        """
        self.set_state(state_dict)

    def calculate_features(self, observation_matrix: npt.ArrayLike, time_steps: npt.ArrayLike) -> npt.ArrayLike:
        """
        Calculate baseline features for the given observations.

        Args:
            observation_matrix: An [n_obs, n_steps] NumPy matrix of observations.
            time_steps: An [n_steps] NumPy matrix of the time steps corresponding to the given observations.

        Returns:
            A NumPy array containing the baseline features for the given observations.

        """
        o = np.clip(observation_matrix, -10, 10)
        al = time_steps.reshape(1, -1) / 100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((1, time_steps.shape[0]))], axis=0)

    def calculate_baseline(
        self, observation_matrix: npt.ArrayLike, time_steps: npt.ArrayLike, feature_matrix: npt.ArrayLike = None
    ) -> npt.ArrayLike:
        """
        Calculate the baseline for the given examples.

        Args:
            observation_matrix: An [n_obs, n_steps] NumPy matrix of observations.
            time_steps: An [n_steps] NumPy matrix of the time steps corresponding to the given observations.
            feature_matrix: A pre-calculated [2 * n_obs + 4, n_steps] feature matrix. If provided the features
            will not be calculated again.

        Returns:
            The baselines for the given observations.

        """
        if self._coeffs is None:
            return np.zeros(time_steps.shape[0])
        if feature_matrix is None:
            feature_matrix = self.calculate_features(observation_matrix, time_steps)
        return np.dot(self._coeffs, feature_matrix)

    def fit(self, feature_matrix: npt.ArrayLike, regressand: npt.ArrayLike) -> None:
        """
        Fit the baseline on the given examples.

        Args:
            feature_matrix: A pre-calculated [2 * n_obs + 4, n_steps] feature matrix. If provided the features
            will not be calculated again.
            regressand: A pre-calculated [n_steps] vector of the regressand.

        """
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                np.dot(feature_matrix, feature_matrix.T) + reg_coeff * np.identity(feature_matrix.shape[0]),
                np.dot(feature_matrix, regressand.T),
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10
