from abc import abstractmethod
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class BaselineException(Exception):
    """Custom exception to encompass errors raised by the baselines."""

    pass


class Baseline:
    """A base class for baseline algorithms, which are used to reduce the variance of policy graident methods."""

    def __init__(self, env_id: str):
        """
        Initialise the base baseline class.

        Args:
            env_id: The I.D. of the environment from which to collect statistics.

        """
        env = gym.make(env_id)
        if not self.valid_environment(env):
            raise BaselineException('Can not use baseline class on environment: (%s, %s)', type(self).__name__, env_id)

    @abstractmethod
    def valid_environment(self, env: gym.Env) -> bool:
        """Determine whether the baseline class can be used in the given environment."""
        ...

    @abstractmethod
    def calculate_baseline(self, observation_matrix: np.ndarray, time_steps: np.ndarray) -> np.ndarray:
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
    def fit(self, feature_matrix: np.ndarray, regressand: np.ndarray) -> None:
        """
        Fit the baseline on the given examples.

        Args:
            feature_matrix: A pre-calculated [n_steps,  n_f] feature matrix, in which n_f is the number of features in
            the baseline.
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
    def update(self, state_dict) -> None:
        """
        Update the baseline.

        Args:
            state_dict: The state dictionary to be assigned to the baseline.

        """
        ...


class LinearBaseline(Baseline):
    """
    A linear baseline class.

    This class uses a linear function for the baseline calculation. It uses the same features are given in the paper,
    Benchmarking Deep Reinforcement Learning for Continuous Control by Yan Duan et. al. (See section 2 of the appendix
     of the paper.) In particular, the features of the linear predictor are of the form:

        baseline(o) = (o, o^^2, 0.01 * t, (0.01 * t)^2, (0.01 * t)^3, 1)

    in which o is the observation, t is the time step and o^^2 should be read as the element-wise product.

    """

    def __init__(self, env_id: str, reg_coeff=1e-5, coeffs: Optional[np.ndarray] = None):
        """Class constructor."""
        super().__init__(env_id=env_id)
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
    def valid_environment(self, env: gym.Env) -> bool:
        """Determine whether the baseline class can be used in the given environment."""
        return isinstance(env.observation_space, gym.spaces.Box)

    @abstractmethod
    def update(self, state_dict) -> None:
        """
        Update the baseline.

        Args:
            state_dict: The state dictionary to be assigned to the baseline.

        """
        self.set_state(state_dict)

    def calculate_features(self, observation_matrix: np.ndarray, time_steps: np.ndarray) -> np.ndarray:
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
        self, observation_matrix: np.ndarray, time_steps: np.ndarray, feature_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
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

    def fit(self, feature_matrix: np.ndarray, regressand: np.ndarray) -> None:
        """
        Fit the baseline on the given examples.

        Args:
            feature_matrix: A pre-calculated [2 * n_obs + 4, n_steps] feature matrix.
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
