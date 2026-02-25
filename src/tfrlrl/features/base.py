from abc import (
    ABC,
    abstractmethod,
)

from numpy.typing import NDArray


class EnvironmentException(Exception):
    """Environment error, such as incompatible state/action dimensions."""

    pass


class FeatureFunction(ABC):
    """
    Abstract base class that represents a feature function.

    Feature function classes are to be used when a hand-coded feature function is to be used, as opposed
    to features learnt through deep learning. For example, in the case of one-hot encodings in toy discrete
    domains.
    """

    @property
    @abstractmethod
    def n_features(self) -> int:
        """The number of features in the feature function."""
        ...

    @abstractmethod
    def __call__(self, observations: NDArray) -> NDArray:
        """
        Construct features for the given observations.

        This function is used to construct the features for the given observations.

        Args:
            observations: The observations for which the features are to be constructed.

        Returns:
            features: The constructed features.

        """
        ...
