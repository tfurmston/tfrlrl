from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np


class Validator(ABC):
    """A base Descriptor class, https://docs.python.org/3/howto/descriptor.html, that is used for validation."""

    def __set_name__(self, owner, name):
        """Set the name of the value of the descriptor."""
        self._name = '_' + name

    def __get__(self, obj, type):
        """Get the value of the descriptor."""
        return getattr(obj, self._name)

    def __set__(self, obj, x):
        """Validate and set the value of the descriptor."""
        self.validate(x)
        setattr(obj, self._name, self._preprocess_value(x))

    def _preprocess_value(self, x):
        """Preprocess the value to be set as the value of the descriptor."""
        return x

    @abstractmethod
    def validate(self, value):
        """Validate the input value of the descriptor."""
        ...


class StringDescriptor(Validator):
    """A Descriptor validator class that validates the value is a string."""

    def validate(self, value: Any):
        """Validate that value is a string."""
        if not isinstance(value, str):
            raise TypeError(f'Expected {value!r} to be a string')


class IntDescriptor(Validator):
    """A Descriptor validator class that validates the value is an integer."""

    def validate(self, value: Any):
        """Validate that value is an integer."""
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f'Expected {value!r} to be an integer')


class FloatDescriptor(Validator):
    """A Descriptor validator class that validates the value is a float."""

    def validate(self, value: Any):
        """Validate that value is a float."""
        if not isinstance(value, float):
            raise TypeError(f'Expected {value!r} to be a float')


class IntFloatDescriptor(Validator):
    """A Descriptor validator class that validates the value is an integer or a float."""

    def validate(self, value: Any):
        """Validate that value is an integer."""
        if not isinstance(value, (int, np.integer, float)):
            raise TypeError(f'Expected {value!r} to be an integer or a float')


class BooleanDescriptor(Validator):
    """A Descriptor validator class that validates the value is a Boolean."""

    def validate(self, value: Any):
        """Validate that value is a Boolean."""
        if not isinstance(value, bool):
            raise TypeError(f'Expected {value!r} to be a Boolean')


class DictDescriptor(Validator):
    """A Descriptor validator class that validates the value is a dictionary."""

    def validate(self, value: Any):
        """Validate that value is a dictionary."""
        if not isinstance(value, dict):
            raise TypeError(f'Expected {value!r} to be a dictionary')


class NumpyArrayDescriptor(Validator):
    """
    A Descriptor validator class for Numpy NdArrays.

    Attributes:
        allowed_instances: A tuple of allow instance types. If the validator allows for the conversion of
        floats/ints to NumPy arrays, then this will include int and float types. Otherwise, it will be a
        singleton tuple of a NumPy array.

    """

    def __init__(self, allow_numeric: bool = True):
        """
        Initialise NumPy descriptor.

        Args:
            allow_numeric: A Boolean indicating whether numeric types are allowed. If so, then numerics, i.e
            floats and ints will be converted to a NumPy array of the appropriate type.

        """
        if allow_numeric:
            self.allowed_instances: tuple[type, ...] = (np.ndarray, int, float)
        else:
            self.allowed_instances = (np.ndarray,)

    def validate(self, value: Any):
        """Validate that value is an NumPy ndarray."""
        if not isinstance(value, self.allowed_instances):
            raise TypeError(f'Expected {value!r} to be on instance of {self.allowed_instances}')

    def _preprocess_value(self, x: Union[np.ndarray, int, float]):
        """Convert to NumPy array in case of floats or ints."""
        return (
            x if isinstance(x, np.ndarray) else x * np.ones(1, dtype=np.float64 if isinstance(x, float) else np.int64)
        )


class NumpyArrayExpandedDescriptor(NumpyArrayDescriptor):
    """A Descriptor validator class for Numpy NdArrays that require an expanded final dimension."""

    def __init__(self, allow_numeric: bool = True):
        """
        Initialise NumPy Array Expanded descriptor.

        Args:
            allow_numeric: A Boolean indicating whether numeric types are allowed. If so, then numerics, i.e
            floats and ints will be converted to a NumPy array of the appropriate type.

        """
        super().__init__(allow_numeric=allow_numeric)

    def _preprocess_value(self, x: Union[np.ndarray, int, float]):
        """Add a new axis to the end of the raw observation vector."""
        return super()._preprocess_value(x)[:, np.newaxis]
