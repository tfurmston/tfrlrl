from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from typing import List

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

    def validate(self, value):
        """Validate that value is a string."""
        if not isinstance(value, str):
            raise TypeError(f'Expected {value!r} to be a string')


class IntDescriptor(Validator):
    """A Descriptor validator class that validates the value is an integer."""

    def validate(self, value):
        """Validate that value is an integer."""
        if not isinstance(value, int):
            raise TypeError(f'Expected {value!r} to be a integer')


class FloatDescriptor(Validator):
    """A Descriptor validator class that validates the value is a float."""

    def validate(self, value):
        """Validate that value is a float."""
        if not isinstance(value, float):
            raise TypeError(f'Expected {value!r} to be a float')


class BooleanDescriptor(Validator):
    """A Descriptor validator class that validates the value is a Boolean."""

    def validate(self, value):
        """Validate that value is a Boolean."""
        if not isinstance(value, bool):
            raise TypeError(f'Expected {value!r} to be a Boolean')


class DictDescriptor(Validator):
    """A Descriptor validator class that validates the value is a dictionary."""

    def validate(self, value):
        """Validate that value is a dictionary."""
        if not isinstance(value, dict):
            raise TypeError(f'Expected {value!r} to be a dictionary')


class ObservationVectorDescriptor(Validator):
    """A Descriptor validator class for observations, which is used to ensure dimensions of observations are correct."""

    def _preprocess_value(self, x):
        """Add a new axis to the end of the observation vector."""
        return x[:, np.newaxis]

    def validate(self, value):
        """Validate that value is an NumPy ndarray."""
        if not isinstance(value, np.ndarray):
            raise TypeError(f'Expected {value!r} to be an NumPy ndarray')


@dataclass
class Step:
    """A Python dataclass to encapsulate a step in a Markov decision process."""

    env_id: StringDescriptor = StringDescriptor()
    time_step: IntDescriptor = IntDescriptor()
    observation: ObservationVectorDescriptor = ObservationVectorDescriptor()
    action: IntDescriptor = IntDescriptor()
    next_observation: ObservationVectorDescriptor = ObservationVectorDescriptor()
    reward: FloatDescriptor = FloatDescriptor()
    info: DictDescriptor = DictDescriptor()
    done: BooleanDescriptor = BooleanDescriptor()


@dataclass
class Steps:
    """A Python dataclass to encapsulate a collection of steps in a Markov decision process."""

    n_steps: int = None
    env_ids: List[str] = None
    time_steps: np.ndarray = None
    observations: np.ndarray = None
    actions: np.ndarray = None
    next_observations: np.ndarray = None
    rewards: np.ndarray = None
    dones: np.ndarray = None
    sample_steps: InitVar[List[Step] | None] = None

    def __post_init__(self, sample_steps: List[Step]):
        """Post-initialisation of steps from a collection of individual steps."""
        self.n_steps = len(sample_steps)
        self.env_ids = [x.env_id for x in sample_steps]
        self.time_steps = np.array([x.time_step for x in sample_steps])
        self.observations = np.concatenate([x.observation for x in sample_steps], axis=-1)
        self.actions = np.array([x.action for x in sample_steps])
        self.next_observations = np.concatenate([x.next_observation for x in sample_steps], axis=-1)
        self.rewards = np.array([x.reward for x in sample_steps])
        self.dones = np.array([x.done for x in sample_steps])
