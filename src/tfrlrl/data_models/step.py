from dataclasses import InitVar, dataclass
from typing import List

import numpy as np

from tfrlrl.data_models.base import (BooleanDescriptor, DictDescriptor,
                                     FloatDescriptor, IntDescriptor,
                                     ObservationVectorDescriptor,
                                     StringDescriptor)


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
