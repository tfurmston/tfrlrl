from dataclasses import InitVar, dataclass
from typing import List

import numpy as np


@dataclass
class StepSample:
    """"""
    env_id: str
    time_step: int
    observation: np.ndarray
    action: int
    next_observation: np.ndarray
    reward: float
    done: bool
    info: dict


# add types
# Add checks if data not proviced as expected
@dataclass
class StepSamples:
    """"""
    n_samples: int = None
    env_ids: List[str] = None
    time_steps: np.ndarray = None
    observations: np.ndarray = None
    actions: np.ndarray = None
    next_observations: np.ndarray = None
    rewards: np.ndarray = None
    dones: np.ndarray = None
    sample_steps: InitVar[List[StepSample] | None] = None

    def __post_init__(self, sample_steps):
        self.n_samples = len(sample_steps)
        self.env_ids = [x.env_id for x in sample_steps]
        self.time_steps = np.array([x.time_step for x in sample_steps])
        self.observations = np.stack([x.observation for x in sample_steps], axis=1)
        self.actions = np.array([x.action for x in sample_steps])
        self.next_observations = np.stack([x.next_observation for x in sample_steps], axis=1)
        self.rewards = np.array([x.reward for x in sample_steps])
        self.dones = np.array([x.done for x in sample_steps])