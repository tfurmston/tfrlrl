from dataclasses import InitVar, make_dataclass
from typing import Callable, List, Tuple, Type

import gymnasium as gym
import numpy as np

from tfrlrl.data_models.base import (BooleanDescriptor, DictDescriptor,
                                     FloatDescriptor, IntDescriptor,
                                     ObservationVectorDescriptor,
                                     StringDescriptor, Validator)


class StepDataclassException(Exception):
    """Custom exception to encompass errors raised by the Step dataclass."""

    pass


def construct_action_space_definition(env_id: str) -> Tuple[str, Type[Validator], Validator]:
    """
    Construct the (dataclass) definition of the action space for the given environment.

    :param env_id: The I.D. of the environment for which the action space definition is to be constructed.
    :return: The label of the data class field for the action, along with the descriptor for the action space.
    """
    env = gym.make(env_id)
    if isinstance(env.action_space, gym.spaces.Discrete):
        return 'action', IntDescriptor, IntDescriptor()
    elif isinstance(env.action_space, gym.spaces.Box):
        return 'action', ObservationVectorDescriptor, ObservationVectorDescriptor()
    raise StepDataclassException('Action space must be Discrete or Box.')


def construct_steps_postinitialisation_fn(env_id: str) -> Callable:
    """
    Construct the post-initialisation function for the Steps dataclass.

    :param env_id: The environment I.D. for which the post-initialisation function is to be constructed.
    :return: The post-initialisation function.
    """
    env = gym.make(env_id)
    if isinstance(env.action_space, gym.spaces.Discrete):
        def post_initialise_steps(self, sample_steps):
            """Post-initialisation of steps from a collection of individual steps."""
            self.n_steps = len(sample_steps)
            self.env_ids = [x.env_id for x in sample_steps]
            self.time_steps = np.array([x.time_step for x in sample_steps])
            self.observations = np.concatenate([x.observation for x in sample_steps], axis=-1)
            self.actions = np.array([x.action for x in sample_steps])
            self.next_observations = np.concatenate([x.next_observation for x in sample_steps], axis=-1)
            self.rewards = np.array([x.reward for x in sample_steps])
            self.dones = np.array([x.done for x in sample_steps])
        return post_initialise_steps
    elif isinstance(env.action_space, gym.spaces.Box):
        def post_initialise_steps(self, sample_steps):
            """Post-initialisation of steps from a collection of individual steps."""
            self.n_steps = len(sample_steps)
            self.env_ids = [x.env_id for x in sample_steps]
            self.time_steps = np.array([x.time_step for x in sample_steps])
            self.observations = np.concatenate([x.observation for x in sample_steps], axis=-1)
            self.actions = np.concatenate([x.action for x in sample_steps], axis=-1)
            self.next_observations = np.concatenate([x.next_observation for x in sample_steps], axis=-1)
            self.rewards = np.array([x.reward for x in sample_steps])
            self.dones = np.array([x.done for x in sample_steps])
        return post_initialise_steps
    raise StepDataclassException('Action space must be Discrete or Box.')


# TODO: Improve naming of base classes
def construct_step_dataclass(env_id: str):
    """
    Construct a dataclass to represent a step in the given environment.

    :param env_id: The environment I.D. for which to make the dataclass.
    :return: The dataclass representing a step in the given environment.
    """
    return make_dataclass(
        'Step',
        [
            ('env_id', StringDescriptor, StringDescriptor()),
            ('time_step', IntDescriptor, IntDescriptor()),
            ('observation', ObservationVectorDescriptor, ObservationVectorDescriptor()),
            construct_action_space_definition(env_id),
            ('next_observation', ObservationVectorDescriptor, ObservationVectorDescriptor()),
            ('reward', FloatDescriptor, FloatDescriptor()),
            ('info', DictDescriptor, DictDescriptor()),
            ('done', BooleanDescriptor, BooleanDescriptor()),
        ],
    )


def construct_steps_dataclass(env_id: str):
    """
    Construct a dataclass to represent a steps in the given environment.

    :param env_id: The environment I.D. for which to make the dataclass.
    :return: The dataclass representing steps in the given environment.
    """
    return make_dataclass(
        'Steps',
        [
            ('n_steps', int, None),
            ('env_ids', List[str], None),
            ('time_steps', np.ndarray, None),
            ('observations', np.ndarray, None),
            ('actions', np.ndarray, None),
            ('next_observations', np.ndarray, None),
            ('rewards', np.ndarray, None),
            ('dones', np.ndarray, None),
            ('sample_steps', InitVar, None),
        ],
        namespace={'__post_init__': construct_steps_postinitialisation_fn(env_id)}
    )


def construct_step_dataclasses(env_id: str):
    """
    Construct a dataclass to represent a step and a range of steps in the given environment.

    :param env_id: The environment I.D. for which to make the dataclass.
    :return: The dataclass representing a step in the given environment.
    :return: The dataclass representing a range steps in the given environment.
    """
    step_sls = construct_step_dataclass(env_id)
    steps_sls = construct_steps_dataclass(env_id)

    return step_sls, steps_sls
