from typing import Dict, Union

import gym
from numpy.typing import NDArray


class Sampler:
    """
    This class provides functionality to sample from a given Gym environment.

    The class is single-threaded, i.e., it samples from a single instance of the environment using a single thread. The
    class provides iterable support, see https://docs.python.org/3/library/stdtypes.html#typeiter.
    """

    def __init__(self, env_id: str):
        """
        Initialise the instance of the Sampler. This entails initialising the environment and setting member variables.

        :param env_id: The Gym environment ID to be used in the sampling.
        """
        self.env = gym.make(env_id)
        self.observation = None
        self.next_observation = None
        self.action = None
        self.reward = None
        self.done = True
        self.info = None

    def __iter__(self):
        """Ensure that the Sampler class supports the iterable protocol."""
        return self

    def __next__(self) -> (NDArray, Union[int, float, NDArray], NDArray, float, bool, Dict):
        """Return the next item in the sampler iterator. If this is not possible, raise a StopIteration exception."""
        if self.done:
            self.observation = self.env.reset()
        else:
            self.observation = self.next_observation

        action = self.env.action_space.sample()
        self.next_observation, self.reward, self.done, self.info = self.env.step(action)
        return self.observation, self.action, self.next_observation, self.reward, self.done, self.info
