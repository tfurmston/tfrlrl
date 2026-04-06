from dataclasses import dataclass

from tfrlrl.data_models.base import NumpyArrayDescriptor


@dataclass
class ReplayBufferSample:
    """A Python dataclass to encapsulate a collection of samples from a replay buffer."""

    observations: NumpyArrayDescriptor = NumpyArrayDescriptor()
    actions: NumpyArrayDescriptor = NumpyArrayDescriptor()
    next_observations: NumpyArrayDescriptor = NumpyArrayDescriptor()
    rewards: NumpyArrayDescriptor = NumpyArrayDescriptor()
    dones: NumpyArrayDescriptor = NumpyArrayDescriptor()
