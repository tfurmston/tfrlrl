from typing import Dict

from torch import (
    Tensor,
    cat,
)


def flatten_tensor_dict(x: Dict[str, Tensor]) -> Tensor:
    """
    Flatten the given dictionary of PyTorch tensors into a signle PyTorch tensor.

    This function takes a dictionary of PyTorch tensors and flattens them into a single tensor.

    Args:
        x: The dictionary of tensors to be flattened.

    Returns:
        The flattened tensor.

    """
    return cat([j.flatten(start_dim=1) for j in x.values()], dim=1)
