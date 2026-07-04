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
        The flattened tensor. The flattened tensor will have a shape of (1, n_elems) in which n_elems
        is the number of elements across the different tensors in the input dictionary.

    """
    return cat([j.flatten() for j in x.values()])[None, :]
