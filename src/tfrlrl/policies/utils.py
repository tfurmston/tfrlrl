from typing import Dict

from torch import (
    Tensor,
    cat,
)


def flatten_tensor_dict(x: Dict[str, Tensor], dim: int = 0) -> Tensor:
    """
    Flatten the given dictionary of PyTorch tensors into a signle PyTorch tensor.

    This function takes a dictionary of PyTorch tensors and flattens them into a single tensor.

    Args:
        x: The dictionary of tensors to be flattened.
        dim: The starting dimension on which to flatten and concatenate the tensors.

    Returns:
        The flattened tensor. The flattened tensor will have the following shape:
                    (start_dims, n_elems / n_start_dim_elems).
        start_dims is the dimensions preceeding dim, which is empty when dim is zero, n_start_dim_elems
        is the number of dimensions in the starting dimensions (or 1 when dim is zero) and  n_elems is
        the number of elements across the different tensors in the input dictionary.

    Raises:
        RuntimeError: When the starting dimensions of the tensors in x, i.e. the dimensions less than
        dim, then a RuntimeError will be thrown by the call to cat.

    """
    return cat([j.flatten(start_dim=dim) for j in x.values()], dim=dim)
