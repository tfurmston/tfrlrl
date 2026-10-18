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


def calculate_jacobian_batch_ndim(jacobian: Dict[str, Tensor], parameters: Dict[str, Tensor]) -> int:
    """
    Calculate the number of leading batch dimensions shared by the tensors in a Jacobian dictionary.

    When the Jacobian of a policy's log-probabilities is calculated (e.g. via torch.func.jacfwd), each
    tensor in the resulting dictionary has shape (*batch_shape, *parameter_shape): the shape of the
    log-probabilities that were differentiated (batch_shape) is prepended to the shape of the parameter the
    Jacobian was taken with respect to (parameter_shape). The number of dimensions in batch_shape varies by
    policy implementation, e.g. whether the state-action pairs are represented with a single leading
    dimension or an additional dummy dimension, so it cannot be inferred from the shape of the actions used
    to calculate the Jacobian. This function instead calculates it directly, by comparing the number of
    dimensions of a Jacobian tensor to the number of dimensions of the (undifferentiated) parameter it
    corresponds to. This is the number of dimensions on which flatten_tensor_dict and unflatten_tensor_dict
    should operate in order to flatten (only) the parameter-shaped part of each tensor in jacobian.

    Args:
        jacobian: A dictionary mapping parameter names to their corresponding Jacobian tensors, as returned
        by BasePyTorchPolicy.calculate_jacobian.
        parameters: A dictionary mapping parameter names to the corresponding (undifferentiated) network
        parameters, e.g. dict(policy.network.named_parameters()).

    Returns:
        The number of leading batch dimensions shared by all of the tensors in jacobian.

    """
    sample_parameter_name = next(iter(jacobian))
    return jacobian[sample_parameter_name].ndim - parameters[sample_parameter_name].ndim


def unflatten_tensor_dict(x: Tensor, reference: Dict[str, Tensor], dim: int = 0) -> Dict[str, Tensor]:
    """
    Unflatten the given PyTorch tensor into a dictionary of PyTorch tensors.

    This function is the inverse of flatten_tensor_dict. It takes a tensor produced by flattening
    the tensors in reference along dim, e.g. via flatten_tensor_dict(reference, dim=dim), and splits
    it back into a dictionary of tensors with the same keys, order and shapes as reference.

    Args:
        x: The tensor to be unflattened.
        reference: A dictionary of tensors whose keys, order and shapes (from dim onwards) are used
        to split and reshape x back into a dictionary of tensors.
        dim: The starting dimension on which x was flattened and concatenated.

    Returns:
        A dictionary of tensors with the same keys, order and shapes as reference.

    Raises:
        RuntimeError: When the total size of x along dim does not match the sum of the flattened
        sizes (from dim onwards) of the tensors in reference, a RuntimeError will be thrown by the
        call to narrow.

    """
    leading_shape = x.shape[:dim]

    unflattened = {}
    start = 0
    for name, ref_tensor in reference.items():
        tail_shape = ref_tensor.shape[dim:]
        n = tail_shape.numel()
        unflattened[name] = x.narrow(dim, start, n).reshape(*leading_shape, *tail_shape)
        start += n

    return unflattened
