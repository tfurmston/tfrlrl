import numpy as np
import pytest
import torch
from torch import nn

from tfrlrl.nn.initialisation import (
    WEIGHT_INIT_REGISTRY,
    init_orthogonal,
    init_pytorch_default,
    init_xavier_uniform_zero_bias,
    validate_weight_init_name,
)


@pytest.mark.parametrize('weight_init', list(WEIGHT_INIT_REGISTRY))
def test_weight_init_registry_has_hidden_and_output_roles(weight_init: str):
    """
    Test that every scheme in WEIGHT_INIT_REGISTRY defines both a 'hidden' and an 'output' role.

    Args:
        weight_init: The name of the weight-initialisation scheme to check.

    """
    assert set(WEIGHT_INIT_REGISTRY[weight_init]) == {'hidden', 'output'}


@pytest.mark.parametrize('weight_init', list(WEIGHT_INIT_REGISTRY))
def test_validate_weight_init_name_accepts_registered_names(weight_init: str):
    """
    Test that validate_weight_init_name does not raise for any registered scheme name.

    Args:
        weight_init: The name of the weight-initialisation scheme to validate.

    """
    validate_weight_init_name(weight_init)


def test_validate_weight_init_name_raises_for_unregistered_name():
    """Test that validate_weight_init_name raises a ValueError for an unregistered scheme name."""
    with pytest.raises(ValueError):
        validate_weight_init_name('not-a-real-scheme')


def test_init_pytorch_default_is_a_no_op():
    """Test that init_pytorch_default leaves the layer's weight and bias unchanged."""
    torch.manual_seed(0)
    layer = nn.Linear(8, 4)
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()

    init_pytorch_default(layer)

    assert torch.equal(layer.weight, original_weight)
    assert torch.equal(layer.bias, original_bias)


def test_init_xavier_uniform_zero_bias():
    """Test that init_xavier_uniform_zero_bias zeros the bias and bounds the weight correctly."""
    layer = nn.Linear(8, 4)

    init_xavier_uniform_zero_bias(layer)

    assert torch.equal(layer.bias, torch.zeros_like(layer.bias))
    bound = np.sqrt(6.0 / (8 + 4))
    assert torch.all(layer.weight.abs() <= bound + 1e-6)


@pytest.mark.parametrize('gain', [0.01, 1.0, np.sqrt(2)])
def test_init_orthogonal_scales_orthonormal_rows_by_gain(gain: float):
    """
    Test that init_orthogonal zeros the bias and produces rows scaled to the given gain.

    The layer's out_features is kept smaller than its in_features, so nn.init.orthogonal_ is guaranteed to
    produce orthonormal rows (rather than orthonormal columns), making weight @ weight.T a predictable,
    gain-scaled identity matrix.

    Args:
        gain: The scaling factor to apply to the orthogonal weight matrix.

    """
    layer = nn.Linear(8, 4)

    init_orthogonal(gain)(layer)

    assert torch.equal(layer.bias, torch.zeros_like(layer.bias))
    gram = layer.weight.detach() @ layer.weight.detach().T
    np.testing.assert_allclose(gram.numpy(), (gain**2) * np.eye(4), atol=1e-5)
