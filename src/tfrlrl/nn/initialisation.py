"""Named weight-initialisation schemes for PyTorch nn.Linear layers."""

from typing import Callable, Dict

from torch import nn

LayerInitFn = Callable[[nn.Linear], None]


def init_pytorch_default(layer: nn.Linear) -> None:
    """No-op: leaves the layer's own constructor-time initialisation in place."""
    pass


def init_xavier_uniform_zero_bias(layer: nn.Linear) -> None:
    """Initialise the layer's weight with Xavier/Glorot-uniform values and its bias to zero."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


def init_orthogonal(gain: float) -> LayerInitFn:
    """
    Construct an orthogonal initialiser with the given gain.

    Args:
        gain: The scaling factor applied to the orthogonal weight matrix.

    Returns:
        A function that initialises a layer's weight orthogonally (scaled by gain) and its bias to zero.

    """

    def _init(layer: nn.Linear) -> None:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    return _init


WEIGHT_INIT_REGISTRY: Dict[str, Dict[str, LayerInitFn]] = {
    'pytorch_default': {'hidden': init_pytorch_default, 'output': init_pytorch_default},
    'xavier_uniform_zero_bias': {
        'hidden': init_xavier_uniform_zero_bias,
        'output': init_xavier_uniform_zero_bias,
    },
    'orthogonal': {
        'hidden': init_orthogonal(gain=nn.init.calculate_gain('tanh')),
        'output': init_orthogonal(gain=0.01),
    },
}


def validate_weight_init_name(weight_init: str) -> None:
    """
    Validate that the given name is a registered weight-initialisation scheme.

    Args:
        weight_init: The name to validate against WEIGHT_INIT_REGISTRY.

    Raises:
        ValueError: If weight_init is not a key of WEIGHT_INIT_REGISTRY.

    """
    if weight_init not in WEIGHT_INIT_REGISTRY:
        raise ValueError(f'weight_init must be one of {sorted(WEIGHT_INIT_REGISTRY)}, got {weight_init!r}.')
