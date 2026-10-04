from dataclasses import dataclass

import numpy as np


@dataclass
class TrustRegionConfig:
    """Configuration for the trust-region step size used in truncated natural policy gradient ascent."""

    delta: float
    reg_coeff: float = 1e-5

    def __post_init__(self):
        """Validate the trust-region configuration."""
        if self.delta <= 0:
            raise ValueError(f'delta must be strictly positive, got {self.delta}.')
        if self.reg_coeff < 0:
            raise ValueError(f'reg_coeff must be non-negative, got {self.reg_coeff}.')


def calculate_trust_region_step_size(quadratic_form: float, delta: float, eps: float = 1e-8) -> float:
    """
    Calculate the trust-region step size for a truncated natural policy gradient step.

    The step size is calculated so that, under a quadratic approximation of the Kullback-Leibler divergence
    between the current and updated policy, the divergence does not exceed delta. See Schulman et al. (2015),
    "Trust Region Policy Optimization", for the underlying derivation.

    Args:
        quadratic_form: The quadratic form, d^T F d, of the natural-gradient direction, d, with the Fisher
        Information matrix, F.
        delta: The Kullback-Leibler divergence budget (the trust-region radius).
        eps: A small value added to the denominator to avoid division by zero.

    Returns:
        The calculated step size. This value will be NaN if quadratic_form is negative enough to make the
        denominator negative, in which case the caller is expected to use a fallback step size.

    """
    with np.errstate(invalid='ignore'):
        return float(np.sqrt(2.0 * delta / (quadratic_form + eps)))
