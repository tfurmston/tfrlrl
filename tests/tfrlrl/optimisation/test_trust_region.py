import numpy as np
import pytest

from tfrlrl.optimisation.trust_region import TrustRegionConfig, calculate_trust_region_step_size


@pytest.mark.parametrize('delta', [0.0, -0.01])
def test_trust_region_config_raises_for_non_positive_delta(delta: float):
    """
    Test that TrustRegionConfig raises a ValueError for a non-positive delta.

    Args:
        delta: A non-positive value for delta.

    """
    with pytest.raises(ValueError):
        TrustRegionConfig(delta=delta)


def test_trust_region_config_raises_for_negative_reg_coeff():
    """Test that TrustRegionConfig raises a ValueError for a negative reg_coeff."""
    with pytest.raises(ValueError):
        TrustRegionConfig(delta=0.01, reg_coeff=-1e-5)


def test_trust_region_config_defaults():
    """Test that TrustRegionConfig has the expected default reg_coeff."""
    config = TrustRegionConfig(delta=0.01)
    assert config.delta == 0.01
    assert config.reg_coeff == 1e-5


@pytest.mark.parametrize(
    'quadratic_form, delta',
    [
        (1.0, 0.01),
        (0.5, 0.02),
        (10.0, 0.5),
    ],
)
def test_calculate_trust_region_step_size_matches_hand_calculation(quadratic_form: float, delta: float):
    """
    Test that calculate_trust_region_step_size matches a hand-calculated value.

    Args:
        quadratic_form: The quadratic form, d^T F d, of the natural-gradient direction.
        delta: The Kullback-Leibler divergence budget.

    """
    step_size = calculate_trust_region_step_size(quadratic_form, delta)
    expected = np.sqrt(2.0 * delta / quadratic_form)
    np.testing.assert_allclose(step_size, expected, rtol=1e-6)


def test_calculate_trust_region_step_size_is_nan_for_negative_quadratic_form():
    """Test that calculate_trust_region_step_size returns NaN when the quadratic form is negative enough."""
    step_size = calculate_trust_region_step_size(quadratic_form=-1.0, delta=0.01, eps=1e-8)
    assert np.isnan(step_size)
