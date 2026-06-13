"""Tests for the reward model dataclasses."""

import numpy as np
import pytest

from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward


class TestAverageEpisodicReward:
    """Tests for the AverageEpisodicReward dataclass."""

    def test_compute_returns_numpy_array(self):
        """Test that compute returns a numpy array."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        assert isinstance(result, np.ndarray)

    def test_compute_known_sequence(self):
        """Test that compute produces expected values for a known sequence."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 1.0, 1.0])
        result = model.compute(rewards)
        # G_0 = (1+1+1)/3 = 1, G_1 = (1+1)/3 = 2/3, G_2 = 1/3
        expected = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(result, expected)

    def test_compute_output_shape(self):
        """Test that the output has the same shape as the input."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        result = model.compute(rewards)
        assert result.shape == rewards.shape

    def test_compute_single_step(self):
        """Test that compute works with a single reward."""
        model = AverageEpisodicReward()
        rewards = np.array([5.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, np.array([5.0]))


class TestDiscountedReward:
    """Tests for the DiscountedReward dataclass."""

    def test_compute_returns_numpy_array(self):
        """Test that compute returns a numpy array."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        assert isinstance(result, np.ndarray)

    def test_compute_output_shape(self):
        """Test that the output has the same shape as the input."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        result = model.compute(rewards)
        assert result.shape == rewards.shape

    def test_compute_known_sequence(self):
        """Test that compute produces expected values for a known reward sequence."""
        gamma = 0.5
        model = DiscountedReward(gamma=gamma)
        rewards = np.array([1.0, 1.0, 1.0])
        result = model.compute(rewards)
        # G_0 = 1 + 0.5 + 0.25 = 1.75
        # G_1 = 1 + 0.5 = 1.5
        # G_2 = 1
        expected = np.array([1.75, 1.5, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_compute_single_step(self):
        """Test that compute works with a single reward."""
        model = DiscountedReward(gamma=0.99)
        rewards = np.array([5.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, np.array([5.0]))

    def test_gamma_must_be_float(self):
        """Test that a non-float gamma raises TypeError."""
        with pytest.raises(TypeError):
            DiscountedReward(gamma=1)  # int, not float

    def test_gamma_zero_raises_value_error(self):
        """Test that gamma=0.0 raises ValueError."""
        with pytest.raises(ValueError):
            DiscountedReward(gamma=0.0)

    def test_gamma_one_raises_value_error(self):
        """Test that gamma=1.0 raises ValueError."""
        with pytest.raises(ValueError):
            DiscountedReward(gamma=1.0)

    def test_gamma_negative_raises_value_error(self):
        """Test that a negative gamma raises ValueError."""
        with pytest.raises(ValueError):
            DiscountedReward(gamma=-0.5)

    def test_gamma_greater_than_one_raises_value_error(self):
        """Test that gamma > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            DiscountedReward(gamma=1.5)

    @pytest.mark.parametrize('gamma', [0.01, 0.5, 0.9, 0.99, 0.999])
    def test_valid_gamma_values(self, gamma: float):
        """Test that valid gamma values in (0, 1) do not raise."""
        model = DiscountedReward(gamma=gamma)
        assert model.gamma == gamma
