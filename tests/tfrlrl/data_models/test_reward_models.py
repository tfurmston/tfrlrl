"""Tests for reward model dataclasses."""

import numpy as np
import pytest

from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward


class TestAverageEpisodicReward:
    """Tests for the AverageEpisodicReward dataclass."""

    def test_compute_returns_ndarray(self):
        """Test that compute returns a numpy array."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        assert isinstance(result, np.ndarray)

    def test_compute_output_length_matches_input(self):
        """Test that the output length equals the number of steps."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        result = model.compute(rewards)
        assert result.shape == (rewards.size,)

    def test_compute_known_sequence(self):
        """Test compute against a known reward sequence."""
        model = AverageEpisodicReward()
        # rewards = [1, 1, 1], T=3
        # G_0 = (1+1+1)/3 = 1.0, G_1 = (1+1)/3 = 2/3, G_2 = 1/3
        rewards = np.array([1.0, 1.0, 1.0])
        result = model.compute(rewards)
        expected = np.array([3.0 / 3, 2.0 / 3, 1.0 / 3])
        np.testing.assert_allclose(result, expected)

    def test_compute_single_step(self):
        """Test compute with a single-step episode."""
        model = AverageEpisodicReward()
        rewards = np.array([5.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, np.array([5.0]))

    def test_compute_all_zero_rewards(self):
        """Test compute with all-zero rewards returns zeros."""
        model = AverageEpisodicReward()
        rewards = np.zeros(4)
        result = model.compute(rewards)
        np.testing.assert_allclose(result, np.zeros(4))


class TestDiscountedReward:
    """Tests for the DiscountedReward dataclass."""

    def test_compute_returns_ndarray(self):
        """Test that compute returns a numpy array."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        assert isinstance(result, np.ndarray)

    def test_compute_output_length_matches_input(self):
        """Test that the output length equals the number of steps."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        result = model.compute(rewards)
        assert result.shape == (rewards.size,)

    def test_compute_gamma_zero_returns_only_current_reward(self):
        """Test that gamma=0 returns only the immediate reward at each step."""
        model = DiscountedReward(gamma=0.0)
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        # With gamma=0, G_t = r_t
        np.testing.assert_allclose(result, rewards)

    def test_compute_gamma_one_equals_cumulative_sum(self):
        """Test that gamma=1 produces a simple cumulative sum of future rewards."""
        model = DiscountedReward(gamma=1.0)
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        # G_0 = 1+2+3 = 6, G_1 = 2+3 = 5, G_2 = 3
        expected = np.array([6.0, 5.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_compute_known_sequence(self):
        """Test compute against a known discounted reward sequence."""
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
        """Test compute with a single-step episode."""
        model = DiscountedReward(gamma=0.99)
        rewards = np.array([5.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, np.array([5.0]))

    @pytest.mark.parametrize('gamma', [0.1, 0.5, 0.9, 0.99])
    def test_compute_returns_decrease_over_time(self, gamma: float):
        """Test that returns are non-increasing for constant positive rewards."""
        model = DiscountedReward(gamma=gamma)
        rewards = np.ones(5)
        result = model.compute(rewards)
        # Later steps have fewer remaining rewards, so returns should be non-increasing
        assert np.all(np.diff(result) <= 0)
