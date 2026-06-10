"""Tests for reward model dataclasses."""

import numpy as np
import pytest

from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward


class TestAverageEpisodicReward:
    """Tests for the AverageEpisodicReward class."""

    def test_returns_numpy_array(self):
        """compute() returns a numpy array."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self):
        """Output length equals input length."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        assert model.compute(rewards).shape == rewards.shape

    def test_known_sequence(self):
        """Verify correct values on a known reward sequence."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0])
        # G_t = (r_t + r_{t+1} + ... + r_{T-1}) / T
        # G_0 = (1+2+3)/3=2, G_1 = (2+3)/3=5/3, G_2 = 3/3=1
        expected = np.array([6.0 / 3, 5.0 / 3, 3.0 / 3])
        np.testing.assert_allclose(model.compute(rewards), expected)

    def test_single_step(self):
        """Single-step episode: return equals reward divided by 1."""
        model = AverageEpisodicReward()
        rewards = np.array([5.0])
        np.testing.assert_allclose(model.compute(rewards), np.array([5.0]))

    def test_zero_rewards(self):
        """All-zero rewards produce all-zero returns."""
        model = AverageEpisodicReward()
        rewards = np.zeros(5)
        np.testing.assert_allclose(model.compute(rewards), np.zeros(5))


class TestDiscountedReward:
    """Tests for the DiscountedReward class."""

    def test_returns_numpy_array(self):
        """compute() returns a numpy array."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self):
        """Output length equals input length."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        assert model.compute(rewards).shape == rewards.shape

    def test_gamma_zero_returns_only_current_reward(self):
        """With gamma=0, each G_t equals only r_t."""
        model = DiscountedReward(gamma=0.0)
        rewards = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(model.compute(rewards), rewards)

    def test_gamma_one_is_undiscounted_sum(self):
        """With gamma=1, G_t is the simple sum of future rewards."""
        model = DiscountedReward(gamma=1.0)
        rewards = np.array([1.0, 2.0, 3.0])
        # G_0 = 1+2+3=6, G_1 = 2+3=5, G_2 = 3
        expected = np.array([6.0, 5.0, 3.0])
        np.testing.assert_allclose(model.compute(rewards), expected)

    def test_known_sequence_with_gamma(self):
        """Verify correct discounted values on a known reward sequence."""
        gamma = 0.9
        model = DiscountedReward(gamma=gamma)
        rewards = np.array([1.0, 1.0, 1.0])
        # G_0 = 1 + 0.9 + 0.81 = 2.71, G_1 = 1 + 0.9 = 1.9, G_2 = 1
        expected = np.array([1 + gamma + gamma**2, 1 + gamma, 1.0])
        np.testing.assert_allclose(model.compute(rewards), expected)

    def test_single_step(self):
        """Single-step episode: return equals the single reward regardless of gamma."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([5.0])
        np.testing.assert_allclose(model.compute(rewards), np.array([5.0]))

    def test_zero_rewards(self):
        """All-zero rewards produce all-zero returns."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.zeros(5)
        np.testing.assert_allclose(model.compute(rewards), np.zeros(5))
