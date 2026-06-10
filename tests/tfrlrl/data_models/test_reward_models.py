"""Tests for reward model dataclasses."""

import numpy as np
import pytest

from tfrlrl.data_models.reward_models import AverageEpisodicReward, DiscountedReward


class TestAverageEpisodicReward:
    """Tests for the AverageEpisodicReward dataclass."""

    def test_single_step(self):
        """A single-step episode: G_0 = r_0 / 1."""
        model = AverageEpisodicReward()
        rewards = np.array([5.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, np.array([5.0]))

    def test_uniform_rewards(self):
        """All rewards equal to 1: each G_t = (T - t) / T."""
        model = AverageEpisodicReward()
        T = 4
        rewards = np.ones(T)
        result = model.compute(rewards)
        expected = np.array([(T - t) / T for t in range(T)])
        np.testing.assert_allclose(result, expected)

    def test_known_sequence(self):
        """Verify against the manual lower-triangular matrix formula."""
        model = AverageEpisodicReward()
        rewards = np.array([1.0, 2.0, 3.0])
        T = rewards.size
        expected = np.matmul(rewards, np.tril(np.ones(T))) / T
        result = model.compute(rewards)
        np.testing.assert_allclose(result, expected)

    def test_returns_ndarray(self):
        """compute() must return a numpy array."""
        model = AverageEpisodicReward()
        result = model.compute(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self):
        """Output array length must equal the input reward length."""
        model = AverageEpisodicReward()
        for T in [1, 3, 10]:
            rewards = np.arange(T, dtype=float)
            assert model.compute(rewards).shape == (T,)


class TestDiscountedReward:
    """Tests for the DiscountedReward dataclass."""

    def test_single_step(self):
        """A single-step episode: G_0 = r_0 regardless of gamma."""
        model = DiscountedReward(gamma=0.9)
        rewards = np.array([7.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, np.array([7.0]))

    def test_gamma_zero(self):
        """gamma=0: each G_t equals only the immediate reward r_t."""
        model = DiscountedReward(gamma=0.0)
        rewards = np.array([1.0, 2.0, 3.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, rewards)

    def test_gamma_one_equals_cumulative_sum(self):
        """gamma=1: G_t = r_t + r_{t+1} + ... + r_{T-1} (undiscounted sum)."""
        model = DiscountedReward(gamma=1.0)
        rewards = np.array([1.0, 2.0, 3.0])
        T = rewards.size
        expected = np.matmul(rewards, np.tril(np.ones(T)))
        result = model.compute(rewards)
        np.testing.assert_allclose(result, expected)

    def test_known_discounted_sequence(self):
        """Verify discounted returns for a known reward sequence and gamma."""
        gamma = 0.5
        model = DiscountedReward(gamma=gamma)
        rewards = np.array([1.0, 1.0, 1.0])
        # G_0 = 1 + 0.5 + 0.25 = 1.75
        # G_1 = 1 + 0.5 = 1.5
        # G_2 = 1
        expected = np.array([1.75, 1.5, 1.0])
        result = model.compute(rewards)
        np.testing.assert_allclose(result, expected)

    def test_default_gamma(self):
        """Default gamma should be 0.99."""
        model = DiscountedReward()
        assert model.gamma == 0.99

    def test_returns_ndarray(self):
        """compute() must return a numpy array."""
        model = DiscountedReward(gamma=0.99)
        result = model.compute(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self):
        """Output array length must equal the input reward length."""
        model = DiscountedReward(gamma=0.99)
        for T in [1, 3, 10]:
            rewards = np.arange(T, dtype=float)
            assert model.compute(rewards).shape == (T,)

    @pytest.mark.parametrize('gamma', [0.9, 0.95, 0.99])
    def test_returns_decrease_with_discount(self, gamma: float):
        """G_t >= G_{t+1} for positive rewards when gamma <= 1."""
        model = DiscountedReward(gamma=gamma)
        rewards = np.ones(5)
        result = model.compute(rewards)
        for t in range(len(result) - 1):
            assert result[t] >= result[t + 1]
