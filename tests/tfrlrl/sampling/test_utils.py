"""Tests for sampling utility functions."""

from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.data_models.statistics import BaseStatistics, StatisticsException
from tfrlrl.sampling.utils import merge_optional_statistics


@dataclass
class MockStatistics(BaseStatistics):
    """Mock statistics class for testing merge_optional_statistics."""

    optional_attr: np.ndarray = None


class TestMergeOptionalStatistics:
    """Tests for the merge_optional_statistics function."""

    def test_all_attributes_none_returns_none(self):
        """Test that when all attributes are None, the function returns None."""
        stats = [MockStatistics(optional_attr=None) for _ in range(3)]
        result = merge_optional_statistics(stats, 'optional_attr')
        assert result is None

    def test_all_attributes_present_returns_concatenated_array(self):
        """Test that when all attributes are present, returns concatenated array."""
        arrays = [np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5, 6]])]
        stats = [MockStatistics(optional_attr=arr) for arr in arrays]

        result = merge_optional_statistics(stats, 'optional_attr')

        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_mixed_none_and_present_raises_exception(self):
        """Test that mixed None and present attributes raises StatisticsException."""
        stats = [
            MockStatistics(optional_attr=np.array([1, 2, 3])),
            MockStatistics(optional_attr=None),
            MockStatistics(optional_attr=np.array([4, 5, 6])),
        ]

        with pytest.raises(StatisticsException) as exc_info:
            merge_optional_statistics(stats, 'optional_attr')

        assert 'All baseline features should either be None or a NumPy array' in str(exc_info.value)

    def test_concatenate_axis_0(self):
        """Test concatenation along axis 0 (default)."""
        arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        stats = [MockStatistics(optional_attr=arr) for arr in arrays]

        result = merge_optional_statistics(stats, 'optional_attr', concatenate_axis=0)

        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(result, expected)

    def test_concatenate_axis_1(self):
        """Test concatenation along axis 1."""
        arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        stats = [MockStatistics(optional_attr=arr) for arr in arrays]

        result = merge_optional_statistics(stats, 'optional_attr', concatenate_axis=1)

        expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        np.testing.assert_array_equal(result, expected)

    def test_single_statistics_all_none(self):
        """Test with single statistics object where attribute is None."""
        stats = [MockStatistics(optional_attr=None)]
        result = merge_optional_statistics(stats, 'optional_attr')
        assert result is None

    def test_single_statistics_present(self):
        """Test with single statistics object where attribute is present."""
        arr = np.array([[1, 2, 3]])
        stats = [MockStatistics(optional_attr=arr)]

        result = merge_optional_statistics(stats, 'optional_attr')

        np.testing.assert_array_equal(result, arr)

    @given(n_stats=st.integers(min_value=2, max_value=10))
    @settings(deadline=1000)
    def test_concatenation_preserves_total_elements(self, n_stats: int):
        """
        Test that concatenation preserves the total number of elements.

        :param n_stats: The number of statistics objects to merge.
        """
        arrays = [np.random.random((3, 4)) for _ in range(n_stats)]
        stats = [MockStatistics(optional_attr=arr) for arr in arrays]

        result = merge_optional_statistics(stats, 'optional_attr', concatenate_axis=0)

        assert result.shape == (3 * n_stats, 4)
        total_elements = sum(arr.size for arr in arrays)
        assert result.size == total_elements
