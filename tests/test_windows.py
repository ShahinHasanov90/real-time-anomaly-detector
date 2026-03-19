"""Tests for the sliding window feature computation engine."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detector.features.windows import FeatureWindowManager, SlidingWindow, WindowStats


# ---------------------------------------------------------------------------
# SlidingWindow tests
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    """Unit tests for the SlidingWindow class."""

    def test_push_and_count(self) -> None:
        """Pushing values should increase the count up to max_size."""
        window = SlidingWindow(max_size=5)
        assert window.count == 0
        for i in range(7):
            window.push(float(i))
        assert window.count == 5  # capped at max_size

    def test_is_ready(self) -> None:
        """is_ready should be True only when at least 2 values exist."""
        window = SlidingWindow(max_size=10)
        assert window.is_ready is False
        window.push(1.0)
        assert window.is_ready is False
        window.push(2.0)
        assert window.is_ready is True

    def test_compute_stats_basic(self) -> None:
        """Verify basic statistics on a known data set."""
        window = SlidingWindow(max_size=100)
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        window.push_many(data)

        stats = window.compute_stats()
        assert stats.mean == pytest.approx(30.0)
        assert stats.min == pytest.approx(10.0)
        assert stats.max == pytest.approx(50.0)
        assert stats.median == pytest.approx(30.0)
        assert stats.count == 5

    def test_compute_stats_std(self) -> None:
        """Standard deviation should use ddof=1 (sample std)."""
        window = SlidingWindow(max_size=100)
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        window.push_many(data)
        stats = window.compute_stats()
        expected_std = float(np.std(data, ddof=1))
        assert stats.std == pytest.approx(expected_std, rel=1e-6)

    def test_zscore_of_current_value(self) -> None:
        """Z-score should be computed for the explicitly provided value."""
        window = SlidingWindow(max_size=100)
        for v in [100.0] * 50:
            window.push(v)
        window.push(100.0)  # one more to avoid zero std concern
        # Push a range to get nonzero std
        window.clear()
        window.push_many([100.0, 102.0, 98.0, 101.0, 99.0])
        stats = window.compute_stats(current_value=110.0)
        assert stats.zscore > 0

    def test_zscore_default_uses_last_value(self) -> None:
        """Without current_value, z-score should be for the last pushed value."""
        window = SlidingWindow(max_size=100)
        window.push_many([10.0, 10.0, 10.0, 10.0, 50.0])
        stats = window.compute_stats()
        assert stats.zscore > 0  # 50 is far from the mean

    def test_percentiles(self) -> None:
        """Percentile values should match numpy's calculation."""
        window = SlidingWindow(max_size=200)
        data = list(range(1, 101))  # 1..100
        window.push_many([float(x) for x in data])
        stats = window.compute_stats()
        assert stats.p25 == pytest.approx(np.percentile(data, 25))
        assert stats.p75 == pytest.approx(np.percentile(data, 75))
        assert stats.iqr == pytest.approx(stats.p75 - stats.p25)

    def test_compute_stats_raises_on_insufficient_data(self) -> None:
        """compute_stats should raise ValueError with < 2 data points."""
        window = SlidingWindow(max_size=10)
        with pytest.raises(ValueError, match="at least 2"):
            window.compute_stats()
        window.push(1.0)
        with pytest.raises(ValueError, match="at least 2"):
            window.compute_stats()

    def test_eviction_order(self) -> None:
        """Oldest values should be evicted first (FIFO)."""
        window = SlidingWindow(max_size=3)
        window.push_many([1.0, 2.0, 3.0, 4.0, 5.0])
        values = window.get_values()
        np.testing.assert_array_equal(values, [3.0, 4.0, 5.0])

    def test_clear(self) -> None:
        """clear() should reset the window to empty."""
        window = SlidingWindow(max_size=10)
        window.push_many([1.0, 2.0, 3.0])
        window.clear()
        assert window.count == 0
        assert window.is_ready is False

    def test_get_values_returns_copy(self) -> None:
        """get_values should return a copy, not a reference."""
        window = SlidingWindow(max_size=10)
        window.push_many([1.0, 2.0, 3.0])
        arr = window.get_values()
        arr[0] = 999.0
        assert window.get_values()[0] == pytest.approx(1.0)

    def test_zero_std_zscore(self) -> None:
        """When all values are identical, z-score should be 0."""
        window = SlidingWindow(max_size=100)
        window.push_many([42.0] * 10)
        stats = window.compute_stats(current_value=42.0)
        assert stats.zscore == 0.0


# ---------------------------------------------------------------------------
# FeatureWindowManager tests
# ---------------------------------------------------------------------------


class TestFeatureWindowManager:
    """Tests for the multi-feature window manager."""

    def test_push_populates_correct_windows(self) -> None:
        """Pushing an event should update only matching feature windows."""
        manager = FeatureWindowManager(
            feature_names=["declared_value", "quantity"], window_size=100
        )
        event = {"declared_value": 1000.0, "quantity": 50.0, "other_field": "ignored"}
        manager.push(event)
        assert manager.get_window("declared_value").count == 1
        assert manager.get_window("quantity").count == 1

    def test_unit_price_derivation(self) -> None:
        """unit_price should be derived when both value and quantity exist."""
        manager = FeatureWindowManager(
            feature_names=["declared_value", "quantity", "unit_price"], window_size=100
        )
        event = {"declared_value": 500.0, "quantity": 10.0}
        manager.push(event)
        assert manager.get_window("unit_price").count == 1
        values = manager.get_window("unit_price").get_values()
        assert values[0] == pytest.approx(50.0)

    def test_unit_price_not_derived_zero_quantity(self) -> None:
        """unit_price should NOT be derived if quantity is zero."""
        manager = FeatureWindowManager(
            feature_names=["unit_price"], window_size=100
        )
        event = {"declared_value": 500.0, "quantity": 0}
        manager.push(event)
        assert manager.get_window("unit_price").count == 0

    def test_compute_all_stats(self) -> None:
        """compute_all_stats should return stats for all ready windows."""
        manager = FeatureWindowManager(
            feature_names=["declared_value", "quantity"], window_size=100
        )
        for i in range(10):
            manager.push({"declared_value": 100.0 + i, "quantity": 50.0 + i})
        stats = manager.compute_all_stats()
        assert "declared_value" in stats
        assert "quantity" in stats
        assert isinstance(stats["declared_value"], WindowStats)

    def test_compute_all_stats_skips_unready(self) -> None:
        """Windows with < 2 values should be excluded from results."""
        manager = FeatureWindowManager(
            feature_names=["declared_value", "quantity"], window_size=100
        )
        manager.push({"declared_value": 100.0})  # only 1 value, no quantity
        stats = manager.compute_all_stats()
        assert len(stats) == 0

    def test_get_feature_vector(self) -> None:
        """Feature vector should contain z-scores for all features."""
        manager = FeatureWindowManager(
            feature_names=["declared_value", "quantity"], window_size=100
        )
        for i in range(20):
            manager.push({"declared_value": 100.0 + i * 0.1, "quantity": 50.0 + i * 0.1})
        vec = manager.get_feature_vector()
        assert vec is not None
        assert vec.shape == (2,)

    def test_get_feature_vector_returns_none_when_unready(self) -> None:
        """Should return None if any window lacks sufficient data."""
        manager = FeatureWindowManager(
            feature_names=["declared_value", "quantity"], window_size=100
        )
        manager.push({"declared_value": 100.0})
        vec = manager.get_feature_vector()
        assert vec is None

    def test_get_window_raises_on_unknown(self) -> None:
        """Requesting a non-existent feature should raise KeyError."""
        manager = FeatureWindowManager(feature_names=["declared_value"], window_size=10)
        with pytest.raises(KeyError):
            manager.get_window("nonexistent")

    def test_feature_names_property(self) -> None:
        """feature_names should reflect the configured features."""
        names = ["a", "b", "c"]
        manager = FeatureWindowManager(feature_names=names, window_size=10)
        assert manager.feature_names == names
