"""Tests for the statistical anomaly detector (Z-score and IQR methods)."""

from __future__ import annotations

import pytest

from anomaly_detector.features.windows import WindowStats
from anomaly_detector.models.statistical import AnomalyResult, AnomalyType, StatisticalDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stats(
    mean: float = 100.0,
    std: float = 10.0,
    min_val: float = 70.0,
    max_val: float = 130.0,
    median: float = 100.0,
    p25: float = 93.0,
    p75: float = 107.0,
    zscore: float = 0.0,
    count: int = 100,
) -> WindowStats:
    """Helper to construct a WindowStats with sensible defaults."""
    return WindowStats(
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        median=median,
        p25=p25,
        p75=p75,
        iqr=p75 - p25,
        zscore=zscore,
        count=count,
    )


@pytest.fixture
def detector() -> StatisticalDetector:
    """Return a StatisticalDetector with default thresholds."""
    return StatisticalDetector(zscore_threshold=3.0, iqr_multiplier=1.5, min_samples=30)


# ---------------------------------------------------------------------------
# Z-score tests
# ---------------------------------------------------------------------------


class TestZScoreDetection:
    """Tests for the z-score detection path."""

    def test_normal_value_not_flagged(self, detector: StatisticalDetector) -> None:
        """A value within the z-score threshold should not be anomalous."""
        stats = _make_stats(zscore=1.5)
        result = detector.check("declared_value", 115.0, stats)
        assert result.is_anomaly is False
        assert result.score == pytest.approx(1.5)

    def test_high_zscore_flagged(self, detector: StatisticalDetector) -> None:
        """A value exceeding the z-score threshold should be flagged."""
        stats = _make_stats(zscore=4.2)
        result = detector.check("declared_value", 142.0, stats)
        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.ZSCORE
        assert result.score == pytest.approx(4.2)

    def test_negative_zscore_flagged(self, detector: StatisticalDetector) -> None:
        """A large negative z-score should also trigger detection."""
        stats = _make_stats(zscore=-3.5)
        result = detector.check("declared_value", 65.0, stats)
        assert result.is_anomaly is True
        assert result.score == pytest.approx(3.5)

    def test_boundary_zscore_not_flagged(self, detector: StatisticalDetector) -> None:
        """A value exactly at the threshold boundary should NOT fire."""
        stats = _make_stats(zscore=3.0)
        result = detector.check("declared_value", 130.0, stats)
        assert result.is_anomaly is False


# ---------------------------------------------------------------------------
# IQR tests
# ---------------------------------------------------------------------------


class TestIQRDetection:
    """Tests for the IQR fence detection path."""

    def test_value_within_fences(self, detector: StatisticalDetector) -> None:
        """A value inside the IQR fences should not be flagged."""
        # Fences: 93 - 1.5*14 = 72, 107 + 1.5*14 = 128
        stats = _make_stats(zscore=1.0)
        result = detector.check("declared_value", 110.0, stats)
        assert result.is_anomaly is False

    def test_value_above_upper_fence(self, detector: StatisticalDetector) -> None:
        """A value above the upper fence should be flagged."""
        # Upper fence: 107 + 1.5*14 = 128
        stats = _make_stats(zscore=3.5)
        result = detector.check("declared_value", 135.0, stats)
        assert result.is_anomaly is True

    def test_value_below_lower_fence(self, detector: StatisticalDetector) -> None:
        """A value below the lower fence should be flagged."""
        # Lower fence: 93 - 1.5*14 = 72
        stats = _make_stats(zscore=-3.5)
        result = detector.check("declared_value", 60.0, stats)
        assert result.is_anomaly is True

    def test_zero_iqr_not_flagged(self, detector: StatisticalDetector) -> None:
        """When IQR is zero, the IQR test should not fire (avoid div by zero)."""
        stats = _make_stats(p25=100.0, p75=100.0, zscore=0.5)
        result = detector.check("declared_value", 100.5, stats)
        assert result.is_anomaly is False


# ---------------------------------------------------------------------------
# Insufficient samples
# ---------------------------------------------------------------------------


class TestInsufficientSamples:
    """Detection should be suppressed when the window is too small."""

    def test_below_min_samples(self, detector: StatisticalDetector) -> None:
        """With fewer than min_samples, no anomaly should be reported."""
        stats = _make_stats(zscore=10.0, count=10)
        result = detector.check("declared_value", 200.0, stats)
        assert result.is_anomaly is False
        assert result.details["reason"] == "insufficient_samples"

    def test_at_min_samples(self, detector: StatisticalDetector) -> None:
        """Exactly at min_samples, detection should activate."""
        stats = _make_stats(zscore=4.0, count=30)
        result = detector.check("declared_value", 140.0, stats)
        assert result.is_anomaly is True


# ---------------------------------------------------------------------------
# Multi-feature detection
# ---------------------------------------------------------------------------


class TestMultiFeature:
    """Tests for the check_multiple method."""

    def test_multiple_features(self, detector: StatisticalDetector) -> None:
        """check_multiple should return one result per matched feature."""
        stats_map = {
            "declared_value": _make_stats(zscore=4.0),
            "quantity": _make_stats(mean=50.0, std=5.0, zscore=1.0),
        }
        values = {"declared_value": 140.0, "quantity": 55.0}
        results = detector.check_multiple(stats_map, values)
        assert len(results) == 2
        anomalous = [r for r in results if r.is_anomaly]
        assert len(anomalous) == 1
        assert anomalous[0].feature == "declared_value"

    def test_missing_feature_skipped(self, detector: StatisticalDetector) -> None:
        """Features present in stats but missing from values are skipped."""
        stats_map = {
            "declared_value": _make_stats(zscore=4.0),
            "quantity": _make_stats(zscore=4.0),
        }
        values = {"declared_value": 140.0}
        results = detector.check_multiple(stats_map, values)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    """Validate that custom threshold parameters take effect."""

    def test_strict_zscore_threshold(self) -> None:
        """A stricter z-score threshold should flag more values."""
        strict = StatisticalDetector(zscore_threshold=2.0, iqr_multiplier=1.5, min_samples=5)
        stats = _make_stats(zscore=2.5, count=50)
        result = strict.check("declared_value", 125.0, stats)
        assert result.is_anomaly is True

    def test_loose_iqr_multiplier(self) -> None:
        """A larger IQR multiplier should allow wider range."""
        loose = StatisticalDetector(zscore_threshold=10.0, iqr_multiplier=3.0, min_samples=5)
        # Fences with 3.0: 93 - 3*14 = 51, 107 + 3*14 = 149
        stats = _make_stats(zscore=2.0, count=50)
        result = loose.check("declared_value", 135.0, stats)
        assert result.is_anomaly is False
