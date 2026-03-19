"""Statistical anomaly detection using Z-score and IQR methods.

Provides a stateless detection interface that operates on pre-computed
``WindowStats`` objects, making it easy to compose with the sliding-window
feature engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from anomaly_detector.features.windows import WindowStats


class AnomalyType(str, Enum):
    """Classification of the detection method that fired."""

    ZSCORE = "zscore"
    IQR = "iqr"


@dataclass(frozen=True)
class AnomalyResult:
    """Immutable result from a single-feature anomaly check."""

    is_anomaly: bool
    feature: str
    value: float
    anomaly_type: AnomalyType | None
    score: float
    details: dict[str, Any]


class StatisticalDetector:
    """Z-score and IQR-based anomaly detector.

    For each feature, two independent tests are applied:

    1. **Z-score test** -- flags values whose absolute z-score exceeds
       ``zscore_threshold``.
    2. **IQR fence test** -- flags values outside the Tukey fences
       ``[Q1 - k*IQR, Q3 + k*IQR]`` where *k* = ``iqr_multiplier``.

    A feature is considered anomalous if *either* test fires.

    Args:
        zscore_threshold: Absolute z-score cutoff (default 3.0).
        iqr_multiplier: IQR fence multiplier (default 1.5).
        min_samples: Minimum window size before detection activates.
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_samples: int = 30,
    ) -> None:
        self._zscore_threshold = zscore_threshold
        self._iqr_multiplier = iqr_multiplier
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        feature_name: str,
        value: float,
        stats: WindowStats,
    ) -> AnomalyResult:
        """Run both Z-score and IQR tests on a single feature value.

        Args:
            feature_name: Human-readable name of the feature.
            value: The current observation to evaluate.
            stats: Pre-computed sliding-window statistics.

        Returns:
            An ``AnomalyResult`` indicating whether the value is anomalous.
        """
        if stats.count < self._min_samples:
            return AnomalyResult(
                is_anomaly=False,
                feature=feature_name,
                value=value,
                anomaly_type=None,
                score=0.0,
                details={"reason": "insufficient_samples", "count": stats.count},
            )

        zscore_flag = self._check_zscore(value, stats)
        iqr_flag = self._check_iqr(value, stats)

        is_anomaly = zscore_flag or iqr_flag
        anomaly_type: AnomalyType | None = None
        if zscore_flag and iqr_flag:
            anomaly_type = AnomalyType.ZSCORE  # prefer z-score when both fire
        elif zscore_flag:
            anomaly_type = AnomalyType.ZSCORE
        elif iqr_flag:
            anomaly_type = AnomalyType.IQR

        lower_fence = stats.p25 - self._iqr_multiplier * stats.iqr
        upper_fence = stats.p75 + self._iqr_multiplier * stats.iqr

        return AnomalyResult(
            is_anomaly=is_anomaly,
            feature=feature_name,
            value=value,
            anomaly_type=anomaly_type,
            score=abs(stats.zscore),
            details={
                "zscore": stats.zscore,
                "zscore_threshold": self._zscore_threshold,
                "zscore_flag": zscore_flag,
                "iqr": stats.iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "iqr_flag": iqr_flag,
                "window_mean": stats.mean,
                "window_std": stats.std,
                "window_count": stats.count,
            },
        )

    def check_multiple(
        self,
        stats_map: dict[str, WindowStats],
        values: dict[str, float],
    ) -> list[AnomalyResult]:
        """Run detection across multiple features.

        Args:
            stats_map: Mapping of feature name to its ``WindowStats``.
            values: Mapping of feature name to the current observation.

        Returns:
            List of ``AnomalyResult`` objects, one per feature present in
            both ``stats_map`` and ``values``.
        """
        results: list[AnomalyResult] = []
        for feature_name, stats in stats_map.items():
            if feature_name not in values:
                continue
            results.append(self.check(feature_name, values[feature_name], stats))
        return results

    # ------------------------------------------------------------------
    # Internal tests
    # ------------------------------------------------------------------

    def _check_zscore(self, value: float, stats: WindowStats) -> bool:
        """Return True if the value's z-score exceeds the threshold."""
        return abs(stats.zscore) > self._zscore_threshold

    def _check_iqr(self, value: float, stats: WindowStats) -> bool:
        """Return True if the value falls outside the IQR fences."""
        if stats.iqr == 0:
            return False
        lower = stats.p25 - self._iqr_multiplier * stats.iqr
        upper = stats.p75 + self._iqr_multiplier * stats.iqr
        return value < lower or value > upper
