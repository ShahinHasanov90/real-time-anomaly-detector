"""Sliding window feature computation for streaming anomaly detection.

Maintains fixed-size circular buffers per feature and computes rolling
statistics (mean, standard deviation, percentiles, z-scores) efficiently
as new data points arrive.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class WindowStats:
    """Container for computed sliding-window statistics on a single feature."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    p25: float
    p75: float
    iqr: float
    zscore: float
    count: int


class SlidingWindow:
    """Fixed-size sliding window that tracks values and computes statistics.

    Uses a circular buffer (``collections.deque``) capped at ``max_size``.
    Statistics are computed on demand from the current buffer contents via
    NumPy for performance.

    Args:
        max_size: Maximum number of values retained in the window.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._buffer: deque[float] = deque(maxlen=max_size)
        self._array_cache: NDArray[np.float64] | None = None
        self._dirty: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Return the current number of values in the window."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """Return True when the window has at least two values."""
        return len(self._buffer) >= 2

    def push(self, value: float) -> None:
        """Append a new value, evicting the oldest if at capacity.

        Args:
            value: The numeric observation to add.
        """
        self._buffer.append(value)
        self._dirty = True

    def push_many(self, values: list[float] | NDArray[np.float64]) -> None:
        """Append multiple values at once.

        Args:
            values: Iterable of numeric observations.
        """
        for v in values:
            self._buffer.append(v)
        self._dirty = True

    def compute_stats(self, current_value: float | None = None) -> WindowStats:
        """Compute descriptive statistics over the current window.

        If ``current_value`` is provided, the z-score is calculated for that
        value relative to the window distribution.  Otherwise the z-score of
        the most recent value in the buffer is returned.

        Args:
            current_value: Optional value for which to compute the z-score.

        Returns:
            A ``WindowStats`` dataclass with all rolling statistics.

        Raises:
            ValueError: If the window contains fewer than two data points.
        """
        if not self.is_ready:
            raise ValueError(
                f"Window needs at least 2 data points, currently has {self.count}"
            )

        arr = self._as_array()
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        p25, median, p75 = np.percentile(arr, [25, 50, 75]).tolist()
        iqr = p75 - p25

        ref = current_value if current_value is not None else float(arr[-1])
        zscore = (ref - mean) / std if std > 0 else 0.0

        return WindowStats(
            mean=mean,
            std=std,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=median,
            p25=p25,
            p75=p75,
            iqr=iqr,
            zscore=zscore,
            count=len(arr),
        )

    def get_values(self) -> NDArray[np.float64]:
        """Return the current window contents as a NumPy array."""
        return self._as_array().copy()

    def clear(self) -> None:
        """Remove all values from the window."""
        self._buffer.clear()
        self._dirty = True
        self._array_cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _as_array(self) -> NDArray[np.float64]:
        """Lazily materialize the deque as a NumPy array, caching the result."""
        if self._dirty or self._array_cache is None:
            self._array_cache = np.array(self._buffer, dtype=np.float64)
            self._dirty = False
        return self._array_cache


class FeatureWindowManager:
    """Manage multiple sliding windows -- one per monitored feature.

    Provides a unified interface for pushing observations and computing
    statistics across all configured feature dimensions.

    Args:
        feature_names: List of feature names to track.
        window_size: Maximum sliding-window size for each feature.
    """

    def __init__(self, feature_names: list[str], window_size: int = 1000) -> None:
        self._feature_names = list(feature_names)
        self._window_size = window_size
        self._windows: dict[str, SlidingWindow] = {
            name: SlidingWindow(max_size=window_size) for name in feature_names
        }

    @property
    def feature_names(self) -> list[str]:
        """Return the list of tracked feature names."""
        return list(self._feature_names)

    def push(self, event: dict[str, Any]) -> None:
        """Push feature values from an event into their respective windows.

        Only features present in both the event and the configured feature
        list are updated.  Missing features are silently skipped.

        Additionally, if both ``declared_value`` and ``quantity`` are
        present, a derived ``unit_price`` feature is computed and pushed.

        Args:
            event: A dictionary representing a single trade event.
        """
        # Derive unit_price when possible
        if (
            "unit_price" in self._windows
            and "declared_value" in event
            and "quantity" in event
            and event["quantity"] > 0
        ):
            event = dict(event)  # shallow copy to avoid mutation
            event["unit_price"] = event["declared_value"] / event["quantity"]

        for name, window in self._windows.items():
            if name in event:
                window.push(float(event[name]))

    def compute_all_stats(
        self, event: dict[str, Any] | None = None
    ) -> dict[str, WindowStats]:
        """Compute statistics for every feature window that is ready.

        Args:
            event: Optional current event whose values are used for z-score
                computation.  If None, the latest buffered value is used.

        Returns:
            Dictionary mapping feature name to its ``WindowStats``.
        """
        results: dict[str, WindowStats] = {}
        for name, window in self._windows.items():
            if not window.is_ready:
                continue
            current_value: float | None = None
            if event is not None and name in event:
                current_value = float(event[name])
            results[name] = window.compute_stats(current_value=current_value)
        return results

    def get_feature_vector(self, event: dict[str, Any] | None = None) -> NDArray[np.float64] | None:
        """Build a feature vector of z-scores suitable for model input.

        Returns None if any tracked window is not yet ready (insufficient
        data).

        Args:
            event: Optional current event for z-score reference.

        Returns:
            1-D NumPy array of z-scores, one per feature, or None.
        """
        stats = self.compute_all_stats(event)
        if len(stats) < len(self._feature_names):
            return None
        return np.array(
            [stats[name].zscore for name in self._feature_names], dtype=np.float64
        )

    def get_window(self, feature_name: str) -> SlidingWindow:
        """Return the underlying ``SlidingWindow`` for a specific feature.

        Args:
            feature_name: Name of the feature.

        Returns:
            The corresponding ``SlidingWindow`` instance.

        Raises:
            KeyError: If the feature name is not tracked.
        """
        return self._windows[feature_name]
