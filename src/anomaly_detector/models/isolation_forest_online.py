"""Online Isolation Forest with periodic retraining.

Wraps scikit-learn's ``IsolationForest`` and maintains a sliding buffer of
recent feature vectors.  The model is retrained every ``retrain_interval``
observations so it adapts to distribution drift without requiring a full
batch pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from anomaly_detector.metrics import (
    ISOLATION_FOREST_RETRAINS,
    ISOLATION_FOREST_SAMPLES,
    MODEL_INFERENCE_LATENCY,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class IsolationForestResult:
    """Result of an Isolation Forest anomaly check."""

    is_anomaly: bool
    score: float
    raw_score: float
    details: dict[str, Any]


class OnlineIsolationForest:
    """Isolation Forest with periodic online retraining.

    Maintains a fixed-size buffer of the most recent feature vectors.
    After every ``retrain_interval`` new observations the model is re-fit
    on the buffered data so that it tracks concept drift.

    Args:
        n_estimators: Number of trees in the forest.
        contamination: Expected proportion of anomalies in the data.
        buffer_size: Maximum number of feature vectors retained for training.
        retrain_interval: Number of new observations between retraining.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        buffer_size: int = 5000,
        retrain_interval: int = 5000,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._contamination = contamination
        self._buffer_size = buffer_size
        self._retrain_interval = retrain_interval
        self._random_state = random_state

        self._model: IsolationForest | None = None
        self._buffer: list[NDArray[np.float64]] = []
        self._samples_since_train: int = 0
        self._total_samples: int = 0
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Return True if the model has been trained at least once."""
        return self._is_trained

    @property
    def buffer_count(self) -> int:
        """Return the number of feature vectors currently buffered."""
        return len(self._buffer)

    def update(self, feature_vector: NDArray[np.float64]) -> None:
        """Add a feature vector to the training buffer and retrain if due.

        Args:
            feature_vector: 1-D array of feature values.
        """
        self._buffer.append(feature_vector.copy())
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size :]

        self._samples_since_train += 1
        self._total_samples += 1

        if self._samples_since_train >= self._retrain_interval and len(self._buffer) >= 50:
            self._train()

    def predict(self, feature_vector: NDArray[np.float64]) -> IsolationForestResult:
        """Score a single feature vector against the trained model.

        If the model has not been trained yet, the result will indicate
        ``is_anomaly=False`` with a score of 0.

        Args:
            feature_vector: 1-D array of feature values.

        Returns:
            An ``IsolationForestResult`` with the anomaly decision and scores.
        """
        if not self._is_trained or self._model is None:
            return IsolationForestResult(
                is_anomaly=False,
                score=0.0,
                raw_score=0.0,
                details={"reason": "model_not_trained", "buffer_size": len(self._buffer)},
            )

        start = time.perf_counter()
        X = feature_vector.reshape(1, -1)
        raw_score = float(self._model.decision_function(X)[0])
        prediction = int(self._model.predict(X)[0])
        elapsed = time.perf_counter() - start

        MODEL_INFERENCE_LATENCY.labels(model="isolation_forest").observe(elapsed)

        # IsolationForest: -1 = anomaly, 1 = normal
        is_anomaly = prediction == -1

        # Normalize the raw score to a 0-1 range where higher = more anomalous
        normalized_score = max(0.0, -raw_score)

        return IsolationForestResult(
            is_anomaly=is_anomaly,
            score=normalized_score,
            raw_score=raw_score,
            details={
                "prediction": prediction,
                "inference_time_ms": elapsed * 1000,
                "total_samples_seen": self._total_samples,
            },
        )

    def force_train(self) -> None:
        """Force an immediate retraining cycle.

        Raises:
            ValueError: If the buffer has fewer than 50 samples.
        """
        if len(self._buffer) < 50:
            raise ValueError(
                f"Need at least 50 samples to train, buffer has {len(self._buffer)}"
            )
        self._train()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train(self) -> None:
        """Fit a new Isolation Forest on the current buffer."""
        X = np.array(self._buffer, dtype=np.float64)
        logger.info(
            "retraining_isolation_forest",
            n_samples=X.shape[0],
            n_features=X.shape[1],
            total_observed=self._total_samples,
        )

        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        self._is_trained = True
        self._samples_since_train = 0

        ISOLATION_FOREST_RETRAINS.inc()
        ISOLATION_FOREST_SAMPLES.set(X.shape[0])
