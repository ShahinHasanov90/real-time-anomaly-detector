"""Alert management with severity classification and deduplication.

The ``AlertManager`` converts raw anomaly results into structured alerts,
assigns severity levels based on configurable score thresholds, and
suppresses duplicate alerts within a sliding time window.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import structlog

from anomaly_detector.metrics import ALERTS_DEDUPLICATED, ALERTS_PUBLISHED, ANOMALIES_DETECTED
from anomaly_detector.models.statistical import AnomalyResult

logger = structlog.get_logger(__name__)


class Severity(str, Enum):
    """Alert severity levels in ascending order."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    """Structured anomaly alert ready for downstream consumption."""

    alert_id: str
    timestamp: float
    event_id: str
    feature: str
    value: float
    score: float
    severity: Severity
    anomaly_type: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the alert to a plain dictionary."""
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class _DedupEntry:
    """Internal bookkeeping for deduplication windows."""

    last_seen: float
    count: int = 1


class AlertManager:
    """Manage alert lifecycle: severity, deduplication, and rate limiting.

    Args:
        severity_thresholds: Mapping of ``Severity`` to minimum score.
            Scores are compared in descending severity order so the
            highest matching level wins.
        dedup_window_seconds: Time window in which duplicate alerts for the
            same (feature, anomaly_type) pair are suppressed.
        max_alerts_per_minute: Maximum number of alerts emitted per minute
            across all features (rate limiter).
    """

    def __init__(
        self,
        severity_thresholds: dict[Severity, float] | None = None,
        dedup_window_seconds: int = 300,
        max_alerts_per_minute: int = 100,
    ) -> None:
        self._severity_thresholds: dict[Severity, float] = severity_thresholds or {
            Severity.LOW: 2.0,
            Severity.MEDIUM: 3.0,
            Severity.HIGH: 4.0,
            Severity.CRITICAL: 5.0,
        }
        self._dedup_window = dedup_window_seconds
        self._max_alerts_per_minute = max_alerts_per_minute

        # Dedup state: key -> _DedupEntry
        self._dedup_cache: dict[str, _DedupEntry] = {}
        # Rate limiting state
        self._minute_window_start: float = time.time()
        self._alerts_this_minute: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, result: AnomalyResult, event_id: str) -> Alert | None:
        """Convert an ``AnomalyResult`` to an ``Alert`` if it should fire.

        Returns None when:
        - The result is not anomalous.
        - The alert is deduplicated (same feature/type within the window).
        - The per-minute rate limit has been reached.

        Args:
            result: Detection result from a model.
            event_id: Identifier of the originating trade event.

        Returns:
            An ``Alert`` instance, or None if suppressed.
        """
        if not result.is_anomaly:
            return None

        severity = self._classify_severity(result.score)
        ANOMALIES_DETECTED.labels(
            severity=severity.value,
            detector=result.anomaly_type.value if result.anomaly_type else "unknown",
        ).inc()

        # Deduplication
        dedup_key = self._dedup_key(result.feature, result.anomaly_type)
        if self._is_duplicate(dedup_key):
            ALERTS_DEDUPLICATED.inc()
            logger.debug(
                "alert_deduplicated",
                feature=result.feature,
                anomaly_type=str(result.anomaly_type),
            )
            return None

        # Rate limiting
        if not self._check_rate_limit():
            logger.warning("alert_rate_limited", alerts_this_minute=self._alerts_this_minute)
            return None

        alert = Alert(
            alert_id=self._generate_alert_id(event_id, result.feature),
            timestamp=time.time(),
            event_id=event_id,
            feature=result.feature,
            value=result.value,
            score=result.score,
            severity=severity,
            anomaly_type=result.anomaly_type.value if result.anomaly_type else "unknown",
            details=result.details,
        )

        self._record_dedup(dedup_key)
        self._alerts_this_minute += 1

        ALERTS_PUBLISHED.labels(severity=severity.value).inc()
        logger.info(
            "alert_created",
            alert_id=alert.alert_id,
            severity=severity.value,
            feature=result.feature,
            score=result.score,
        )
        return alert

    def process_batch(
        self, results: list[AnomalyResult], event_id: str
    ) -> list[Alert]:
        """Process a list of anomaly results and return non-suppressed alerts.

        Args:
            results: Detection results from one or more models.
            event_id: Identifier of the originating trade event.

        Returns:
            List of alerts that passed dedup and rate-limit filters.
        """
        alerts: list[Alert] = []
        for result in results:
            alert = self.process(result, event_id)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def cleanup_expired(self) -> int:
        """Remove expired entries from the deduplication cache.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        expired = [
            key
            for key, entry in self._dedup_cache.items()
            if now - entry.last_seen > self._dedup_window
        ]
        for key in expired:
            del self._dedup_cache[key]
        return len(expired)

    # ------------------------------------------------------------------
    # Severity classification
    # ------------------------------------------------------------------

    def _classify_severity(self, score: float) -> Severity:
        """Map a numeric anomaly score to a severity level.

        Checks thresholds from highest to lowest and returns the first
        match.
        """
        ordered = sorted(
            self._severity_thresholds.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for severity, threshold in ordered:
            if score >= threshold:
                return severity
        return Severity.LOW

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_key(feature: str, anomaly_type: Any) -> str:
        """Build a deduplication cache key."""
        return f"{feature}:{anomaly_type}"

    def _is_duplicate(self, key: str) -> bool:
        """Check if an alert with this key was emitted within the window."""
        entry = self._dedup_cache.get(key)
        if entry is None:
            return False
        return (time.time() - entry.last_seen) < self._dedup_window

    def _record_dedup(self, key: str) -> None:
        """Record an alert emission for deduplication tracking."""
        now = time.time()
        existing = self._dedup_cache.get(key)
        if existing is not None:
            existing.last_seen = now
            existing.count += 1
        else:
            self._dedup_cache[key] = _DedupEntry(last_seen=now)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> bool:
        """Return True if we are within the per-minute alert budget."""
        now = time.time()
        if now - self._minute_window_start >= 60.0:
            self._minute_window_start = now
            self._alerts_this_minute = 0
        return self._alerts_this_minute < self._max_alerts_per_minute

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_alert_id(event_id: str, feature: str) -> str:
        """Generate a deterministic alert ID from event and feature."""
        raw = f"{event_id}:{feature}:{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
