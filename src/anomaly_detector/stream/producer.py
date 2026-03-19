"""Kafka producer for publishing anomaly alerts to downstream consumers.

Wraps ``kafka-python``'s ``KafkaProducer`` with JSON serialization,
structured logging, and delivery-callback error handling.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from kafka import KafkaProducer
from kafka.errors import KafkaError

from anomaly_detector.metrics import PROCESSING_ERRORS

logger = structlog.get_logger(__name__)


class AlertProducer:
    """Publish structured anomaly alerts to a Kafka topic.

    Args:
        bootstrap_servers: Kafka broker address(es).
        topic: Destination topic for alert messages.
        acks: Producer acknowledgment level (default ``"all"``).
        retries: Number of send retries on transient failures.
        linger_ms: Batching linger time in milliseconds.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "anomaly-alerts",
        acks: str = "all",
        retries: int = 3,
        linger_ms: int = 10,
    ) -> None:
        self._topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            acks=acks,
            retries=retries,
            linger_ms=linger_ms,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )

    def send_alert(self, alert: dict[str, Any], key: str | None = None) -> None:
        """Send a single alert message to Kafka.

        The message key defaults to the ``alert_id`` field when not
        provided explicitly, ensuring that alerts for the same event
        land on the same partition.

        Args:
            alert: Serializable alert dictionary.
            key: Optional partition key.
        """
        if key is None:
            key = alert.get("alert_id", "")

        try:
            future = self._producer.send(self._topic, value=alert, key=key)
            future.add_callback(self._on_success)
            future.add_errback(self._on_error)
        except KafkaError:
            PROCESSING_ERRORS.labels(error_type="producer_send").inc()
            logger.exception("alert_send_failed", alert_id=alert.get("alert_id"))

    def flush(self, timeout: float | None = None) -> None:
        """Block until all buffered messages have been delivered.

        Args:
            timeout: Maximum seconds to wait.  ``None`` means wait
                indefinitely.
        """
        self._producer.flush(timeout=timeout)

    def close(self) -> None:
        """Flush pending messages and release producer resources."""
        try:
            self._producer.flush(timeout=10)
            self._producer.close(timeout=10)
            logger.info("alert_producer_closed")
        except Exception:
            logger.exception("alert_producer_close_error")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _on_success(metadata: Any) -> None:
        """Log successful delivery at debug level."""
        logger.debug(
            "alert_delivered",
            topic=metadata.topic,
            partition=metadata.partition,
            offset=metadata.offset,
        )

    @staticmethod
    def _on_error(exc: Exception) -> None:
        """Log delivery failure and increment error counter."""
        PROCESSING_ERRORS.labels(error_type="producer_delivery").inc()
        logger.error("alert_delivery_failed", error=str(exc))
