"""Kafka consumer for ingesting trade events and orchestrating detection.

This module is the main entry point for the anomaly detection pipeline.
It consumes trade events from a Kafka topic, computes sliding-window
features, runs statistical and Isolation Forest detection, and publishes
alerts via the ``AlertProducer``.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import Any

import structlog
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from anomaly_detector.alerting.manager import AlertManager, Severity
from anomaly_detector.config import DetectorConfig, load_config
from anomaly_detector.features.windows import FeatureWindowManager
from anomaly_detector.metrics import (
    EVENTS_CONSUMED,
    EVENTS_PROCESSED,
    FEATURE_COMPUTATION_LATENCY,
    PROCESSING_ERRORS,
    PROCESSING_LATENCY,
    WINDOW_SIZE_CURRENT,
    start_metrics_server,
)
from anomaly_detector.models.isolation_forest_online import OnlineIsolationForest
from anomaly_detector.models.statistical import StatisticalDetector
from anomaly_detector.stream.producer import AlertProducer

logger = structlog.get_logger(__name__)


class TradeEventConsumer:
    """Kafka consumer that drives the real-time anomaly detection pipeline.

    Lifecycle:
        1. Consume a JSON trade event from Kafka.
        2. Push feature values into sliding windows.
        3. Compute rolling statistics and build a feature vector.
        4. Run the statistical detector (Z-score / IQR).
        5. Run the online Isolation Forest.
        6. Merge results, create alerts, and publish them.

    Args:
        config: Fully-resolved detector configuration.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._running = False

        # Kafka consumer
        self._consumer = KafkaConsumer(
            config.kafka.input_topic,
            bootstrap_servers=config.kafka.bootstrap_servers,
            group_id=config.kafka.consumer_group,
            auto_offset_reset=config.kafka.auto_offset_reset,
            max_poll_records=config.kafka.max_poll_records,
            session_timeout_ms=config.kafka.session_timeout_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            enable_auto_commit=True,
        )

        # Alert producer
        self._producer = AlertProducer(
            bootstrap_servers=config.kafka.bootstrap_servers,
            topic=config.kafka.output_topic,
        )

        # Feature engine
        self._feature_manager = FeatureWindowManager(
            feature_names=config.detection.features,
            window_size=config.detection.window_size,
        )

        # Detection models
        self._statistical = StatisticalDetector(
            zscore_threshold=config.detection.zscore_threshold,
            iqr_multiplier=config.detection.iqr_multiplier,
            min_samples=config.detection.min_samples,
        )
        self._isolation_forest = OnlineIsolationForest(
            n_estimators=config.detection.n_estimators,
            contamination=config.detection.contamination,
            buffer_size=config.detection.window_size,
            retrain_interval=config.detection.retrain_interval,
        )

        # Alert management
        severity_thresholds = {
            Severity.LOW: config.alerting.severity_thresholds.low,
            Severity.MEDIUM: config.alerting.severity_thresholds.medium,
            Severity.HIGH: config.alerting.severity_thresholds.high,
            Severity.CRITICAL: config.alerting.severity_thresholds.critical,
        }
        self._alert_manager = AlertManager(
            severity_thresholds=severity_thresholds,
            dedup_window_seconds=config.alerting.dedup_window_seconds,
            max_alerts_per_minute=config.alerting.max_alerts_per_minute,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start consuming events. Blocks until ``stop()`` is called."""
        self._running = True
        logger.info(
            "consumer_started",
            topic=self._config.kafka.input_topic,
            group=self._config.kafka.consumer_group,
        )

        try:
            while self._running:
                messages = self._consumer.poll(timeout_ms=1000)
                for tp, records in messages.items():
                    for record in records:
                        EVENTS_CONSUMED.labels(topic=tp.topic).inc()
                        try:
                            self._process_event(record.value)
                        except Exception:
                            PROCESSING_ERRORS.labels(error_type="processing").inc()
                            logger.exception("event_processing_error", offset=record.offset)

                # Periodic dedup cache cleanup
                self._alert_manager.cleanup_expired()

        except KafkaError:
            PROCESSING_ERRORS.labels(error_type="kafka").inc()
            logger.exception("kafka_consumer_error")
        finally:
            self._consumer.close()
            self._producer.close()
            logger.info("consumer_stopped")

    def stop(self) -> None:
        """Signal the consumer loop to exit gracefully."""
        self._running = False

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_event(self, event: dict[str, Any]) -> None:
        """Process a single trade event through the full pipeline."""
        start = time.perf_counter()
        event_id = event.get("event_id", "unknown")

        # 1. Derive unit_price if possible
        enriched = dict(event)
        if "declared_value" in enriched and "quantity" in enriched and enriched["quantity"] > 0:
            enriched["unit_price"] = enriched["declared_value"] / enriched["quantity"]

        # 2. Push into sliding windows
        feat_start = time.perf_counter()
        self._feature_manager.push(enriched)
        stats_map = self._feature_manager.compute_all_stats(enriched)
        FEATURE_COMPUTATION_LATENCY.observe(time.perf_counter() - feat_start)

        # Update window size metrics
        for fname in self._config.detection.features:
            try:
                w = self._feature_manager.get_window(fname)
                WINDOW_SIZE_CURRENT.labels(feature=fname).set(w.count)
            except KeyError:
                pass

        # 3. Statistical detection
        values = {f: float(enriched[f]) for f in stats_map if f in enriched}
        stat_results = self._statistical.check_multiple(stats_map, values)

        # 4. Isolation Forest
        feature_vector = self._feature_manager.get_feature_vector(enriched)
        if feature_vector is not None:
            self._isolation_forest.update(feature_vector)
            if_result = self._isolation_forest.predict(feature_vector)
            if if_result.is_anomaly:
                from anomaly_detector.models.statistical import AnomalyResult, AnomalyType

                stat_results.append(
                    AnomalyResult(
                        is_anomaly=True,
                        feature="multi_feature",
                        value=float(if_result.score),
                        anomaly_type=AnomalyType.ZSCORE,
                        score=if_result.score * 5.0,  # scale to severity range
                        details={
                            "detector": "isolation_forest",
                            "raw_score": if_result.raw_score,
                            **if_result.details,
                        },
                    )
                )

        # 5. Create and publish alerts
        alerts = self._alert_manager.process_batch(stat_results, event_id)
        for alert in alerts:
            self._producer.send_alert(alert.to_dict())

        EVENTS_PROCESSED.inc()
        PROCESSING_LATENCY.observe(time.perf_counter() - start)


def main() -> None:
    """CLI entry point for the anomaly detector consumer."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    config = load_config()
    start_metrics_server(port=config.metrics.port)

    consumer = TradeEventConsumer(config)

    def _shutdown(signum: int, frame: Any) -> None:
        logger.info("shutdown_signal_received", signal=signum)
        consumer.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    consumer.run()


if __name__ == "__main__":
    main()
