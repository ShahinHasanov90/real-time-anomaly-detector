"""Prometheus metrics instrumentation for the anomaly detection pipeline."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

# ---------------------------------------------------------------------------
# Application info
# ---------------------------------------------------------------------------
APP_INFO = Info(
    "anomaly_detector",
    "Real-time anomaly detection service metadata",
)

# ---------------------------------------------------------------------------
# Throughput counters
# ---------------------------------------------------------------------------
EVENTS_CONSUMED = Counter(
    "anomaly_detector_events_consumed_total",
    "Total number of trade events consumed from Kafka",
    ["topic"],
)

EVENTS_PROCESSED = Counter(
    "anomaly_detector_events_processed_total",
    "Total number of events successfully processed through the pipeline",
)

ANOMALIES_DETECTED = Counter(
    "anomaly_detector_anomalies_detected_total",
    "Total number of anomalies detected",
    ["severity", "detector"],
)

ALERTS_PUBLISHED = Counter(
    "anomaly_detector_alerts_published_total",
    "Total number of alerts published to the output topic",
    ["severity"],
)

ALERTS_DEDUPLICATED = Counter(
    "anomaly_detector_alerts_deduplicated_total",
    "Total number of alerts suppressed by deduplication",
)

# ---------------------------------------------------------------------------
# Latency histograms
# ---------------------------------------------------------------------------
PROCESSING_LATENCY = Histogram(
    "anomaly_detector_processing_latency_seconds",
    "End-to-end processing latency per event",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

FEATURE_COMPUTATION_LATENCY = Histogram(
    "anomaly_detector_feature_computation_latency_seconds",
    "Time spent computing sliding window features",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

MODEL_INFERENCE_LATENCY = Histogram(
    "anomaly_detector_model_inference_latency_seconds",
    "Time spent on model inference per event",
    ["model"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

# ---------------------------------------------------------------------------
# Model health gauges
# ---------------------------------------------------------------------------
WINDOW_SIZE_CURRENT = Gauge(
    "anomaly_detector_window_size_current",
    "Current number of samples in the sliding window",
    ["feature"],
)

ISOLATION_FOREST_SAMPLES = Gauge(
    "anomaly_detector_isolation_forest_training_samples",
    "Number of samples used in the last Isolation Forest training",
)

ISOLATION_FOREST_RETRAINS = Counter(
    "anomaly_detector_isolation_forest_retrains_total",
    "Total number of Isolation Forest retraining cycles",
)

# ---------------------------------------------------------------------------
# Error tracking
# ---------------------------------------------------------------------------
PROCESSING_ERRORS = Counter(
    "anomaly_detector_processing_errors_total",
    "Total number of processing errors",
    ["error_type"],
)

KAFKA_CONSUMER_LAG = Gauge(
    "anomaly_detector_kafka_consumer_lag",
    "Estimated consumer lag in number of messages",
    ["partition"],
)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: TCP port to expose metrics on.
    """
    APP_INFO.info(
        {
            "version": "0.1.0",
            "service": "real-time-anomaly-detector",
        }
    )
    start_http_server(port)
