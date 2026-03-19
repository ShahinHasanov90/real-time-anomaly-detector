"""Configuration loading and validation for the anomaly detector."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class KafkaConfig:
    """Kafka connection and topic configuration."""

    bootstrap_servers: str = "localhost:9092"
    input_topic: str = "trade-events"
    output_topic: str = "anomaly-alerts"
    consumer_group: str = "anomaly-detector"
    auto_offset_reset: str = "latest"
    max_poll_records: int = 500
    session_timeout_ms: int = 30000


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection configuration."""

    url: str = "redis://localhost:6379/0"
    key_prefix: str = "anomaly:"
    ttl_seconds: int = 86400


@dataclass(frozen=True)
class DetectionConfig:
    """Detection model parameters."""

    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    window_size: int = 1000
    min_samples: int = 30
    retrain_interval: int = 5000
    contamination: float = 0.05
    n_estimators: int = 100
    features: list[str] = field(
        default_factory=lambda: ["declared_value", "quantity", "weight_kg", "unit_price"]
    )


@dataclass(frozen=True)
class SeverityThresholds:
    """Z-score thresholds for alert severity classification."""

    low: float = 2.0
    medium: float = 3.0
    high: float = 4.0
    critical: float = 5.0


@dataclass(frozen=True)
class AlertingConfig:
    """Alert management configuration."""

    dedup_window_seconds: int = 300
    severity_thresholds: SeverityThresholds = field(default_factory=SeverityThresholds)
    max_alerts_per_minute: int = 100


@dataclass(frozen=True)
class MetricsConfig:
    """Prometheus metrics configuration."""

    port: int = 8000
    path: str = "/metrics"


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"


@dataclass(frozen=True)
class DetectorConfig:
    """Top-level configuration for the anomaly detector."""

    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides with ANOMALY_DETECTOR_ prefix.

    Environment variables use double underscores to represent nesting.
    Example: ANOMALY_DETECTOR_KAFKA__BOOTSTRAP_SERVERS overrides
    config_dict["kafka"]["bootstrap_servers"].

    Args:
        config_dict: Base configuration dictionary to override.

    Returns:
        Configuration dictionary with environment overrides applied.
    """
    prefix = "ANOMALY_DETECTOR_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        target = config_dict
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config_dict


def _build_severity_thresholds(raw: dict[str, Any] | None) -> SeverityThresholds:
    """Build SeverityThresholds from a raw dictionary."""
    if raw is None:
        return SeverityThresholds()
    return SeverityThresholds(
        low=float(raw.get("low", 2.0)),
        medium=float(raw.get("medium", 3.0)),
        high=float(raw.get("high", 4.0)),
        critical=float(raw.get("critical", 5.0)),
    )


def load_config(config_path: str | Path | None = None) -> DetectorConfig:
    """Load configuration from a YAML file with environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file. If None, uses
            the default path ``config/detector_config.yaml``.

    Returns:
        A fully resolved ``DetectorConfig`` instance.
    """
    if config_path is None:
        config_path = Path("config/detector_config.yaml")
    else:
        config_path = Path(config_path)

    config_dict: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config_dict = yaml.safe_load(fh) or {}

    config_dict = _apply_env_overrides(config_dict)

    kafka_raw = config_dict.get("kafka", {})
    redis_raw = config_dict.get("redis", {})
    detection_raw = config_dict.get("detection", {})
    alerting_raw = config_dict.get("alerting", {})
    metrics_raw = config_dict.get("metrics", {})
    logging_raw = config_dict.get("logging", {})

    return DetectorConfig(
        kafka=KafkaConfig(
            bootstrap_servers=kafka_raw.get("bootstrap_servers", "localhost:9092"),
            input_topic=kafka_raw.get("input_topic", "trade-events"),
            output_topic=kafka_raw.get("output_topic", "anomaly-alerts"),
            consumer_group=kafka_raw.get("consumer_group", "anomaly-detector"),
            auto_offset_reset=kafka_raw.get("auto_offset_reset", "latest"),
            max_poll_records=int(kafka_raw.get("max_poll_records", 500)),
            session_timeout_ms=int(kafka_raw.get("session_timeout_ms", 30000)),
        ),
        redis=RedisConfig(
            url=redis_raw.get("url", "redis://localhost:6379/0"),
            key_prefix=redis_raw.get("key_prefix", "anomaly:"),
            ttl_seconds=int(redis_raw.get("ttl_seconds", 86400)),
        ),
        detection=DetectionConfig(
            zscore_threshold=float(detection_raw.get("zscore_threshold", 3.0)),
            iqr_multiplier=float(detection_raw.get("iqr_multiplier", 1.5)),
            window_size=int(detection_raw.get("window_size", 1000)),
            min_samples=int(detection_raw.get("min_samples", 30)),
            retrain_interval=int(detection_raw.get("retrain_interval", 5000)),
            contamination=float(detection_raw.get("contamination", 0.05)),
            n_estimators=int(detection_raw.get("n_estimators", 100)),
            features=detection_raw.get(
                "features", ["declared_value", "quantity", "weight_kg", "unit_price"]
            ),
        ),
        alerting=AlertingConfig(
            dedup_window_seconds=int(alerting_raw.get("dedup_window_seconds", 300)),
            severity_thresholds=_build_severity_thresholds(
                alerting_raw.get("severity_thresholds")
            ),
            max_alerts_per_minute=int(alerting_raw.get("max_alerts_per_minute", 100)),
        ),
        metrics=MetricsConfig(
            port=int(metrics_raw.get("port", 8000)),
            path=metrics_raw.get("path", "/metrics"),
        ),
        logging=LoggingConfig(
            level=logging_raw.get("level", "INFO"),
            format=logging_raw.get("format", "json"),
        ),
    )
