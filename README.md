# Real-Time Anomaly Detector

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)

A production-grade real-time anomaly detection system for streaming trade and customs data. Combines sliding-window statistics with online machine learning to detect price anomalies, volume spikes, and unusual trading patterns as they happen.

---

## Architecture

```
                      +------------------+
                      |   Data Sources   |
                      | (Trade Events /  |
                      |  Customs Data)   |
                      +--------+---------+
                               |
                               v
                      +--------+---------+
                      |   Apache Kafka   |
                      |  (Input Topic)   |
                      +--------+---------+
                               |
                               v
                  +------------+------------+
                  |     Kafka Consumer      |
                  |  (stream/consumer.py)   |
                  +------------+------------+
                               |
                               v
                  +------------+------------+
                  |    Feature Engine       |
                  |  (features/windows.py)  |
                  |  Rolling Mean, Std,     |
                  |  Percentiles, Ratios    |
                  +------------+------------+
                               |
                               v
                  +------------+-------------+
                  |    Anomaly Models         |
                  | +----------------------+ |
                  | | Statistical Detector | |
                  | | (Z-Score / IQR)      | |
                  | +----------------------+ |
                  | | Online Isolation     | |
                  | | Forest               | |
                  | +----------------------+ |
                  +------------+-------------+
                               |
                               v
                  +------------+------------+
                  |    Alert Manager        |
                  |  Deduplication,         |
                  |  Severity Levels        |
                  +------------+------------+
                               |
                     +---------+---------+
                     |                   |
                     v                   v
            +--------+-------+  +--------+--------+
            | Kafka Output   |  | Prometheus       |
            | (Alert Topic)  |  | Metrics          |
            +----------------+  +-----------------+
```

## Features

- **Streaming Ingestion** -- Kafka-based consumer/producer for high-throughput event processing
- **Sliding Window Statistics** -- Configurable rolling windows for mean, standard deviation, percentiles, and z-scores
- **Statistical Detection** -- Z-score and IQR-based anomaly detection with adaptive thresholds
- **Online Isolation Forest** -- Scikit-learn Isolation Forest with periodic incremental retraining on recent data
- **Alert Management** -- Severity-based alerting (LOW / MEDIUM / HIGH / CRITICAL) with time-window deduplication
- **Prometheus Metrics** -- Built-in instrumentation for latency, throughput, anomaly rates, and model health
- **Structured Logging** -- JSON-formatted logs via structlog for observability pipelines
- **Docker-Ready** -- Full docker-compose stack with Kafka, Redis, and the detector service

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/ShahinHasanov90/real-time-anomaly-detector.git
cd real-time-anomaly-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Docker Setup

```bash
# Start the full stack (Kafka + Redis + Detector)
docker-compose up -d

# View logs
docker-compose logs -f anomaly-detector

# Stop all services
docker-compose down
```

### Sending Test Events

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

event = {
    "event_id": "txn-001",
    "timestamp": "2026-03-19T12:00:00Z",
    "commodity_code": "8471.30",
    "declared_value": 15000.00,
    "quantity": 500,
    "weight_kg": 120.5,
    "origin_country": "CN",
    "destination_country": "US",
}

producer.send("trade-events", value=event)
producer.flush()
```

## Configuration

The detector is configured via `config/detector_config.yaml`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `kafka.bootstrap_servers` | `localhost:9092` | Kafka broker addresses |
| `kafka.input_topic` | `trade-events` | Topic to consume trade events from |
| `kafka.output_topic` | `anomaly-alerts` | Topic to publish alerts to |
| `kafka.consumer_group` | `anomaly-detector` | Consumer group ID |
| `detection.zscore_threshold` | `3.0` | Z-score threshold for statistical detection |
| `detection.iqr_multiplier` | `1.5` | IQR multiplier for outlier fencing |
| `detection.window_size` | `1000` | Sliding window size (number of events) |
| `detection.retrain_interval` | `5000` | Events between Isolation Forest retraining |
| `alerting.dedup_window_seconds` | `300` | Deduplication window in seconds |
| `redis.url` | `redis://localhost:6379/0` | Redis connection URL |

Environment variables override YAML configuration with the prefix `ANOMALY_DETECTOR_`.

## Project Structure

```
real-time-anomaly-detector/
    src/anomaly_detector/
        stream/
            consumer.py        # Kafka consumer for trade events
            producer.py        # Alert producer to Kafka
        models/
            statistical.py     # Z-score and IQR detection
            isolation_forest_online.py  # Online Isolation Forest
        features/
            windows.py         # Sliding window feature computation
        alerting/
            manager.py         # Alert management and deduplication
        config.py              # Configuration loading
        metrics.py             # Prometheus instrumentation
    config/
        detector_config.yaml   # Default configuration
    tests/
        test_statistical.py    # Statistical model tests
        test_windows.py        # Sliding window tests
    docker-compose.yml
    Dockerfile
    setup.py
    requirements.txt
```

## License

MIT License. See [LICENSE](LICENSE) for details.
