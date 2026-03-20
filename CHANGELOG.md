# Changelog

## [1.1.0] - 2026-02-20
### Added
- Online Isolation Forest with periodic incremental retraining
- Prometheus metrics for latency, throughput, and model health
- Alert deduplication with configurable time windows

### Changed
- Optimized sliding window computation for 3x throughput improvement

## [1.0.0] - 2025-11-01
### Added
- Initial release: Kafka consumer with Z-score and IQR detection
- Sliding window feature computation
- Severity-tiered alerting (LOW/MEDIUM/HIGH/CRITICAL)
- Docker Compose deployment stack
