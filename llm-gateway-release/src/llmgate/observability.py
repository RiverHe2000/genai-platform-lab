"""Prometheus metrics, JSON-lines logs and the request-id context."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

REGISTRY = CollectorRegistry()
REQUESTS = Counter(
    "llmgate_requests_total",
    "Requests by route, backend and status",
    ["route", "backend", "status"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "llmgate_request_latency_seconds",
    "End-to-end request latency",
    ["backend"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
    registry=REGISTRY,
)
TTFT = Histogram(
    "llmgate_ttft_seconds",
    "Time to first streamed token",
    ["backend"],
    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=REGISTRY,
)
TOKENS = Counter(
    "llmgate_tokens_total", "Tokens by backend and kind", ["backend", "kind"], registry=REGISTRY
)
BREAKER = Gauge(
    "llmgate_breaker_open", "1 when the breaker is open", ["backend"], registry=REGISTRY
)
GUARDRAIL_EVENTS = Counter(
    "llmgate_guardrail_events_total", "Guardrail events", ["rail", "action"], registry=REGISTRY
)
REJECTED = Counter(
    "llmgate_rejected_total", "Requests rejected before a backend", ["reason"], registry=REGISTRY
)
FALLBACKS = Counter(
    "llmgate_fallbacks_total", "Fallback hops", ["from_backend", "to_backend"], registry=REGISTRY
)
SHADOW = Counter(
    "llmgate_shadow_total", "Shadow requests", ["backend", "outcome"], registry=REGISTRY
)


def render_metrics() -> bytes:
    return bytes(generate_latest(REGISTRY))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "event_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, json_lines: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if json_lines else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"event_fields": {"event": event, **fields}})
