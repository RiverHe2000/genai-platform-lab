"""JSON-lines application logging (the audit trail is separate and always on)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
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
