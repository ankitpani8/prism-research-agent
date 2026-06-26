"""Structured JSON logging (req 6, production thinking).

One JSON object per line on stdout — greppable, ship-to-Cloud-Logging friendly.
A logged message that is itself JSON is merged into the envelope, so callers do:

    log.info(json.dumps({"event": "research_start", "trace_id": tid}))

and get {"ts","level","logger","event","trace_id"} on one line.
"""
from __future__ import annotations

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
               "level": record.levelname, "logger": record.name}
        msg = record.getMessage()
        try:
            merged = json.loads(msg)
            if isinstance(merged, dict):
                out.update(merged)
            else:
                out["msg"] = msg
        except (ValueError, TypeError):
            out["msg"] = msg
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
