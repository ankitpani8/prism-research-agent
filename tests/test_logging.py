"""JSON log formatter merges a JSON message into the envelope."""
import json
import logging

from core.logging_config import JsonFormatter


def test_json_formatter_merges_dict_message():
    rec = logging.LogRecord("prism.api", logging.INFO, __file__, 1,
                            json.dumps({"event": "research_start", "trace_id": "abc"}), None, None)
    out = json.loads(JsonFormatter().format(rec))
    assert out["event"] == "research_start" and out["trace_id"] == "abc"
    assert out["level"] == "INFO" and out["logger"] == "prism.api"


def test_json_formatter_plain_message():
    rec = logging.LogRecord("x", logging.WARNING, __file__, 1, "plain text", None, None)
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "plain text" and out["level"] == "WARNING"
