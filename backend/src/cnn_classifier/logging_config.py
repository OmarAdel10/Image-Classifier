"""Logging configuration for the CNN classifier backend."""

import logging
import logging.config
import os
from typing import Any


def configure_logging() -> None:
    """Configure structured logging for the application.

    Reads configuration from environment variables:
    - CNN_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.
    - CNN_LOG_JSON: If "1" or "true", use JSON format. Default: false.
    """
    log_level_str = os.getenv("CNN_LOG_LEVEL", "INFO").upper()
    log_json = os.getenv("CNN_LOG_JSON", "false").lower() in ("1", "true", "yes")

    # Validate log level, fallback to INFO if invalid
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    log_level = log_level_str if log_level_str in valid_levels else "INFO"

    # Base configuration
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "()": "cnn_classifier.logging_config._ConsoleFormatter",
                "use_json": log_json,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "slowapi": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(config)


class _ConsoleFormatter(logging.Formatter):
    """Formatter with optional JSON output and request ID support."""

    def __init__(self, use_json: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_json = use_json
        self._json_fields = [
            "timestamp",
            "level",
            "logger",
            "request_id",
            "message",
        ]

    def format(self, record: logging.LogRecord) -> str:
        # Extract request_id if present
        request_id = getattr(record, "request_id", None)

        if self.use_json:
            import json
            from datetime import datetime

            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if request_id:
                log_data["request_id"] = request_id
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in {
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "name",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "request_id",
                }:
                    log_data[key] = value
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)

        # Plain text format with colors
        color_map = {
            "DEBUG": "\033[36m",    # cyan
            "INFO": "\033[32m",     # green
            "WARNING": "\033[33m",  # yellow
            "ERROR": "\033[31m",    # red
            "CRITICAL": "\033[35m", # magenta
        }
        reset = "\033[0m"
        level_color = color_map.get(record.levelname, "")

        timestamp = self.formatTime(record, "%H:%M:%S")
        request_id_str = f" [{request_id}]" if request_id else ""

        return (
            f"{timestamp} {level_color}{record.levelname:<8}{reset} "
            f"{record.name}{request_id_str} - {record.getMessage()}"
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
