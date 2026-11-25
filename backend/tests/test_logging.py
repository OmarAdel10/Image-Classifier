"""Tests for logging configuration."""

import logging
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure backend/src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cnn_classifier.logging_config import configure_logging, get_logger


class TestLoggingConfig:
    """Tests for the logging configuration."""

    def setup_method(self) -> None:
        """Reset logging before each test."""
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        # Also clear our test logger
        test_logger = logging.getLogger("cnn_classifier.test")
        for handler in test_logger.handlers[:]:
            test_logger.removeHandler(handler)

    def test_configure_logging_default_level(self) -> None:
        """Test default log level is INFO."""
        with patch.dict(os.environ, {}, clear=True):
            configure_logging()
            root_logger = logging.getLogger()
            assert root_logger.level == logging.INFO

    def test_configure_logging_custom_level(self) -> None:
        """Test custom log level from environment."""
        with patch.dict(os.environ, {"CNN_LOG_LEVEL": "DEBUG"}):
            configure_logging()
            root_logger = logging.getLogger()
            assert root_logger.level == logging.DEBUG

    def test_configure_logging_invalid_level_fallbacks(self) -> None:
        """Test invalid log level falls back to INFO."""
        with patch.dict(os.environ, {"CNN_LOG_LEVEL": "INVALID"}):
            configure_logging()
            root_logger = logging.getLogger()
            # Should default to INFO since INVALID isn't a valid level
            # Note: logging.getLevelName returns the string for invalid levels
            # but setLevel will raise; our config should handle this gracefully
            assert root_logger.level in (logging.INFO, logging.INFO)

    def test_get_logger_returns_logger(self) -> None:
        """Test get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_log_output_format(self) -> None:
        """Test log output has expected format with request_id."""
        from cnn_classifier.logging_config import _ConsoleFormatter

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_ConsoleFormatter(use_json=False))
        logger = logging.getLogger("test.format")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        logger.info("test message", extra={"request_id": "abc123"})
        output = stream.getvalue()
        assert "test message" in output
        assert "[abc123]" in output

    def test_uvicorn_loggers_configured(self) -> None:
        """Test uvicorn loggers are configured and don't propagate."""
        with patch.dict(os.environ, {}, clear=True):
            configure_logging()
            for name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
                logger = logging.getLogger(name)
                assert not logger.propagate
                assert len(logger.handlers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])