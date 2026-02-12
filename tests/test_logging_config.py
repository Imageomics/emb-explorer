"""Tests for shared/utils/logging_config.py."""

import logging
import os

from shared.utils.logging_config import configure_logging, get_logger


class TestGetLogger:
    def test_returns_logger_with_correct_name(self, reset_logging):
        logger = get_logger("my.module")
        assert logger.name == "my.module"

    def test_returns_logger_instance(self, reset_logging):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestConfigureLogging:
    def test_adds_console_handler(self, reset_logging):
        configure_logging()
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1

    def test_idempotent(self, reset_logging):
        configure_logging()
        handler_count = len(logging.getLogger().handlers)
        configure_logging()
        assert len(logging.getLogger().handlers) == handler_count

    def test_file_handler_created(self, reset_logging, tmp_path):
        import shared.utils.logging_config as log_mod
        original_dir = log_mod._LOG_DIR
        log_mod._LOG_DIR = str(tmp_path)
        try:
            configure_logging(log_to_file=True)
            root = logging.getLogger()
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            assert os.path.exists(os.path.join(str(tmp_path), "emb_explorer.log"))
        finally:
            log_mod._LOG_DIR = original_dir
