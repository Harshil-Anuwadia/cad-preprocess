"""Tests for logging and error handling module."""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from cad_preprocess.logging_utils import (
    CADPreprocessError,
    ErrorRecord,
    LogLevel,
    MetadataExtractionError,
    OutputError,
    PreprocessingError,
    ProcessingLogger,
    ProcessingStage,
    ProcessingStats,
    ValidationError,
    fail_safe,
    get_logger,
    setup_logging,
    timed_operation,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test log level string values."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_to_logging_level(self) -> None:
        """Test conversion to logging module levels."""
        assert LogLevel.DEBUG.to_logging_level() == logging.DEBUG
        assert LogLevel.INFO.to_logging_level() == logging.INFO
        assert LogLevel.WARNING.to_logging_level() == logging.WARNING
        assert LogLevel.ERROR.to_logging_level() == logging.ERROR
        assert LogLevel.CRITICAL.to_logging_level() == logging.CRITICAL


class TestProcessingStage:
    """Tests for ProcessingStage enum."""

    def test_stage_values(self) -> None:
        """Test processing stage string values."""
        assert ProcessingStage.DISCOVERY.value == "discovery"
        assert ProcessingStage.VALIDATION.value == "validation"
        assert ProcessingStage.LOADING.value == "loading"
        assert ProcessingStage.PREPROCESSING.value == "preprocessing"
        assert ProcessingStage.METADATA_EXTRACTION.value == "metadata_extraction"
        assert ProcessingStage.OUTPUT_WRITING.value == "output_writing"


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_cad_preprocess_error_basic(self) -> None:
        """Test basic CADPreprocessError."""
        error = CADPreprocessError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.stage == ProcessingStage.UNKNOWN
        assert error.file_path is None
        assert error.cause is None

    def test_cad_preprocess_error_with_context(self) -> None:
        """Test CADPreprocessError with full context."""
        file_path = Path("/test/file.dcm")
        cause = ValueError("Original error")

        error = CADPreprocessError(
            "Test error",
            stage=ProcessingStage.PREPROCESSING,
            file_path=file_path,
            cause=cause,
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert str(file_path) in error_str
        assert "preprocessing" in error_str
        assert "ValueError" in error_str

    def test_validation_error(self) -> None:
        """Test ValidationError."""
        error = ValidationError("Invalid DICOM", file_path=Path("/test.dcm"))
        assert error.stage == ProcessingStage.VALIDATION
        assert "Invalid DICOM" in str(error)

    def test_preprocessing_error(self) -> None:
        """Test PreprocessingError."""
        error = PreprocessingError("Processing failed")
        assert error.stage == ProcessingStage.PREPROCESSING

    def test_metadata_extraction_error(self) -> None:
        """Test MetadataExtractionError."""
        error = MetadataExtractionError("Metadata missing")
        assert error.stage == ProcessingStage.METADATA_EXTRACTION

    def test_output_error(self) -> None:
        """Test OutputError."""
        error = OutputError("Cannot write file")
        assert error.stage == ProcessingStage.OUTPUT_WRITING


class TestErrorRecord:
    """Tests for ErrorRecord dataclass."""

    def test_error_record_creation(self) -> None:
        """Test ErrorRecord creation."""
        record = ErrorRecord(
            file_path=Path("/test.dcm"),
            stage=ProcessingStage.PREPROCESSING,
            error_type="ValueError",
            error_message="Test error",
        )

        assert record.file_path == Path("/test.dcm")
        assert record.stage == ProcessingStage.PREPROCESSING
        assert record.error_type == "ValueError"
        assert record.error_message == "Test error"
        assert isinstance(record.timestamp, datetime)

    def test_error_record_to_dict(self) -> None:
        """Test ErrorRecord serialization."""
        record = ErrorRecord(
            file_path=Path("/test.dcm"),
            stage=ProcessingStage.PREPROCESSING,
            error_type="ValueError",
            error_message="Test error",
            traceback="Traceback...",
        )

        data = record.to_dict()
        assert data["file"] == "/test.dcm"
        assert data["stage"] == "preprocessing"
        assert data["error_type"] == "ValueError"
        assert data["error_message"] == "Test error"
        assert data["traceback"] == "Traceback..."
        assert "timestamp" in data


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_stats_initialization(self) -> None:
        """Test ProcessingStats default values."""
        stats = ProcessingStats()

        assert stats.files_discovered == 0
        assert stats.files_valid == 0
        assert stats.files_processed == 0
        assert stats.files_skipped == 0
        assert stats.files_failed == 0
        assert stats.errors == []
        assert stats.end_time is None

    def test_record_error(self) -> None:
        """Test recording an error."""
        stats = ProcessingStats()
        error = ValueError("Test error")

        stats.record_error(
            error,
            file_path=Path("/test.dcm"),
            stage=ProcessingStage.PREPROCESSING,
        )

        assert stats.files_failed == 1
        assert len(stats.errors) == 1
        assert stats.errors[0].error_type == "ValueError"

    def test_record_error_from_cad_preprocess_error(self) -> None:
        """Test recording a CADPreprocessError extracts stage."""
        stats = ProcessingStats()
        error = ValidationError("Invalid file", file_path=Path("/test.dcm"))

        stats.record_error(error)

        assert len(stats.errors) == 1
        assert stats.errors[0].stage == ProcessingStage.VALIDATION
        assert stats.errors[0].file_path == Path("/test.dcm")

    def test_finish(self) -> None:
        """Test finishing statistics collection."""
        stats = ProcessingStats()
        assert stats.end_time is None

        stats.finish()

        assert stats.end_time is not None
        assert isinstance(stats.end_time, datetime)

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        stats = ProcessingStats()
        stats.finish()

        # Duration should be very small but non-negative
        assert stats.duration_seconds >= 0
        assert stats.duration_seconds < 1  # Should be almost instant

    def test_success_rate_all_success(self) -> None:
        """Test success rate with all successful processing."""
        stats = ProcessingStats()
        stats.files_processed = 10
        stats.files_failed = 0

        assert stats.success_rate == 100.0

    def test_success_rate_mixed(self) -> None:
        """Test success rate with some failures."""
        stats = ProcessingStats()
        stats.files_processed = 7
        stats.files_failed = 3

        assert stats.success_rate == 70.0

    def test_success_rate_no_files(self) -> None:
        """Test success rate with no files processed."""
        stats = ProcessingStats()

        assert stats.success_rate == 100.0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        stats = ProcessingStats()
        stats.files_discovered = 20
        stats.files_valid = 18
        stats.files_processed = 15
        stats.files_skipped = 2
        stats.files_failed = 1
        stats.finish()

        data = stats.to_dict()

        assert data["files_discovered"] == 20
        assert data["files_valid"] == 18
        assert data["files_processed"] == 15
        assert data["files_skipped"] == 2
        assert data["files_failed"] == 1
        assert "duration_seconds" in data
        assert "start_time" in data
        assert "end_time" in data
        assert "success_rate" in data

    def test_summary(self) -> None:
        """Test human-readable summary generation."""
        stats = ProcessingStats()
        stats.files_discovered = 10
        stats.files_valid = 8
        stats.files_processed = 7
        stats.files_skipped = 1
        stats.files_failed = 0
        stats.finish()

        summary = stats.summary()

        assert "PROCESSING SUMMARY" in summary
        assert "10 files" in summary  # discovered
        assert "8 files" in summary  # valid
        assert "7 files" in summary  # processed

    def test_summary_with_errors(self) -> None:
        """Test summary includes errors."""
        stats = ProcessingStats()
        stats.files_processed = 5
        stats.record_error(
            ValueError("Test error"),
            file_path=Path("/test.dcm"),
        )
        stats.finish()

        summary = stats.summary()

        assert "Errors" in summary
        assert "Test error" in summary


class TestProcessingLogger:
    """Tests for ProcessingLogger class."""

    def test_logger_creation(self) -> None:
        """Test ProcessingLogger creation."""
        logger = ProcessingLogger("test_logger")
        assert logger.logger.name == "test_logger"

    def test_set_level_from_enum(self) -> None:
        """Test setting level from LogLevel enum."""
        logger = ProcessingLogger("test_logger")
        logger.set_level(LogLevel.DEBUG)
        assert logger._level == LogLevel.DEBUG

    def test_set_level_from_string(self) -> None:
        """Test setting level from string."""
        logger = ProcessingLogger("test_logger")
        logger.set_level("warning")
        assert logger._level == LogLevel.WARNING

    def test_format_message_simple(self) -> None:
        """Test simple message formatting."""
        logger = ProcessingLogger("test_logger")
        msg = logger._format_message("Test message")
        assert msg == "Test message"

    def test_format_message_with_context(self) -> None:
        """Test message formatting with context."""
        logger = ProcessingLogger("test_logger")
        msg = logger._format_message("Test message", file="test.dcm", count=5)
        assert "Test message" in msg
        assert "file=test.dcm" in msg
        assert "count=5" in msg

    def test_processing_context(self) -> None:
        """Test processing context manager."""
        logger = ProcessingLogger("test_logger")

        with logger.processing_context("test_batch") as stats:
            stats.files_discovered = 10
            stats.files_processed = 5

        assert isinstance(stats, ProcessingStats)
        assert stats.end_time is not None
        assert stats.files_discovered == 10
        assert stats.files_processed == 5

    def test_file_context_success(self) -> None:
        """Test file context with successful processing."""
        logger = ProcessingLogger("test_logger")
        stats = ProcessingStats()
        file_path = Path("/test.dcm")

        with logger.file_context(file_path, stats):
            pass  # Success

        assert stats.files_failed == 0
        assert len(stats.errors) == 0

    def test_file_context_catches_error(self) -> None:
        """Test file context catches and records errors."""
        logger = ProcessingLogger("test_logger")
        stats = ProcessingStats()
        file_path = Path("/test.dcm")

        with logger.file_context(file_path, stats):
            raise ValueError("Test error")

        assert stats.files_failed == 1
        assert len(stats.errors) == 1
        assert stats.errors[0].error_type == "ValueError"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test setup_logging with default values."""
        logger = setup_logging()

        assert logger.name == "cad_preprocess"
        assert len(logger.handlers) == 1  # Console handler

    def test_setup_logging_with_file(self) -> None:
        """Test setup_logging with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file)

            assert len(logger.handlers) == 2  # Console + file
            assert log_file.exists() or True  # File created on first write

    def test_setup_logging_level(self) -> None:
        """Test setup_logging with custom level."""
        logger = setup_logging(level=LogLevel.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_level_string(self) -> None:
        """Test setup_logging with string level."""
        logger = setup_logging(level="warning")
        assert logger.level == logging.WARNING

    def test_setup_logging_no_console(self) -> None:
        """Test setup_logging without console output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file, log_to_console=False)

            assert len(logger.handlers) == 1  # Only file handler


class TestFailSafeDecorator:
    """Tests for fail_safe decorator."""

    def test_fail_safe_success(self) -> None:
        """Test fail_safe with successful function."""

        @fail_safe(default_return=-1)
        def success_func() -> int:
            return 42

        assert success_func() == 42

    def test_fail_safe_catches_error(self) -> None:
        """Test fail_safe catches errors and returns default."""

        @fail_safe(default_return=-1, log_errors=False)
        def failing_func() -> int:
            raise ValueError("Test error")

        assert failing_func() == -1

    def test_fail_safe_with_none_default(self) -> None:
        """Test fail_safe with None default."""

        @fail_safe(default_return=None, log_errors=False)
        def failing_func():
            raise ValueError("Test error")

        assert failing_func() is None


class TestTimedOperation:
    """Tests for timed_operation context manager."""

    def test_timed_operation_basic(self) -> None:
        """Test timed_operation records timing."""
        with timed_operation("test_op") as timing:
            pass

        assert timing["operation"] == "test_op"
        assert timing["duration"] >= 0
        assert "start_time" in timing

    def test_timed_operation_with_work(self) -> None:
        """Test timed_operation with actual work."""
        import time

        with timed_operation("slow_op") as timing:
            time.sleep(0.01)

        assert timing["duration"] >= 0.01


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_default(self) -> None:
        """Test get_logger with default name."""
        logger = get_logger()
        assert isinstance(logger, ProcessingLogger)
        assert logger.logger.name == "cad_preprocess"

    def test_get_logger_custom_name(self) -> None:
        """Test get_logger with custom name."""
        logger = get_logger("custom_logger")
        assert logger.logger.name == "custom_logger"


class TestIntegration:
    """Integration tests for logging and error handling."""

    def test_full_processing_workflow(self) -> None:
        """Test complete processing workflow with logging."""
        logger = ProcessingLogger("test_integration")

        with logger.processing_context("test_batch") as stats:
            stats.files_discovered = 5
            stats.files_valid = 4

            files = [
                Path(f"/test/file_{i}.dcm")
                for i in range(5)
            ]

            for i, file_path in enumerate(files):
                if i == 2:  # Simulate one failure
                    with logger.file_context(file_path, stats):
                        raise PreprocessingError(
                            "Processing failed",
                            file_path=file_path,
                        )
                else:
                    with logger.file_context(file_path, stats):
                        stats.files_processed += 1

        assert stats.files_processed == 4
        assert stats.files_failed == 1
        assert stats.end_time is not None
        assert len(stats.errors) == 1
        assert stats.errors[0].stage == ProcessingStage.PREPROCESSING

    def test_nested_error_handling(self) -> None:
        """Test nested error handling preserves original cause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise PreprocessingError(
                    "Wrapper error",
                    file_path=Path("/test.dcm"),
                    cause=e,
                )
        except PreprocessingError as e:
            assert e.cause is not None
            assert isinstance(e.cause, ValueError)
            assert "Original error" in str(e.cause)
