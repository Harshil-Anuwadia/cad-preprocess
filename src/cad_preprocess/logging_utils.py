"""
Logging and Error Handling Module for CAD Preprocess.

This module provides structured logging and fail-safe error handling
for the preprocessing pipeline.

Logging features:
- Configurable log levels (debug, info, warning, error, critical)
- File and console logging
- Structured log messages
- Processing statistics tracking

Error handling strategy:
- Fail-safe: Never crash on single file failure
- Log errors and continue processing
- Return comprehensive summary at end

Logged items:
- Processed files count
- Skipped files count
- Errors with reasons
- Processing time
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_logging_level(self) -> int:
        """Convert to logging module level."""
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


class ProcessingStage(str, Enum):
    """Processing pipeline stages."""

    DISCOVERY = "discovery"
    VALIDATION = "validation"
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    METADATA_EXTRACTION = "metadata_extraction"
    OUTPUT_WRITING = "output_writing"
    UNKNOWN = "unknown"


# Custom exception classes


class CADPreprocessError(Exception):
    """Base exception for CAD Preprocess."""

    def __init__(
        self,
        message: str,
        stage: ProcessingStage = ProcessingStage.UNKNOWN,
        file_path: Optional[Path] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.file_path = file_path
        self.cause = cause

    def __str__(self) -> str:
        parts = [self.message]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.stage != ProcessingStage.UNKNOWN:
            parts.append(f"Stage: {self.stage.value}")
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)


class ValidationError(CADPreprocessError):
    """Error during DICOM validation."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message,
            stage=ProcessingStage.VALIDATION,
            file_path=file_path,
            cause=cause,
        )


class PreprocessingError(CADPreprocessError):
    """Error during image preprocessing."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message,
            stage=ProcessingStage.PREPROCESSING,
            file_path=file_path,
            cause=cause,
        )


class MetadataExtractionError(CADPreprocessError):
    """Error during metadata extraction."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message,
            stage=ProcessingStage.METADATA_EXTRACTION,
            file_path=file_path,
            cause=cause,
        )


class OutputError(CADPreprocessError):
    """Error during output writing."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message,
            stage=ProcessingStage.OUTPUT_WRITING,
            file_path=file_path,
            cause=cause,
        )


@dataclass
class ErrorRecord:
    """Record of a single error."""

    file_path: Optional[Path]
    stage: ProcessingStage
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": str(self.file_path) if self.file_path else None,
            "stage": self.stage.value,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback,
        }


@dataclass
class ProcessingStats:
    """Statistics for a processing run."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    files_discovered: int = 0
    files_valid: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    errors: List[ErrorRecord] = field(default_factory=list)

    def record_error(
        self,
        error: Exception,
        file_path: Optional[Path] = None,
        stage: ProcessingStage = ProcessingStage.UNKNOWN,
        include_traceback: bool = False,
    ) -> None:
        """Record an error."""
        tb = None
        if include_traceback:
            tb = traceback.format_exc()

        # Extract stage from CADPreprocessError if available
        if isinstance(error, CADPreprocessError):
            stage = error.stage
            file_path = error.file_path or file_path

        record = ErrorRecord(
            file_path=file_path,
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=tb,
        )
        self.errors.append(record)
        self.files_failed += 1

    def finish(self) -> None:
        """Mark processing as finished."""
        self.end_time = datetime.now()

    @property
    def duration_seconds(self) -> float:
        """Get processing duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        total = self.files_processed + self.files_failed
        if total == 0:
            return 100.0
        return (self.files_processed / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "files_discovered": self.files_discovered,
            "files_valid": self.files_valid,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "success_rate": round(self.success_rate, 1),
            "errors": [e.to_dict() for e in self.errors],
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "PROCESSING SUMMARY",
            "=" * 50,
            f"Duration:    {self.duration_seconds:.2f} seconds",
            "-" * 50,
            f"Discovered:  {self.files_discovered} files",
            f"Valid:       {self.files_valid} files",
            f"Processed:   {self.files_processed} files",
            f"Skipped:     {self.files_skipped} files",
            f"Failed:      {self.files_failed} files",
            f"Success:     {self.success_rate:.1f}%",
            "=" * 50,
        ]

        if self.errors:
            lines.append("\nErrors (first 10):")
            for error in self.errors[:10]:
                file_str = error.file_path.name if error.file_path else "unknown"
                lines.append(f"  - [{error.stage.value}] {file_str}: {error.error_message}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors")

        return "\n".join(lines)


class ProcessingLogger:
    """
    Structured logger for processing operations.

    Provides consistent logging format and processing statistics tracking.

    Example:
        >>> logger = ProcessingLogger("cad_preprocess")
        >>> with logger.processing_context("batch_001") as stats:
        ...     for file in files:
        ...         with logger.file_context(file, stats):
        ...             process(file)
        ...             stats.files_processed += 1
        >>> print(stats.summary())
    """

    def __init__(
        self,
        name: str = "cad_preprocess",
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        """
        Initialize the processing logger.

        Args:
            name: Logger name.
            level: Default log level.
        """
        self.logger = logging.getLogger(name)
        self._level = level

    def set_level(self, level: Union[LogLevel, str]) -> None:
        """Set log level."""
        if isinstance(level, str):
            level = LogLevel(level)
        self._level = level
        self.logger.setLevel(level.to_logging_level())

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message, **kwargs))

    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format log message with additional context."""
        if not kwargs:
            return message

        context_parts = [f"{k}={v}" for k, v in kwargs.items()]
        return f"{message} | {' '.join(context_parts)}"

    def log_file_start(self, file_path: Path, index: int, total: int) -> None:
        """Log start of file processing."""
        self.info(
            f"Processing [{index}/{total}]: {file_path.name}",
            file=str(file_path),
        )

    def log_file_success(self, file_path: Path) -> None:
        """Log successful file processing."""
        self.debug(f"Successfully processed: {file_path.name}")

    def log_file_skipped(self, file_path: Path, reason: str) -> None:
        """Log skipped file."""
        self.info(f"Skipped: {file_path.name}", reason=reason)

    def log_file_error(
        self,
        file_path: Path,
        error: Exception,
        stage: ProcessingStage = ProcessingStage.UNKNOWN,
    ) -> None:
        """Log file processing error."""
        self.error(
            f"Failed: {file_path.name}",
            stage=stage.value,
            error_type=type(error).__name__,
            error=str(error),
        )

    def log_batch_start(self, total_files: int, batch_id: Optional[str] = None) -> None:
        """Log start of batch processing."""
        self.info(
            f"Starting batch processing",
            total_files=total_files,
            batch_id=batch_id or "default",
        )

    def log_batch_complete(self, stats: ProcessingStats) -> None:
        """Log batch processing completion."""
        self.info(
            "Batch processing complete",
            processed=stats.files_processed,
            skipped=stats.files_skipped,
            failed=stats.files_failed,
            duration=f"{stats.duration_seconds:.2f}s",
        )

    @contextmanager
    def processing_context(self, batch_id: Optional[str] = None):
        """
        Context manager for tracking processing statistics.

        Args:
            batch_id: Optional identifier for the batch.

        Yields:
            ProcessingStats object for tracking.
        """
        stats = ProcessingStats()
        self.debug(f"Starting processing context", batch_id=batch_id)

        try:
            yield stats
        finally:
            stats.finish()
            self.log_batch_complete(stats)

    @contextmanager
    def file_context(
        self,
        file_path: Path,
        stats: ProcessingStats,
        stage: ProcessingStage = ProcessingStage.PREPROCESSING,
    ):
        """
        Context manager for processing a single file with error handling.

        Implements fail-safe behavior: logs errors and continues.

        Args:
            file_path: Path to file being processed.
            stats: ProcessingStats to update.
            stage: Current processing stage.

        Yields:
            None

        Note:
            Exceptions are caught, logged, and recorded in stats.
            Processing continues after errors.
        """
        try:
            yield
        except CADPreprocessError as e:
            self.log_file_error(file_path, e, e.stage)
            stats.record_error(e, file_path, e.stage)
        except Exception as e:
            self.log_file_error(file_path, e, stage)
            stats.record_error(e, file_path, stage, include_traceback=True)


def setup_logging(
    level: Union[LogLevel, str] = LogLevel.INFO,
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Log level (string or LogLevel enum).
        log_file: Optional path to log file.
        log_to_console: Whether to log to console.
        log_format: Custom log format string.
        date_format: Custom date format string.

    Returns:
        Configured logger.
    """
    if isinstance(level, str):
        level = LogLevel(level)

    log_level = level.to_logging_level()

    # Default formats
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger for cad_preprocess
    root_logger = logging.getLogger("cad_preprocess")
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def fail_safe(
    stage: ProcessingStage = ProcessingStage.UNKNOWN,
    default_return: Any = None,
    log_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for fail-safe function execution.

    Catches exceptions, logs them, and returns a default value
    instead of propagating the error.

    Args:
        stage: Processing stage for error classification.
        default_return: Value to return on error.
        log_errors: Whether to log errors.

    Returns:
        Decorated function.

    Example:
        >>> @fail_safe(stage=ProcessingStage.PREPROCESSING, default_return=None)
        ... def process_file(path):
        ...     # processing code
        ...     return result
    """
    logger = logging.getLogger("cad_preprocess")

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                return default_return

        return wrapper  # type: ignore

    return decorator


@contextmanager
def timed_operation(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing operations.

    Args:
        operation_name: Name of the operation for logging.
        logger: Optional logger (uses default if not provided).

    Yields:
        Dict that will contain timing information.

    Example:
        >>> with timed_operation("preprocessing") as timing:
        ...     process_image(image)
        >>> print(f"Took {timing['duration']:.2f}s")
    """
    if logger is None:
        logger = logging.getLogger("cad_preprocess")

    timing: Dict[str, Any] = {
        "operation": operation_name,
        "start_time": time.time(),
        "duration": 0.0,
    }

    logger.debug(f"Starting: {operation_name}")

    try:
        yield timing
    finally:
        timing["duration"] = time.time() - timing["start_time"]
        logger.debug(f"Completed: {operation_name} ({timing['duration']:.2f}s)")


def get_logger(name: str = "cad_preprocess") -> ProcessingLogger:
    """
    Get a ProcessingLogger instance.

    Args:
        name: Logger name.

    Returns:
        ProcessingLogger instance.
    """
    return ProcessingLogger(name)
