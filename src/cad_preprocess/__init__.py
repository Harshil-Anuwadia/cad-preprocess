"""
CAD Preprocess - Reusable preprocessing software for medical DICOM images.

This package provides standardized preprocessing for DICOM images to support
CAD (Computer-Aided Detection/Diagnosis) systems. The preprocessing logic
is written once and can be reused across training, inference, and UI workflows.
"""

__version__ = "0.1.0"
__author__ = "CAD Preprocess Team"

from cad_preprocess.config import (
    Config,
    InputSettings,
    LoggingSettings,
    MetadataSettings,
    OutputSettings,
    PreprocessingSettings,
    ResizingSettings,
    WindowingSettings,
    create_config_template,
    load_config,
)
from cad_preprocess.input_handler import InputHandler, discover_dicom_files, validate_dicom_file
from cad_preprocess.integration import (
    BatchResult,
    CADPreprocessor,
    PreprocessResult,
    ProcessingManifest,
    get_config_hash,
    preprocess,
    verify_preprocessing_consistency,
)
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
from cad_preprocess.metadata_extractor import (
    ExtractionResult,
    MetadataExtractor,
    MetadataProfile,
)
from cad_preprocess.output_writer import (
    NamingPolicy,
    OutputConfig,
    OutputWriter,
    OverwritePolicy,
    WriteResult,
)
from cad_preprocess.preprocessing_engine import (
    InterpolationMethod,
    NormalizationMethod,
    PreprocessingConfig,
    PreprocessingEngine,
    PreprocessingResult,
    ResizingConfig,
    WindowingConfig,
    WindowingStrategy,
)

__all__ = [
    "__version__",
    # Configuration
    "Config",
    "load_config",
    "create_config_template",
    "PreprocessingSettings",
    "WindowingSettings",
    "ResizingSettings",
    "MetadataSettings",
    "OutputSettings",
    "InputSettings",
    "LoggingSettings",
    # Input handler
    "InputHandler",
    "discover_dicom_files",
    "validate_dicom_file",
    # Logging and error handling
    "setup_logging",
    "get_logger",
    "ProcessingLogger",
    "ProcessingStats",
    "ProcessingStage",
    "LogLevel",
    "ErrorRecord",
    "CADPreprocessError",
    "ValidationError",
    "PreprocessingError",
    "MetadataExtractionError",
    "OutputError",
    "fail_safe",
    "timed_operation",
    # Integration (high-level API)
    "preprocess",
    "CADPreprocessor",
    "PreprocessResult",
    "BatchResult",
    "ProcessingManifest",
    "get_config_hash",
    "verify_preprocessing_consistency",
    # Preprocessing engine
    "PreprocessingEngine",
    "PreprocessingConfig",
    "PreprocessingResult",
    "WindowingConfig",
    "WindowingStrategy",
    "ResizingConfig",
    "NormalizationMethod",
    "InterpolationMethod",
    # Metadata extractor
    "MetadataExtractor",
    "MetadataProfile",
    "ExtractionResult",
    # Output writer
    "OutputWriter",
    "OutputConfig",
    "WriteResult",
    "NamingPolicy",
    "OverwritePolicy",
]
