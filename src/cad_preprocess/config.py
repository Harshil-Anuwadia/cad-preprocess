"""
Configuration System Module for CAD Preprocess.

This module provides a YAML-based configuration system for managing
all preprocessing parameters. It ensures safe, reproducible preprocessing
by centralizing all settings.

Configurable items:
- Windowing strategy and values
- Output image size
- Normalization method
- Metadata extraction profile
- Logging level
- Output policies

Default behavior is designed to be safe and reproducible.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from cad_preprocess.metadata_extractor import MetadataProfile
from cad_preprocess.output_writer import NamingPolicy, OverwritePolicy
from cad_preprocess.preprocessing_engine import (
    InterpolationMethod,
    NormalizationMethod,
    PreprocessingConfig,
    ResizingConfig,
    WindowingConfig,
    WindowingStrategy,
)

# Configure module logger
logger = logging.getLogger(__name__)


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


@dataclass
class WindowingSettings:
    """Windowing configuration settings."""

    strategy: str = "use_dicom_window"
    window_center: Optional[float] = None
    window_width: Optional[float] = None

    def to_windowing_config(self) -> WindowingConfig:
        """Convert to WindowingConfig."""
        return WindowingConfig(
            strategy=WindowingStrategy(self.strategy),
            window_center=self.window_center,
            window_width=self.window_width,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "window_center": self.window_center,
            "window_width": self.window_width,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowingSettings":
        """Create from dictionary."""
        return cls(
            strategy=data.get("strategy", "use_dicom_window"),
            window_center=data.get("window_center"),
            window_width=data.get("window_width"),
        )


@dataclass
class ResizingSettings:
    """Image resizing configuration settings."""

    enabled: bool = True
    target_height: int = 1024
    target_width: int = 1024
    keep_aspect_ratio: bool = True
    interpolation: str = "bilinear"
    padding_value: int = 0

    def to_resizing_config(self) -> ResizingConfig:
        """Convert to ResizingConfig."""
        return ResizingConfig(
            enabled=self.enabled,
            target_size=(self.target_height, self.target_width),
            keep_aspect_ratio=self.keep_aspect_ratio,
            interpolation=InterpolationMethod(self.interpolation),
            padding_value=self.padding_value,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "keep_aspect_ratio": self.keep_aspect_ratio,
            "interpolation": self.interpolation,
            "padding_value": self.padding_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResizingSettings":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            target_height=data.get("target_height", 1024),
            target_width=data.get("target_width", 1024),
            keep_aspect_ratio=data.get("keep_aspect_ratio", True),
            interpolation=data.get("interpolation", "bilinear"),
            padding_value=data.get("padding_value", 0),
        )


@dataclass
class PreprocessingSettings:
    """Complete preprocessing configuration settings."""

    windowing: WindowingSettings = field(default_factory=WindowingSettings)
    normalization: str = "min_max"
    resizing: ResizingSettings = field(default_factory=ResizingSettings)
    output_dtype: str = "uint8"

    def to_preprocessing_config(self) -> PreprocessingConfig:
        """Convert to PreprocessingConfig."""
        return PreprocessingConfig(
            windowing=self.windowing.to_windowing_config(),
            normalization=NormalizationMethod(self.normalization),
            resizing=self.resizing.to_resizing_config(),
            output_dtype=self.output_dtype,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "windowing": self.windowing.to_dict(),
            "normalization": self.normalization,
            "resizing": self.resizing.to_dict(),
            "output_dtype": self.output_dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingSettings":
        """Create from dictionary."""
        return cls(
            windowing=WindowingSettings.from_dict(data.get("windowing", {})),
            normalization=data.get("normalization", "min_max"),
            resizing=ResizingSettings.from_dict(data.get("resizing", {})),
            output_dtype=data.get("output_dtype", "uint8"),
        )


@dataclass
class MetadataSettings:
    """Metadata extraction configuration settings."""

    profiles: List[str] = field(default_factory=lambda: ["minimal"])
    additional_fields: List[str] = field(default_factory=list)
    include_all_profiles: bool = False

    def get_profiles(self) -> List[MetadataProfile]:
        """Get list of MetadataProfile enums."""
        return [MetadataProfile(p) for p in self.profiles]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profiles": self.profiles,
            "additional_fields": self.additional_fields,
            "include_all_profiles": self.include_all_profiles,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataSettings":
        """Create from dictionary."""
        return cls(
            profiles=data.get("profiles", ["minimal"]),
            additional_fields=data.get("additional_fields", []),
            include_all_profiles=data.get("include_all_profiles", False),
        )


@dataclass
class OutputSettings:
    """Output configuration settings."""

    naming_policy: str = "sop_instance_uid"
    overwrite_policy: str = "skip"
    image_format: str = "png"
    images_subdir: str = "images"
    metadata_subdir: str = "metadata"
    logs_subdir: str = "logs"

    def get_naming_policy(self) -> NamingPolicy:
        """Get NamingPolicy enum."""
        return NamingPolicy(self.naming_policy)

    def get_overwrite_policy(self) -> OverwritePolicy:
        """Get OverwritePolicy enum."""
        return OverwritePolicy(self.overwrite_policy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "naming_policy": self.naming_policy,
            "overwrite_policy": self.overwrite_policy,
            "image_format": self.image_format,
            "images_subdir": self.images_subdir,
            "metadata_subdir": self.metadata_subdir,
            "logs_subdir": self.logs_subdir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputSettings":
        """Create from dictionary."""
        return cls(
            naming_policy=data.get("naming_policy", "sop_instance_uid"),
            overwrite_policy=data.get("overwrite_policy", "skip"),
            image_format=data.get("image_format", "png"),
            images_subdir=data.get("images_subdir", "images"),
            metadata_subdir=data.get("metadata_subdir", "metadata"),
            logs_subdir=data.get("logs_subdir", "logs"),
        )


@dataclass
class InputSettings:
    """Input handling configuration settings."""

    recursive: bool = True
    validate: bool = True
    check_pixel_data: bool = True
    check_dimensions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recursive": self.recursive,
            "validate": self.validate,
            "check_pixel_data": self.check_pixel_data,
            "check_dimensions": self.check_dimensions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputSettings":
        """Create from dictionary."""
        return cls(
            recursive=data.get("recursive", True),
            validate=data.get("validate", True),
            check_pixel_data=data.get("check_pixel_data", True),
            check_dimensions=data.get("check_dimensions", True),
        )


@dataclass
class LoggingSettings:
    """Logging configuration settings."""

    level: str = "info"
    log_to_file: bool = True
    log_to_console: bool = True
    log_filename: str = "processing.log"

    def get_log_level(self) -> int:
        """Get logging module level."""
        return LogLevel(self.level).to_logging_level()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "log_to_file": self.log_to_file,
            "log_to_console": self.log_to_console,
            "log_filename": self.log_filename,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingSettings":
        """Create from dictionary."""
        return cls(
            level=data.get("level", "info"),
            log_to_file=data.get("log_to_file", True),
            log_to_console=data.get("log_to_console", True),
            log_filename=data.get("log_filename", "processing.log"),
        )


@dataclass
class Config:
    """
    Complete configuration for CAD Preprocess.

    This class holds all configuration settings for the preprocessing
    pipeline. It can be loaded from YAML files or created programmatically.

    Sections:
    - preprocessing: Windowing, normalization, resizing settings
    - metadata: Profile selection and additional fields
    - output: Naming and overwrite policies
    - input: File discovery and validation settings
    - logging: Log level and output settings

    Example YAML:
    ```yaml
    preprocessing:
      windowing:
        strategy: use_dicom_window
      normalization: min_max
      resizing:
        enabled: true
        target_height: 1024
        target_width: 1024
    metadata:
      profiles:
        - minimal
        - ml
    output:
      naming_policy: sop_instance_uid
      overwrite_policy: skip
    logging:
      level: info
    ```
    """

    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    metadata: MetadataSettings = field(default_factory=MetadataSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    input: InputSettings = field(default_factory=InputSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "preprocessing": self.preprocessing.to_dict(),
            "metadata": self.metadata.to_dict(),
            "output": self.output.to_dict(),
            "input": self.input.to_dict(),
            "logging": self.logging.to_dict(),
        }

    def to_yaml(self, indent: int = 2) -> str:
        """Convert to YAML string."""
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            indent=indent,
            sort_keys=False,
            allow_unicode=True,
        )

    def save(self, file_path: Union[Path, str]) -> None:
        """
        Save configuration to YAML file.

        Args:
            file_path: Path to save configuration.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

        logger.info(f"Saved configuration to: {file_path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            preprocessing=PreprocessingSettings.from_dict(data.get("preprocessing", {})),
            metadata=MetadataSettings.from_dict(data.get("metadata", {})),
            output=OutputSettings.from_dict(data.get("output", {})),
            input=InputSettings.from_dict(data.get("input", {})),
            logging=LoggingSettings.from_dict(data.get("logging", {})),
        )

    @classmethod
    def from_yaml(cls, yaml_string: str) -> "Config":
        """
        Create Config from YAML string.

        Args:
            yaml_string: YAML configuration string.

        Returns:
            Config instance.
        """
        data = yaml.safe_load(yaml_string) or {}
        return cls.from_dict(data)

    @classmethod
    def load(cls, file_path: Union[Path, str]) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file.

        Returns:
            Config instance.

        Raises:
            FileNotFoundError: If file does not exist.
            yaml.YAMLError: If file is not valid YAML.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        config = cls.from_yaml(content)
        logger.info(f"Loaded configuration from: {file_path}")

        return config

    @classmethod
    def default(cls) -> "Config":
        """
        Create default configuration.

        Default settings are designed to be safe and reproducible:
        - Use DICOM window values when available
        - Min-max normalization
        - Resize to 1024x1024 with aspect ratio preservation
        - Extract minimal metadata profile
        - Skip existing files (no overwrite)
        - Info-level logging

        Returns:
            Config with default settings.
        """
        return cls()

    def validate(self) -> List[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Validate windowing
        if self.preprocessing.windowing.strategy == "fixed_window":
            if self.preprocessing.windowing.window_center is None:
                errors.append("fixed_window strategy requires window_center")
            if self.preprocessing.windowing.window_width is None:
                errors.append("fixed_window strategy requires window_width")

        # Validate normalization
        valid_normalizations = ["min_max", "z_score"]
        if self.preprocessing.normalization not in valid_normalizations:
            errors.append(
                f"Invalid normalization: {self.preprocessing.normalization}. "
                f"Must be one of: {valid_normalizations}"
            )

        # Validate resizing
        if self.preprocessing.resizing.target_height <= 0:
            errors.append("target_height must be positive")
        if self.preprocessing.resizing.target_width <= 0:
            errors.append("target_width must be positive")

        valid_interpolations = ["nearest", "bilinear", "bicubic", "lanczos"]
        if self.preprocessing.resizing.interpolation not in valid_interpolations:
            errors.append(
                f"Invalid interpolation: {self.preprocessing.resizing.interpolation}. "
                f"Must be one of: {valid_interpolations}"
            )

        # Validate metadata profiles
        valid_profiles = ["minimal", "patient", "geometry", "ml", "acquisition"]
        for profile in self.metadata.profiles:
            if profile not in valid_profiles:
                errors.append(
                    f"Invalid metadata profile: {profile}. "
                    f"Must be one of: {valid_profiles}"
                )

        # Validate output settings
        valid_naming = ["sop_instance_uid", "original_filename", "sequential"]
        if self.output.naming_policy not in valid_naming:
            errors.append(
                f"Invalid naming_policy: {self.output.naming_policy}. "
                f"Must be one of: {valid_naming}"
            )

        valid_overwrite = ["skip", "overwrite", "rename", "error"]
        if self.output.overwrite_policy not in valid_overwrite:
            errors.append(
                f"Invalid overwrite_policy: {self.output.overwrite_policy}. "
                f"Must be one of: {valid_overwrite}"
            )

        # Validate logging
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if self.logging.level not in valid_levels:
            errors.append(
                f"Invalid logging level: {self.logging.level}. "
                f"Must be one of: {valid_levels}"
            )

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


def load_config(file_path: Optional[Union[Path, str]] = None) -> Config:
    """
    Load configuration from file or return default.

    This is a convenience function that:
    1. If file_path is provided, loads from that file
    2. If CAD_PREPROCESS_CONFIG env var is set, loads from that path
    3. Otherwise returns default configuration

    Args:
        file_path: Optional path to configuration file.

    Returns:
        Config instance.
    """
    if file_path is not None:
        return Config.load(file_path)

    env_path = os.environ.get("CAD_PREPROCESS_CONFIG")
    if env_path:
        return Config.load(env_path)

    return Config.default()


def create_default_config_file(output_path: Union[Path, str]) -> None:
    """
    Create a default configuration file.

    Useful for initializing a new project with documented defaults.

    Args:
        output_path: Path to write the configuration file.
    """
    config = Config.default()
    config.save(output_path)


# YAML configuration template with comments
CONFIG_TEMPLATE = """# CAD Preprocess Configuration
# =============================
# This file configures the DICOM preprocessing pipeline.

# Preprocessing Settings
# ----------------------
preprocessing:
  # Windowing (VOI LUT) settings
  windowing:
    # Strategy: "use_dicom_window" or "fixed_window"
    strategy: use_dicom_window
    # Required if strategy is "fixed_window":
    # window_center: 40
    # window_width: 400

  # Normalization method: "min_max" or "z_score"
  normalization: min_max

  # Image resizing settings
  resizing:
    enabled: true
    target_height: 1024
    target_width: 1024
    keep_aspect_ratio: true
    # Interpolation: "nearest", "bilinear", "bicubic", "lanczos"
    interpolation: bilinear
    padding_value: 0

  # Output data type: "uint8" or "float32"
  output_dtype: uint8

# Metadata Extraction Settings
# ----------------------------
metadata:
  # Profiles to extract: minimal, patient, geometry, ml, acquisition
  profiles:
    - minimal
    - ml
  # Additional DICOM fields to extract
  additional_fields: []
  # Set to true to extract all profiles
  include_all_profiles: false

# Output Settings
# ---------------
output:
  # Naming policy: "sop_instance_uid", "original_filename", "sequential"
  naming_policy: sop_instance_uid
  # Overwrite policy: "skip", "overwrite", "rename", "error"
  overwrite_policy: skip
  # Output image format
  image_format: png
  # Output subdirectories
  images_subdir: images
  metadata_subdir: metadata
  logs_subdir: logs

# Input Settings
# --------------
input:
  # Scan directories recursively
  recursive: true
  # Validate DICOM files before processing
  validate: true
  # Check for PixelData tag
  check_pixel_data: true
  # Check for Rows and Columns
  check_dimensions: true

# Logging Settings
# ----------------
logging:
  # Log level: "debug", "info", "warning", "error", "critical"
  level: info
  # Write logs to file
  log_to_file: true
  # Write logs to console
  log_to_console: true
  # Log filename (in logs directory)
  log_filename: processing.log
"""


def create_config_template(output_path: Union[Path, str]) -> None:
    """
    Create a configuration template with comments.

    This creates a well-documented template that users can customize.

    Args:
        output_path: Path to write the template.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(CONFIG_TEMPLATE)

    logger.info(f"Created configuration template: {output_path}")
