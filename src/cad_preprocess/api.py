"""
API Interface Module.

This module provides a clean, simple API for other Python programs to reuse
the preprocessing logic. It is the primary entry point for programmatic usage.

Design Goals:
- Simple function signature that covers 90% of use cases
- Returns structured results with counts and error summaries
- Accepts configuration as dict or YAML file path
- Consistent behavior for single files and directories

Example Usage:
    >>> from cad_preprocess.api import preprocess
    >>> 
    >>> # Basic usage with directory
    >>> result = preprocess("./dicoms", "./output")
    >>> print(f"Processed: {result.processed_count}")
    >>> print(f"Skipped: {result.skipped_count}")
    >>> 
    >>> # With configuration file
    >>> result = preprocess("./dicoms", "./output", config="config.yaml")
    >>> 
    >>> # With configuration dict
    >>> result = preprocess("./dicoms", "./output", config={
    ...     "preprocessing": {
    ...         "windowing": {"strategy": "fixed_window", "window_center": 40, "window_width": 400}
    ...     }
    ... })
    >>> 
    >>> # Check for errors
    >>> if result.errors:
    ...     for error in result.error_summary:
    ...         print(f"  {error['file']}: {error['message']}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from cad_preprocess.config import Config, load_config
from cad_preprocess.input_handler import InputHandler
from cad_preprocess.logging_utils import ProcessingLogger
from cad_preprocess.metadata_extractor import MetadataExtractor
from cad_preprocess.output_writer import OutputConfig, OutputWriter
from cad_preprocess.preprocessing_engine import PreprocessingEngine


@dataclass
class PreprocessingResult:
    """
    Result of the preprocessing operation.
    
    Attributes:
        processed_count: Number of files successfully processed.
        skipped_count: Number of files skipped (invalid DICOM, already exists, etc.).
        error_count: Number of files that failed with errors.
        error_summary: List of error details for failed files.
        processed_files: List of successfully processed file paths.
        skipped_files: List of skipped file paths with reasons.
        config_hash: Hash of the configuration used (for reproducibility).
        output_dir: Path to the output directory.
    """
    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    error_summary: List[Dict[str, Any]] = field(default_factory=list)
    processed_files: List[str] = field(default_factory=list)
    skipped_files: List[Dict[str, str]] = field(default_factory=list)
    config_hash: str = ""
    output_dir: Optional[str] = None
    
    @property
    def total_files(self) -> int:
        """Total number of files encountered."""
        return self.processed_count + self.skipped_count + self.error_count
    
    @property
    def success_rate(self) -> float:
        """Percentage of files successfully processed."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_count / self.total_files) * 100
    
    @property
    def errors(self) -> bool:
        """Whether any errors occurred."""
        return self.error_count > 0
    
    def __repr__(self) -> str:
        return (
            f"PreprocessingResult("
            f"processed={self.processed_count}, "
            f"skipped={self.skipped_count}, "
            f"errors={self.error_count})"
        )
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            "PREPROCESSING SUMMARY",
            "=" * 50,
            f"Total files:      {self.total_files}",
            f"Processed:        {self.processed_count}",
            f"Skipped:          {self.skipped_count}",
            f"Errors:           {self.error_count}",
            f"Success rate:     {self.success_rate:.1f}%",
        ]
        
        if self.config_hash:
            lines.append(f"Config hash:      {self.config_hash[:16]}...")
        
        if self.output_dir:
            lines.append(f"Output:           {self.output_dir}")
        
        lines.append("=" * 50)
        
        if self.error_summary:
            lines.append("\nERRORS:")
            for err in self.error_summary[:10]:  # Show first 10 errors
                lines.append(f"  - {err.get('file', 'unknown')}: {err.get('message', 'Unknown error')}")
            if len(self.error_summary) > 10:
                lines.append(f"  ... and {len(self.error_summary) - 10} more errors")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processed_count": self.processed_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "error_summary": self.error_summary,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "config_hash": self.config_hash,
            "output_dir": self.output_dir,
            "total_files": self.total_files,
            "success_rate": self.success_rate,
        }


def preprocess(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
) -> PreprocessingResult:
    """
    Preprocess DICOM files for CAD system usage.
    
    This is the primary API entry point for preprocessing DICOM images.
    It handles both single files and directories, applies the preprocessing
    pipeline, and writes outputs to the specified location.
    
    Args:
        input_path: Path to a DICOM file or directory containing DICOM files.
        output_path: Path to the output directory for processed files.
        config: Configuration as one of:
            - str/Path: Path to a YAML configuration file
            - dict: Configuration dictionary
            - None: Use default configuration
    
    Returns:
        PreprocessingResult with:
            - processed_count: Number of successfully processed files
            - skipped_count: Number of skipped files
            - error_summary: List of error details for failed files
    
    Examples:
        # Basic usage
        >>> result = preprocess("./dicoms", "./output")
        >>> print(f"Processed {result.processed_count} files")
        
        # With YAML config file
        >>> result = preprocess("./dicoms", "./output", config="config.yaml")
        
        # With configuration dict
        >>> result = preprocess("./dicoms", "./output", config={
        ...     "preprocessing": {
        ...         "windowing": {
        ...             "strategy": "fixed_window",
        ...             "window_center": 40,
        ...             "window_width": 400
        ...         },
        ...         "normalization": "min_max",
        ...         "resizing": {
        ...             "target_height": 512,
        ...             "target_width": 512
        ...         }
        ...     },
        ...     "metadata": {
        ...         "profiles": ["minimal", "ml"]
        ...     },
        ...     "output": {
        ...         "naming_policy": "sop_instance_uid",
        ...         "format": "png"
        ...     }
        ... })
        
        # Check results
        >>> if result.errors:
        ...     print("Some files failed:")
        ...     for err in result.error_summary:
        ...         print(f"  {err['file']}: {err['message']}")
        
        # Get summary
        >>> print(result.summary())
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Initialize result
    result = PreprocessingResult(output_dir=str(output_path))
    
    # Load configuration
    cfg = _load_configuration(config)
    
    # Initialize components
    input_handler = InputHandler(
        validate=cfg.input.validate,
        recursive=cfg.input.recursive,
        check_pixel_data=cfg.input.check_pixel_data,
        check_dimensions=cfg.input.check_dimensions,
    )
    
    preprocessing_engine = PreprocessingEngine(
        config=cfg.preprocessing.to_preprocessing_config()
    )
    
    metadata_extractor = MetadataExtractor(
        profiles=cfg.metadata.profiles,
        additional_fields=cfg.metadata.additional_fields,
        include_all_profiles=cfg.metadata.include_all_profiles,
    )
    
    output_config = OutputConfig(
        output_root=output_path,
        naming_policy=cfg.output.get_naming_policy(),
        overwrite_policy=cfg.output.get_overwrite_policy(),
        image_format=cfg.output.image_format,
        images_subdir=cfg.output.images_subdir,
        metadata_subdir=cfg.output.metadata_subdir,
        logs_subdir=cfg.output.logs_subdir,
    )
    output_writer = OutputWriter(output_path, config=output_config)
    
    # Compute config hash
    result.config_hash = _compute_config_hash(cfg)
    
    # Discover files
    if input_path.is_file():
        files_to_process = [input_path]
        skipped_files: List[Dict[str, str]] = []
    else:
        discovery = input_handler.discover(input_path)
        files_to_process = discovery.valid_files
        skipped_files = [
            {"file": str(f), "reason": "Invalid DICOM"}
            for f in discovery.invalid_files
        ]
    
    # Track skipped files
    result.skipped_count = len(skipped_files)
    result.skipped_files = skipped_files
    
    # Process each file
    logger = ProcessingLogger("cad_preprocess.api")
    
    for file_path in files_to_process:
        try:
            # Preprocess
            prep_result = preprocessing_engine.process(file_path)
            
            if not prep_result.success:
                result.error_count += 1
                result.error_summary.append({
                    "file": str(file_path),
                    "stage": "preprocessing",
                    "message": prep_result.error_message or "Unknown preprocessing error",
                })
                continue
            
            # Extract metadata
            meta_result = metadata_extractor.extract(file_path)
            metadata = meta_result.metadata if meta_result.success else {}
            
            # Get SOP Instance UID for naming
            sop_uid = metadata.get("SOPInstanceUID")
            
            # Write image
            img_result = output_writer.write_image(
                prep_result.image,
                sop_instance_uid=sop_uid,
                original_path=file_path,
            )
            
            if not img_result.success:
                result.error_count += 1
                result.error_summary.append({
                    "file": str(file_path),
                    "stage": "output",
                    "message": img_result.message or "Failed to write image",
                })
                continue
            
            # Write metadata
            output_writer.write_metadata(
                metadata,
                sop_instance_uid=sop_uid,
                original_path=file_path,
            )
            
            # Success
            result.processed_count += 1
            result.processed_files.append(str(file_path))
            
        except Exception as e:
            result.error_count += 1
            result.error_summary.append({
                "file": str(file_path),
                "stage": "unknown",
                "message": f"{type(e).__name__}: {str(e)}",
            })
            logger.error(f"Error processing {file_path}: {e}")
    
    return result


def _load_configuration(
    config: Optional[Union[str, Path, Dict[str, Any]]]
) -> Config:
    """
    Load configuration from various input types.
    
    Args:
        config: Configuration as path string, Path object, dict, or None.
    
    Returns:
        Config object.
    """
    if config is None:
        return Config.default()
    
    if isinstance(config, dict):
        return _config_from_dict(config)
    
    # Assume it's a path
    return load_config(config)


def _config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Create Config object from dictionary.
    
    Args:
        config_dict: Configuration dictionary.
    
    Returns:
        Config object.
    """
    from cad_preprocess.config import (
        Config,
        InputSettings,
        LoggingSettings,
        MetadataSettings,
        OutputSettings,
        PreprocessingSettings,
        ResizingSettings,
        WindowingSettings,
    )
    
    # Start with defaults
    cfg = Config.default()
    
    # Override preprocessing settings
    if "preprocessing" in config_dict:
        prep = config_dict["preprocessing"]
        
        if "windowing" in prep:
            w = prep["windowing"]
            cfg.preprocessing.windowing = WindowingSettings(
                strategy=w.get("strategy", "from_dicom"),
                window_center=w.get("window_center"),
                window_width=w.get("window_width"),
            )
        
        if "normalization" in prep:
            cfg.preprocessing.normalization = prep["normalization"]
        
        if "resizing" in prep:
            r = prep["resizing"]
            cfg.preprocessing.resizing = ResizingSettings(
                target_height=r.get("target_height", 1024),
                target_width=r.get("target_width", 1024),
                keep_aspect_ratio=r.get("keep_aspect_ratio", r.get("preserve_aspect_ratio", True)),
                interpolation=r.get("interpolation", "bilinear"),
            )
        
        if "output_dtype" in prep:
            cfg.preprocessing.output_dtype = prep["output_dtype"]
    
    # Override metadata settings
    if "metadata" in config_dict:
        meta = config_dict["metadata"]
        cfg.metadata = MetadataSettings(
            profiles=meta.get("profiles", ["minimal"]),
            additional_fields=meta.get("additional_fields", []),
            include_all_profiles=meta.get("include_all_profiles", False),
        )
    
    # Override output settings
    if "output" in config_dict:
        out = config_dict["output"]
        cfg.output = OutputSettings(
            naming_policy=out.get("naming_policy", "sop_instance_uid"),
            overwrite_policy=out.get("overwrite_policy", "skip"),
            image_format=out.get("format", out.get("image_format", "png")),
            images_subdir=out.get("images_subdir", "images"),
            metadata_subdir=out.get("metadata_subdir", "metadata"),
            logs_subdir=out.get("logs_subdir", "logs"),
        )
    
    # Override input settings
    if "input" in config_dict:
        inp = config_dict["input"]
        cfg.input = InputSettings(
            validate=inp.get("validate", True),
            recursive=inp.get("recursive", True),
            check_pixel_data=inp.get("check_pixel_data", True),
            check_dimensions=inp.get("check_dimensions", True),
        )
    
    return cfg


def _compute_config_hash(config: Config) -> str:
    """
    Compute deterministic hash of configuration for reproducibility.
    
    Args:
        config: Config object.
    
    Returns:
        SHA-256 hash string.
    """
    import hashlib
    import json
    
    config_dict = {
        "preprocessing": {
            "windowing": {
                "strategy": config.preprocessing.windowing.strategy,
                "window_center": config.preprocessing.windowing.window_center,
                "window_width": config.preprocessing.windowing.window_width,
            },
            "normalization": config.preprocessing.normalization,
            "resizing": {
                "target_height": config.preprocessing.resizing.target_height,
                "target_width": config.preprocessing.resizing.target_width,
                "keep_aspect_ratio": config.preprocessing.resizing.keep_aspect_ratio,
                "interpolation": config.preprocessing.resizing.interpolation,
            },
            "output_dtype": config.preprocessing.output_dtype,
        },
        "metadata": {
            "profiles": sorted(config.metadata.profiles),
            "additional_fields": sorted(config.metadata.additional_fields),
        },
    }
    
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


# Convenience aliases
process = preprocess
run_preprocessing = preprocess
