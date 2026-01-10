"""
Integration Module for CAD Systems.

This module provides a high-level Python API for integrating the preprocessing
pipeline with CAD (Computer-Aided Detection/Diagnosis) systems.

Used by:
- ML training pipelines
- Model inference pipelines
- CAD visualization software

Guarantees:
- Identical preprocessing for train and inference
- Traceable outputs with manifest files
- Reproducible configuration via hashing

Non-goals (explicitly out of scope):
- Model training
- Model inference
- Clinical decision making

Example Usage:
    # Simple one-liner for quick preprocessing
    >>> from cad_preprocess import preprocess
    >>> result = preprocess("input.dcm", "output/", config="config.yaml")

    # Stateful preprocessor for batch operations
    >>> from cad_preprocess import CADPreprocessor
    >>> processor = CADPreprocessor.from_config("config.yaml")
    >>> for file in dicom_files:
    ...     result = processor.process_file(file)

    # Identical preprocessing for training and inference
    >>> processor = CADPreprocessor.from_config("model_config.yaml")
    >>> train_results = processor.process_directory("train_data/")
    >>> # Save config hash for reproducibility
    >>> config_hash = processor.config_hash
    >>> # Later, for inference with identical preprocessing:
    >>> inference_processor = CADPreprocessor.from_config("model_config.yaml")
    >>> assert inference_processor.config_hash == config_hash
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from cad_preprocess.config import Config, load_config
from cad_preprocess.input_handler import InputHandler
from cad_preprocess.logging_utils import (
    ProcessingLogger,
    ProcessingStage,
    ProcessingStats,
    setup_logging,
)
from cad_preprocess.metadata_extractor import MetadataExtractor
from cad_preprocess.output_writer import OutputConfig, OutputWriter
from cad_preprocess.preprocessing_engine import PreprocessingEngine, PreprocessingResult


@dataclass
class ProcessingManifest:
    """
    Manifest for tracking preprocessing operations.

    Provides traceability by recording what was processed, when, and with
    what configuration.
    """

    manifest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    config_hash: str = ""
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    input_path: str = ""
    output_path: str = ""
    processed_files: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    version: str = ""

    def add_file(
        self,
        input_file: Path,
        output_file: Optional[Path],
        metadata_file: Optional[Path],
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Add a processed file to the manifest."""
        self.processed_files.append({
            "input": str(input_file),
            "output": str(output_file) if output_file else None,
            "metadata": str(metadata_file) if metadata_file else None,
            "success": success,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "manifest_id": self.manifest_id,
            "created_at": self.created_at.isoformat(),
            "config_hash": self.config_hash,
            "config_snapshot": self.config_snapshot,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "processed_files": self.processed_files,
            "stats": self.stats,
            "version": self.version,
        }

    def save(self, output_dir: Path) -> Path:
        """Save manifest to file."""
        manifest_path = output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return manifest_path

    @classmethod
    def load(cls, manifest_path: Path) -> "ProcessingManifest":
        """Load manifest from file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = cls()
        manifest.manifest_id = data.get("manifest_id", "")
        manifest.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        manifest.config_hash = data.get("config_hash", "")
        manifest.config_snapshot = data.get("config_snapshot", {})
        manifest.input_path = data.get("input_path", "")
        manifest.output_path = data.get("output_path", "")
        manifest.processed_files = data.get("processed_files", [])
        manifest.stats = data.get("stats", {})
        manifest.version = data.get("version", "")
        return manifest


@dataclass
class PreprocessResult:
    """
    Result of a preprocessing operation.

    Attributes:
        success: Whether preprocessing succeeded.
        image: Processed image as numpy array (if successful).
        metadata: Extracted metadata dictionary.
        output_image_path: Path to saved image (if written).
        output_metadata_path: Path to saved metadata (if written).
        original_path: Original input file path.
        error_message: Error message if failed.
        processing_time: Time taken in seconds.
    """

    success: bool
    image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_image_path: Optional[Path] = None
    output_metadata_path: Optional[Path] = None
    original_path: Optional[Path] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without image data)."""
        return {
            "success": self.success,
            "has_image": self.image is not None,
            "image_shape": list(self.image.shape) if self.image is not None else None,
            "metadata_keys": list(self.metadata.keys()),
            "output_image_path": str(self.output_image_path) if self.output_image_path else None,
            "output_metadata_path": str(self.output_metadata_path) if self.output_metadata_path else None,
            "original_path": str(self.original_path) if self.original_path else None,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
        }


@dataclass
class BatchResult:
    """
    Result of a batch preprocessing operation.

    Attributes:
        success: Whether the batch completed (may have individual failures).
        results: List of individual PreprocessResult objects.
        stats: Processing statistics.
        manifest: Processing manifest for traceability.
        config_hash: Hash of configuration used.
    """

    success: bool
    results: List[PreprocessResult] = field(default_factory=list)
    stats: Optional[ProcessingStats] = None
    manifest: Optional[ProcessingManifest] = None
    config_hash: str = ""

    @property
    def total_processed(self) -> int:
        """Count of successfully processed files."""
        return sum(1 for r in self.results if r.success)

    @property
    def total_failed(self) -> int:
        """Count of failed files."""
        return sum(1 for r in self.results if not r.success)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "BATCH PROCESSING SUMMARY",
            "=" * 50,
            f"Total files:    {len(self.results)}",
            f"Successful:     {self.total_processed}",
            f"Failed:         {self.total_failed}",
            f"Config hash:    {self.config_hash[:16]}...",
            "=" * 50,
        ]
        return "\n".join(lines)


class CADPreprocessor:
    """
    High-level preprocessor for CAD system integration.

    Provides a stateful interface for preprocessing DICOM files with
    guaranteed reproducibility and traceability.

    Guarantees:
        - Identical preprocessing for training and inference when using
          the same configuration
        - Traceable outputs via manifest files
        - Reproducible configuration via config hashing

    Example:
        >>> processor = CADPreprocessor.from_config("config.yaml")
        >>> result = processor.process_file("image.dcm")
        >>> if result.success:
        ...     print(f"Processed: {result.image.shape}")

        # For batch processing
        >>> batch = processor.process_directory("dicoms/", "output/")
        >>> print(batch.summary())
    """

    def __init__(
        self,
        config: Config,
        output_dir: Optional[Path] = None,
        write_outputs: bool = True,
        create_manifest: bool = True,
    ) -> None:
        """
        Initialize the CAD preprocessor.

        Args:
            config: Configuration object.
            output_dir: Default output directory.
            write_outputs: Whether to write images/metadata to disk.
            create_manifest: Whether to create processing manifests.
        """
        self._config = config
        self._output_dir = Path(output_dir) if output_dir else None
        self._write_outputs = write_outputs
        self._create_manifest = create_manifest

        # Compute config hash for reproducibility
        self._config_hash = self._compute_config_hash(config)

        # Initialize components
        self._input_handler = InputHandler(
            validate=config.input.validate,
            recursive=config.input.recursive,
            check_pixel_data=config.input.check_pixel_data,
            check_dimensions=config.input.check_dimensions,
        )

        self._preprocessing_engine = PreprocessingEngine(
            config=config.preprocessing.to_preprocessing_config()
        )

        self._metadata_extractor = MetadataExtractor(
            profiles=config.metadata.profiles,
            additional_fields=config.metadata.additional_fields,
            include_all_profiles=config.metadata.include_all_profiles,
        )

        self._output_writer: Optional[OutputWriter] = None
        if output_dir and write_outputs:
            output_config = OutputConfig(
                output_root=Path(output_dir),
                naming_policy=config.output.get_naming_policy(),
                overwrite_policy=config.output.get_overwrite_policy(),
                image_format=config.output.image_format,
                images_subdir=config.output.images_subdir,
                metadata_subdir=config.output.metadata_subdir,
                logs_subdir=config.output.logs_subdir,
            )
            self._output_writer = OutputWriter(Path(output_dir), config=output_config)

        self._logger = ProcessingLogger("cad_preprocess.integration")

        # Import version
        from cad_preprocess import __version__
        self._version = __version__

    @classmethod
    def from_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        write_outputs: bool = True,
        create_manifest: bool = True,
    ) -> "CADPreprocessor":
        """
        Create preprocessor from configuration file.

        Args:
            config_path: Path to YAML configuration file.
            output_dir: Default output directory.
            write_outputs: Whether to write outputs to disk.
            create_manifest: Whether to create manifests.

        Returns:
            Configured CADPreprocessor instance.
        """
        config = load_config(config_path)
        return cls(
            config=config,
            output_dir=Path(output_dir) if output_dir else None,
            write_outputs=write_outputs,
            create_manifest=create_manifest,
        )

    @property
    def config(self) -> Config:
        """Get the configuration."""
        return self._config

    @property
    def config_hash(self) -> str:
        """
        Get configuration hash for reproducibility verification.

        Two preprocessors with the same config_hash are guaranteed to
        produce identical outputs for the same inputs.
        """
        return self._config_hash

    def verify_config_hash(self, expected_hash: str) -> bool:
        """
        Verify that the current configuration matches an expected hash.

        Use this to ensure identical preprocessing between training
        and inference.

        Args:
            expected_hash: Expected configuration hash.

        Returns:
            True if hashes match, False otherwise.
        """
        return self._config_hash == expected_hash

    def _compute_config_hash(self, config: Config) -> str:
        """Compute deterministic hash of configuration."""
        # Create a deterministic string representation
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
                    "preserve_aspect_ratio": config.preprocessing.resizing.preserve_aspect_ratio,
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

    def process_file(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PreprocessResult:
        """
        Process a single DICOM file.

        Args:
            input_path: Path to DICOM file.
            output_dir: Override output directory.

        Returns:
            PreprocessResult with processed image and metadata.
        """
        import time
        start_time = time.time()

        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else self._output_dir

        try:
            # Preprocess image
            prep_result = self._preprocessing_engine.process(input_path)

            if not prep_result.success:
                return PreprocessResult(
                    success=False,
                    original_path=input_path,
                    error_message=prep_result.error_message,
                    processing_time=time.time() - start_time,
                )

            # Extract metadata
            meta_result = self._metadata_extractor.extract(input_path)
            metadata = meta_result.metadata if meta_result.success else {}

            # Write outputs if configured
            output_image_path = None
            output_metadata_path = None

            if self._write_outputs and output_dir and self._output_writer:
                sop_uid = metadata.get("SOPInstanceUID")

                img_result = self._output_writer.write_image(
                    prep_result.image,
                    sop_instance_uid=sop_uid,
                    original_path=input_path,
                )
                if img_result.success:
                    output_image_path = img_result.file_path

                meta_write_result = self._output_writer.write_metadata(
                    metadata,
                    sop_instance_uid=sop_uid,
                    original_path=input_path,
                )
                if meta_write_result.success:
                    output_metadata_path = meta_write_result.file_path

            return PreprocessResult(
                success=True,
                image=prep_result.image,
                metadata=metadata,
                output_image_path=output_image_path,
                output_metadata_path=output_metadata_path,
                original_path=input_path,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            self._logger.error(f"Error processing {input_path}: {e}")
            return PreprocessResult(
                success=False,
                original_path=input_path,
                error_message=f"{type(e).__name__}: {e}",
                processing_time=time.time() - start_time,
            )

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> BatchResult:
        """
        Process all DICOM files in a directory.

        Args:
            input_dir: Input directory path.
            output_dir: Output directory (overrides default).

        Returns:
            BatchResult with all processing results.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else self._output_dir

        # Update output writer if needed
        if output_dir and self._write_outputs:
            output_config = OutputConfig(
                output_root=output_dir,
                naming_policy=self._config.output.get_naming_policy(),
                overwrite_policy=self._config.output.get_overwrite_policy(),
                image_format=self._config.output.image_format,
                images_subdir=self._config.output.images_subdir,
                metadata_subdir=self._config.output.metadata_subdir,
                logs_subdir=self._config.output.logs_subdir,
            )
            self._output_writer = OutputWriter(output_dir, config=output_config)

        # Create manifest
        manifest = None
        if self._create_manifest:
            manifest = ProcessingManifest(
                config_hash=self._config_hash,
                config_snapshot=self._get_config_snapshot(),
                input_path=str(input_dir),
                output_path=str(output_dir) if output_dir else "",
                version=self._version,
            )

        results: List[PreprocessResult] = []

        with self._logger.processing_context("batch") as stats:
            # Discover files
            discovery = self._input_handler.discover(input_dir)
            stats.files_discovered = discovery.total_discovered
            stats.files_valid = discovery.total_valid

            self._logger.log_batch_start(discovery.total_valid)

            # Process each file
            for i, file_path in enumerate(discovery.valid_files, 1):
                self._logger.log_file_start(file_path, i, discovery.total_valid)

                result = self.process_file(file_path, output_dir)
                results.append(result)

                if result.success:
                    stats.files_processed += 1
                    self._logger.log_file_success(file_path)
                else:
                    stats.files_failed += 1
                    self._logger.log_file_error(
                        file_path,
                        Exception(result.error_message or "Unknown error"),
                        ProcessingStage.PREPROCESSING,
                    )

                # Update manifest
                if manifest:
                    manifest.add_file(
                        input_file=file_path,
                        output_file=result.output_image_path,
                        metadata_file=result.output_metadata_path,
                        success=result.success,
                        error_message=result.error_message,
                    )

            # Finalize manifest
            if manifest:
                manifest.stats = stats.to_dict()
                if output_dir:
                    manifest.save(output_dir)

        return BatchResult(
            success=True,
            results=results,
            stats=stats,
            manifest=manifest,
            config_hash=self._config_hash,
        )

    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> BatchResult:
        """
        Process a specific list of DICOM files.

        Args:
            file_paths: List of file paths to process.
            output_dir: Output directory.

        Returns:
            BatchResult with all processing results.
        """
        output_dir = Path(output_dir) if output_dir else self._output_dir

        # Update output writer if needed
        if output_dir and self._write_outputs:
            output_config = OutputConfig(
                output_root=output_dir,
                naming_policy=self._config.output.get_naming_policy(),
                overwrite_policy=self._config.output.get_overwrite_policy(),
                image_format=self._config.output.image_format,
            )
            self._output_writer = OutputWriter(output_dir, config=output_config)

        results: List[PreprocessResult] = []

        manifest = None
        if self._create_manifest:
            manifest = ProcessingManifest(
                config_hash=self._config_hash,
                config_snapshot=self._get_config_snapshot(),
                input_path="[multiple files]",
                output_path=str(output_dir) if output_dir else "",
                version=self._version,
            )

        with self._logger.processing_context("files") as stats:
            stats.files_discovered = len(file_paths)
            stats.files_valid = len(file_paths)

            for i, file_path in enumerate(file_paths, 1):
                file_path = Path(file_path)
                self._logger.log_file_start(file_path, i, len(file_paths))

                result = self.process_file(file_path, output_dir)
                results.append(result)

                if result.success:
                    stats.files_processed += 1
                else:
                    stats.files_failed += 1

                if manifest:
                    manifest.add_file(
                        input_file=file_path,
                        output_file=result.output_image_path,
                        metadata_file=result.output_metadata_path,
                        success=result.success,
                        error_message=result.error_message,
                    )

            if manifest:
                manifest.stats = stats.to_dict()
                if output_dir:
                    manifest.save(output_dir)

        return BatchResult(
            success=True,
            results=results,
            stats=stats,
            manifest=manifest,
            config_hash=self._config_hash,
        )

    def iterate_files(
        self,
        input_dir: Union[str, Path],
    ) -> Iterator[PreprocessResult]:
        """
        Iterate over DICOM files, yielding processed results.

        Memory-efficient processing for large datasets.

        Args:
            input_dir: Input directory path.

        Yields:
            PreprocessResult for each file.
        """
        input_dir = Path(input_dir)
        discovery = self._input_handler.discover(input_dir)

        for file_path in discovery.valid_files:
            yield self.process_file(file_path)

    def _get_config_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the configuration for manifest."""
        return {
            "preprocessing": {
                "windowing": {
                    "strategy": self._config.preprocessing.windowing.strategy,
                    "window_center": self._config.preprocessing.windowing.window_center,
                    "window_width": self._config.preprocessing.windowing.window_width,
                },
                "normalization": self._config.preprocessing.normalization,
                "resizing": {
                    "target_height": self._config.preprocessing.resizing.target_height,
                    "target_width": self._config.preprocessing.resizing.target_width,
                    "preserve_aspect_ratio": self._config.preprocessing.resizing.preserve_aspect_ratio,
                },
                "output_dtype": self._config.preprocessing.output_dtype,
            },
            "metadata_profiles": self._config.metadata.profiles,
            "output_format": self._config.output.image_format,
        }


# Convenience functions for simple usage


def preprocess(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Union[str, Path, Config]] = None,
    write_outputs: bool = True,
) -> Union[PreprocessResult, BatchResult]:
    """
    Preprocess DICOM file(s) with a single function call.

    This is the simplest way to use the preprocessing pipeline.

    Args:
        input_path: Path to DICOM file or directory.
        output_dir: Output directory for processed files.
        config: Configuration file path or Config object.
        write_outputs: Whether to write outputs to disk.

    Returns:
        PreprocessResult for single file, BatchResult for directory.

    Examples:
        # Process single file
        >>> result = preprocess("image.dcm", "output/")
        >>> if result.success:
        ...     print(f"Shape: {result.image.shape}")

        # Process directory
        >>> batch = preprocess("dicoms/", "output/", config="config.yaml")
        >>> print(f"Processed: {batch.total_processed} files")

        # Get image array without writing to disk
        >>> result = preprocess("image.dcm", write_outputs=False)
        >>> image_array = result.image
    """
    input_path = Path(input_path)

    # Load configuration
    if config is None:
        cfg = Config.default()
    elif isinstance(config, Config):
        cfg = config
    else:
        cfg = load_config(config)

    # Create preprocessor
    processor = CADPreprocessor(
        config=cfg,
        output_dir=Path(output_dir) if output_dir else None,
        write_outputs=write_outputs,
        create_manifest=write_outputs,
    )

    # Process based on input type
    if input_path.is_file():
        return processor.process_file(input_path, output_dir)
    elif input_path.is_dir():
        return processor.process_directory(input_path, output_dir)
    else:
        return PreprocessResult(
            success=False,
            error_message=f"Input path does not exist: {input_path}",
        )


def get_config_hash(config: Union[str, Path, Config]) -> str:
    """
    Get the configuration hash for reproducibility verification.

    Use this to store the config hash during training and verify it
    matches during inference.

    Args:
        config: Configuration file path or Config object.

    Returns:
        SHA-256 hash of the configuration.

    Example:
        # During training
        >>> train_hash = get_config_hash("model_config.yaml")
        >>> save_to_model_metadata(train_hash)

        # During inference
        >>> inference_hash = get_config_hash("model_config.yaml")
        >>> assert inference_hash == train_hash, "Config mismatch!"
    """
    if isinstance(config, Config):
        cfg = config
    else:
        cfg = load_config(config)

    processor = CADPreprocessor(cfg, write_outputs=False, create_manifest=False)
    return processor.config_hash


def verify_preprocessing_consistency(
    config1: Union[str, Path, Config],
    config2: Union[str, Path, Config],
) -> bool:
    """
    Verify that two configurations will produce identical preprocessing.

    Args:
        config1: First configuration.
        config2: Second configuration.

    Returns:
        True if preprocessing will be identical.

    Example:
        >>> is_consistent = verify_preprocessing_consistency(
        ...     "train_config.yaml",
        ...     "inference_config.yaml",
        ... )
        >>> if not is_consistent:
        ...     raise ValueError("Training and inference configs differ!")
    """
    hash1 = get_config_hash(config1)
    hash2 = get_config_hash(config2)
    return hash1 == hash2
