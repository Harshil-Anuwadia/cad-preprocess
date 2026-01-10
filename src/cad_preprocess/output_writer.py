"""
Output Writer Module for CAD Preprocess.

This module handles persisting processed images and metadata to disk
in a reproducible, organized directory structure.

Directory structure:
- images/     - Processed images (PNG format)
- metadata/   - Extracted metadata (JSON format)
- logs/       - Processing logs

Naming policy: Uses SOPInstanceUID for consistent, unique file naming.
Overwrite policy: Configurable (default: skip existing files).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# Configure module logger
logger = logging.getLogger(__name__)


class OverwritePolicy(str, Enum):
    """Policy for handling existing files."""

    SKIP = "skip"  # Skip if file exists
    OVERWRITE = "overwrite"  # Overwrite existing files
    RENAME = "rename"  # Rename with suffix if exists
    ERROR = "error"  # Raise error if exists


class NamingPolicy(str, Enum):
    """Policy for naming output files."""

    SOP_INSTANCE_UID = "sop_instance_uid"  # Use SOPInstanceUID
    ORIGINAL_FILENAME = "original_filename"  # Use original file name
    SEQUENTIAL = "sequential"  # Use sequential numbering


# Default directory names
DEFAULT_IMAGES_DIR = "images"
DEFAULT_METADATA_DIR = "metadata"
DEFAULT_LOGS_DIR = "logs"


@dataclass
class OutputConfig:
    """Configuration for output writer."""

    output_root: Path
    images_subdir: str = DEFAULT_IMAGES_DIR
    metadata_subdir: str = DEFAULT_METADATA_DIR
    logs_subdir: str = DEFAULT_LOGS_DIR
    naming_policy: NamingPolicy = NamingPolicy.SOP_INSTANCE_UID
    overwrite_policy: OverwritePolicy = OverwritePolicy.SKIP
    image_format: str = "png"
    create_dirs: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        self.output_root = Path(self.output_root).resolve()

        if isinstance(self.naming_policy, str):
            self.naming_policy = NamingPolicy(self.naming_policy)

        if isinstance(self.overwrite_policy, str):
            self.overwrite_policy = OverwritePolicy(self.overwrite_policy)

    @property
    def images_dir(self) -> Path:
        """Get full path to images directory."""
        return self.output_root / self.images_subdir

    @property
    def metadata_dir(self) -> Path:
        """Get full path to metadata directory."""
        return self.output_root / self.metadata_subdir

    @property
    def logs_dir(self) -> Path:
        """Get full path to logs directory."""
        return self.output_root / self.logs_subdir


@dataclass
class WriteResult:
    """Result of a write operation."""

    success: bool
    file_path: Optional[Path] = None
    action: str = "written"  # written, skipped, overwritten, renamed, error
    error_message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


@dataclass
class BatchWriteResult:
    """Result of batch write operations."""

    total: int = 0
    written: int = 0
    skipped: int = 0
    errors: int = 0
    results: List[WriteResult] = field(default_factory=list)

    def add(self, result: WriteResult) -> None:
        """Add a write result."""
        self.results.append(result)
        self.total += 1
        if result.success:
            if result.action == "skipped":
                self.skipped += 1
            else:
                self.written += 1
        else:
            self.errors += 1


def generate_filename(
    sop_instance_uid: Optional[str] = None,
    original_path: Optional[Path] = None,
    sequence_number: Optional[int] = None,
    naming_policy: NamingPolicy = NamingPolicy.SOP_INSTANCE_UID,
) -> str:
    """
    Generate a filename based on naming policy.

    Args:
        sop_instance_uid: SOPInstanceUID from DICOM.
        original_path: Original file path.
        sequence_number: Sequential number for naming.
        naming_policy: Policy to use for naming.

    Returns:
        Generated filename (without extension).
    """
    if naming_policy == NamingPolicy.SOP_INSTANCE_UID:
        if sop_instance_uid:
            return sop_instance_uid
        elif original_path:
            return original_path.stem
        else:
            return f"image_{sequence_number or 0:06d}"

    elif naming_policy == NamingPolicy.ORIGINAL_FILENAME:
        if original_path:
            return original_path.stem
        elif sop_instance_uid:
            return sop_instance_uid
        else:
            return f"image_{sequence_number or 0:06d}"

    elif naming_policy == NamingPolicy.SEQUENTIAL:
        return f"image_{sequence_number or 0:06d}"

    else:
        return f"image_{sequence_number or 0:06d}"


def resolve_output_path(
    output_dir: Path,
    filename: str,
    extension: str,
    overwrite_policy: OverwritePolicy,
) -> tuple[Path, str]:
    """
    Resolve output path based on overwrite policy.

    Args:
        output_dir: Output directory.
        filename: Base filename.
        extension: File extension (with or without dot).
        overwrite_policy: Policy for existing files.

    Returns:
        Tuple of (resolved path, action to take).

    Raises:
        FileExistsError: If policy is ERROR and file exists.
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    output_path = output_dir / f"{filename}{extension}"

    if not output_path.exists():
        return output_path, "write"

    # File exists - apply policy
    if overwrite_policy == OverwritePolicy.SKIP:
        return output_path, "skip"

    elif overwrite_policy == OverwritePolicy.OVERWRITE:
        return output_path, "overwrite"

    elif overwrite_policy == OverwritePolicy.RENAME:
        # Find unique name with suffix
        counter = 1
        while True:
            new_path = output_dir / f"{filename}_{counter}{extension}"
            if not new_path.exists():
                return new_path, "rename"
            counter += 1
            if counter > 10000:  # Safety limit
                raise RuntimeError("Too many files with same name")

    elif overwrite_policy == OverwritePolicy.ERROR:
        raise FileExistsError(f"File already exists: {output_path}")

    return output_path, "write"


def write_image(
    image: np.ndarray,
    output_path: Path,
    image_format: str = "png",
) -> None:
    """
    Write an image array to file.

    Args:
        image: Image array (2D uint8 or float).
        output_path: Output file path.
        image_format: Image format (png, etc.).
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Handle grayscale
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image, mode="L")
    else:
        pil_image = Image.fromarray(image)

    pil_image.save(output_path, format=image_format.upper())
    logger.debug(f"Wrote image: {output_path}")


def write_metadata(
    metadata: Dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> None:
    """
    Write metadata to JSON file.

    Args:
        metadata: Metadata dictionary.
        output_path: Output file path.
        indent: JSON indentation level.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=indent, ensure_ascii=False)

    logger.debug(f"Wrote metadata: {output_path}")


def write_log(
    log_data: Dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> None:
    """
    Write log data to JSON file.

    Args:
        log_data: Log data dictionary.
        output_path: Output file path.
        indent: JSON indentation level.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=indent, ensure_ascii=False, default=str)

    logger.debug(f"Wrote log: {output_path}")


class OutputWriter:
    """
    Output writer for processed DICOM images and metadata.

    This class handles persisting processed images and extracted metadata
    to disk in an organized, reproducible directory structure.

    Directory structure:
    ```
    output_root/
    ├── images/          # Processed PNG images
    ├── metadata/        # JSON metadata files
    └── logs/            # Processing logs
    ```

    File naming uses SOPInstanceUID by default for consistent, unique names.

    Example:
        >>> writer = OutputWriter("/path/to/output")
        >>> writer.write_image(processed_array, sop_uid="1.2.3.4.5")
        >>> writer.write_metadata({"Modality": "CR"}, sop_uid="1.2.3.4.5")
    """

    def __init__(
        self,
        output_root: Union[Path, str],
        config: Optional[OutputConfig] = None,
        naming_policy: NamingPolicy = NamingPolicy.SOP_INSTANCE_UID,
        overwrite_policy: OverwritePolicy = OverwritePolicy.SKIP,
        image_format: str = "png",
    ) -> None:
        """
        Initialize the output writer.

        Args:
            output_root: Root directory for all outputs.
            config: Full configuration (overrides other args if provided).
            naming_policy: Policy for naming output files.
            overwrite_policy: Policy for handling existing files.
            image_format: Format for output images.
        """
        if config is not None:
            self.config = config
        else:
            self.config = OutputConfig(
                output_root=Path(output_root),
                naming_policy=naming_policy,
                overwrite_policy=overwrite_policy,
                image_format=image_format,
            )

        self._sequence_counter = 0
        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Create output directories if they don't exist."""
        if self.config.create_dirs:
            self.config.images_dir.mkdir(parents=True, exist_ok=True)
            self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.config.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized output directories at: {self.config.output_root}")

    def _get_filename(
        self,
        sop_instance_uid: Optional[str] = None,
        original_path: Optional[Path] = None,
    ) -> str:
        """Generate filename based on policy."""
        filename = generate_filename(
            sop_instance_uid=sop_instance_uid,
            original_path=original_path,
            sequence_number=self._sequence_counter,
            naming_policy=self.config.naming_policy,
        )
        self._sequence_counter += 1
        return filename

    def write_image(
        self,
        image: np.ndarray,
        sop_instance_uid: Optional[str] = None,
        original_path: Optional[Union[Path, str]] = None,
        filename: Optional[str] = None,
    ) -> WriteResult:
        """
        Write a processed image to the images directory.

        Args:
            image: Image array (2D, uint8 or float [0,1]).
            sop_instance_uid: SOPInstanceUID for naming.
            original_path: Original DICOM file path.
            filename: Override filename (without extension).

        Returns:
            WriteResult with status and path.
        """
        try:
            # Determine filename
            if filename is None:
                filename = self._get_filename(
                    sop_instance_uid=sop_instance_uid,
                    original_path=Path(original_path) if original_path else None,
                )

            # Resolve output path
            output_path, action = resolve_output_path(
                output_dir=self.config.images_dir,
                filename=filename,
                extension=self.config.image_format,
                overwrite_policy=self.config.overwrite_policy,
            )

            if action == "skip":
                logger.debug(f"Skipped existing image: {output_path}")
                return WriteResult(success=True, file_path=output_path, action="skipped")

            # Write image
            write_image(image, output_path, self.config.image_format)

            return WriteResult(
                success=True,
                file_path=output_path,
                action="overwritten" if action == "overwrite" else "written",
            )

        except Exception as e:
            logger.error(f"Failed to write image: {type(e).__name__}: {e}")
            return WriteResult(
                success=False,
                action="error",
                error_message=f"{type(e).__name__}: {e}",
            )

    def write_metadata(
        self,
        metadata: Dict[str, Any],
        sop_instance_uid: Optional[str] = None,
        original_path: Optional[Union[Path, str]] = None,
        filename: Optional[str] = None,
    ) -> WriteResult:
        """
        Write metadata to the metadata directory.

        Args:
            metadata: Metadata dictionary.
            sop_instance_uid: SOPInstanceUID for naming.
            original_path: Original DICOM file path.
            filename: Override filename (without extension).

        Returns:
            WriteResult with status and path.
        """
        try:
            # Determine filename
            if filename is None:
                # Use SOPInstanceUID from metadata if available
                uid = sop_instance_uid or metadata.get("SOPInstanceUID")
                filename = self._get_filename(
                    sop_instance_uid=uid,
                    original_path=Path(original_path) if original_path else None,
                )

            # Resolve output path
            output_path, action = resolve_output_path(
                output_dir=self.config.metadata_dir,
                filename=filename,
                extension="json",
                overwrite_policy=self.config.overwrite_policy,
            )

            if action == "skip":
                logger.debug(f"Skipped existing metadata: {output_path}")
                return WriteResult(success=True, file_path=output_path, action="skipped")

            # Write metadata
            write_metadata(metadata, output_path)

            return WriteResult(
                success=True,
                file_path=output_path,
                action="overwritten" if action == "overwrite" else "written",
            )

        except Exception as e:
            logger.error(f"Failed to write metadata: {type(e).__name__}: {e}")
            return WriteResult(
                success=False,
                action="error",
                error_message=f"{type(e).__name__}: {e}",
            )

    def write_processing_result(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any],
        sop_instance_uid: Optional[str] = None,
        original_path: Optional[Union[Path, str]] = None,
    ) -> tuple[WriteResult, WriteResult]:
        """
        Write both image and metadata for a processing result.

        Uses the same filename for both outputs.

        Args:
            image: Processed image array.
            metadata: Extracted metadata.
            sop_instance_uid: SOPInstanceUID for naming.
            original_path: Original DICOM file path.

        Returns:
            Tuple of (image_result, metadata_result).
        """
        # Generate consistent filename for both
        uid = sop_instance_uid or metadata.get("SOPInstanceUID")
        filename = self._get_filename(
            sop_instance_uid=uid,
            original_path=Path(original_path) if original_path else None,
        )

        # Write both with same filename
        image_result = self.write_image(image, filename=filename)
        metadata_result = self.write_metadata(metadata, filename=filename)

        return image_result, metadata_result

    def write_processing_log(
        self,
        log_data: Dict[str, Any],
        log_name: str = "processing_log",
    ) -> WriteResult:
        """
        Write a processing log file.

        Args:
            log_data: Log data dictionary.
            log_name: Name for the log file.

        Returns:
            WriteResult with status and path.
        """
        try:
            # Add timestamp
            log_data["timestamp"] = datetime.now().isoformat()

            # Generate unique log filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{log_name}_{timestamp_str}"

            output_path = self.config.logs_dir / f"{filename}.json"

            write_log(log_data, output_path)

            return WriteResult(success=True, file_path=output_path, action="written")

        except Exception as e:
            logger.error(f"Failed to write log: {type(e).__name__}: {e}")
            return WriteResult(
                success=False,
                action="error",
                error_message=f"{type(e).__name__}: {e}",
            )

    def write_batch_summary(
        self,
        summary: Dict[str, Any],
    ) -> WriteResult:
        """
        Write a batch processing summary.

        Args:
            summary: Summary data including counts and any errors.

        Returns:
            WriteResult with status and path.
        """
        return self.write_processing_log(summary, log_name="batch_summary")

    def get_image_path(
        self,
        sop_instance_uid: str,
    ) -> Path:
        """
        Get the expected path for an image file.

        Args:
            sop_instance_uid: SOPInstanceUID.

        Returns:
            Expected file path.
        """
        return self.config.images_dir / f"{sop_instance_uid}.{self.config.image_format}"

    def get_metadata_path(
        self,
        sop_instance_uid: str,
    ) -> Path:
        """
        Get the expected path for a metadata file.

        Args:
            sop_instance_uid: SOPInstanceUID.

        Returns:
            Expected file path.
        """
        return self.config.metadata_dir / f"{sop_instance_uid}.json"

    def exists(
        self,
        sop_instance_uid: str,
        check_image: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        """
        Check if outputs already exist for a SOPInstanceUID.

        Args:
            sop_instance_uid: SOPInstanceUID to check.
            check_image: Check for image file.
            check_metadata: Check for metadata file.

        Returns:
            True if all checked outputs exist.
        """
        if check_image:
            if not self.get_image_path(sop_instance_uid).exists():
                return False
        if check_metadata:
            if not self.get_metadata_path(sop_instance_uid).exists():
                return False
        return True

    def clean(self, confirm: bool = False) -> bool:
        """
        Clean (delete) all output directories.

        Args:
            confirm: Must be True to actually delete.

        Returns:
            True if cleaning was performed.
        """
        if not confirm:
            logger.warning("Clean operation requires confirm=True")
            return False

        try:
            if self.config.output_root.exists():
                shutil.rmtree(self.config.output_root)
                logger.info(f"Cleaned output directory: {self.config.output_root}")

            # Recreate directories
            self._initialize_directories()
            return True

        except Exception as e:
            logger.error(f"Failed to clean output directory: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current outputs.

        Returns:
            Dictionary with counts and paths.
        """
        image_count = len(list(self.config.images_dir.glob(f"*.{self.config.image_format}")))
        metadata_count = len(list(self.config.metadata_dir.glob("*.json")))
        log_count = len(list(self.config.logs_dir.glob("*.json")))

        return {
            "output_root": str(self.config.output_root),
            "images_count": image_count,
            "metadata_count": metadata_count,
            "logs_count": log_count,
            "naming_policy": self.config.naming_policy.value,
            "overwrite_policy": self.config.overwrite_policy.value,
        }

    def __repr__(self) -> str:
        return (
            f"OutputWriter(output_root='{self.config.output_root}', "
            f"naming={self.config.naming_policy.value}, "
            f"overwrite={self.config.overwrite_policy.value})"
        )
