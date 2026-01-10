"""
Input Handler Module for CAD Preprocess.

This module handles:
- Single DICOM file input
- Directory input with recursive scanning
- DICOM file validation
- Production of normalized list of valid input files

Validation rules:
- File must be readable by pydicom
- PixelData tag must exist
- Rows and Columns must be present

On validation failure: skip and log (fail-safe behavior)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import pydicom
from pydicom.errors import InvalidDicomError

# Configure module logger
logger = logging.getLogger(__name__)

# Supported DICOM file extensions
DICOM_EXTENSIONS: Set[str] = {".dcm", ".dicom"}


@dataclass
class ValidationResult:
    """Result of DICOM file validation."""

    is_valid: bool
    file_path: Path
    error_message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class DiscoveryResult:
    """Result of DICOM file discovery and validation."""

    valid_files: List[Path] = field(default_factory=list)
    skipped_files: List[Tuple[Path, str]] = field(default_factory=list)
    total_discovered: int = 0
    total_valid: int = 0
    total_skipped: int = 0

    def add_valid(self, file_path: Path) -> None:
        """Add a valid file to the result."""
        self.valid_files.append(file_path)
        self.total_valid += 1

    def add_skipped(self, file_path: Path, reason: str) -> None:
        """Add a skipped file with reason to the result."""
        self.skipped_files.append((file_path, reason))
        self.total_skipped += 1

    def __repr__(self) -> str:
        return (
            f"DiscoveryResult(total_discovered={self.total_discovered}, "
            f"total_valid={self.total_valid}, total_skipped={self.total_skipped})"
        )


def is_dicom_extension(file_path: Path) -> bool:
    """
    Check if a file has a recognized DICOM extension.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the file has a .dcm or .dicom extension (case-insensitive).
    """
    return file_path.suffix.lower() in DICOM_EXTENSIONS


def validate_dicom_file(
    file_path: Path,
    check_pixel_data: bool = True,
    check_dimensions: bool = True,
) -> ValidationResult:
    """
    Validate a DICOM file for preprocessing compatibility.

    Validation checks:
    1. File is readable by pydicom
    2. PixelData tag exists (if check_pixel_data=True)
    3. Rows and Columns attributes exist (if check_dimensions=True)

    Args:
        file_path: Path to the DICOM file to validate.
        check_pixel_data: Whether to check for PixelData tag.
        check_dimensions: Whether to check for Rows and Columns.

    Returns:
        ValidationResult with validation status and any error message.
    """
    file_path = Path(file_path).resolve()

    # Check file exists
    if not file_path.exists():
        return ValidationResult(
            is_valid=False,
            file_path=file_path,
            error_message=f"File does not exist: {file_path}",
        )

    # Check file is a file (not directory)
    if not file_path.is_file():
        return ValidationResult(
            is_valid=False,
            file_path=file_path,
            error_message=f"Path is not a file: {file_path}",
        )

    # Try to read with pydicom
    try:
        # Use stop_before_pixels=True for faster validation when only checking metadata
        # We'll do a two-pass check if pixel data validation is needed
        ds = pydicom.dcmread(file_path, stop_before_pixels=not check_pixel_data)
    except InvalidDicomError as e:
        return ValidationResult(
            is_valid=False,
            file_path=file_path,
            error_message=f"Invalid DICOM file: {e}",
        )
    except PermissionError:
        return ValidationResult(
            is_valid=False,
            file_path=file_path,
            error_message=f"Permission denied: {file_path}",
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            file_path=file_path,
            error_message=f"Failed to read DICOM file: {type(e).__name__}: {e}",
        )

    # Check for PixelData
    if check_pixel_data:
        if not hasattr(ds, "PixelData") or ds.PixelData is None:
            return ValidationResult(
                is_valid=False,
                file_path=file_path,
                error_message="DICOM file missing PixelData tag",
            )

    # Check for Rows and Columns
    if check_dimensions:
        if not hasattr(ds, "Rows") or ds.Rows is None:
            return ValidationResult(
                is_valid=False,
                file_path=file_path,
                error_message="DICOM file missing Rows attribute",
            )
        if not hasattr(ds, "Columns") or ds.Columns is None:
            return ValidationResult(
                is_valid=False,
                file_path=file_path,
                error_message="DICOM file missing Columns attribute",
            )

    return ValidationResult(is_valid=True, file_path=file_path)


def scan_directory(
    directory: Path,
    recursive: bool = True,
    extensions: Optional[Set[str]] = None,
) -> Iterator[Path]:
    """
    Scan a directory for files with specific extensions.

    Args:
        directory: Directory path to scan.
        recursive: Whether to scan subdirectories recursively.
        extensions: Set of file extensions to match (case-insensitive).
                   If None, uses DICOM_EXTENSIONS.

    Yields:
        Paths to files matching the specified extensions.
    """
    directory = Path(directory).resolve()
    extensions = extensions or DICOM_EXTENSIONS

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return

    if not directory.is_dir():
        logger.warning(f"Path is not a directory: {directory}")
        return

    # Choose glob pattern based on recursive flag
    pattern = "**/*" if recursive else "*"

    try:
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                yield file_path
    except PermissionError as e:
        logger.warning(f"Permission denied scanning directory: {e}")
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {type(e).__name__}: {e}")


def discover_dicom_files(
    input_path: Path | str,
    recursive: bool = True,
    validate: bool = True,
    check_pixel_data: bool = True,
    check_dimensions: bool = True,
) -> DiscoveryResult:
    """
    Discover and optionally validate DICOM files from a path.

    Accepts either a single DICOM file or a directory containing DICOM files.
    Invalid files are skipped and logged (fail-safe behavior).

    Args:
        input_path: Path to a single DICOM file or directory.
        recursive: Whether to scan directories recursively.
        validate: Whether to validate discovered files.
        check_pixel_data: Whether to check for PixelData during validation.
        check_dimensions: Whether to check for Rows/Columns during validation.

    Returns:
        DiscoveryResult containing lists of valid and skipped files.
    """
    input_path = Path(input_path).resolve()
    result = DiscoveryResult()

    # Collect files to process
    files_to_process: List[Path] = []

    if input_path.is_file():
        # Single file input
        if is_dicom_extension(input_path):
            files_to_process.append(input_path)
        else:
            # Try to validate anyway - some DICOM files don't have extension
            logger.info(
                f"File {input_path} does not have DICOM extension, attempting validation"
            )
            files_to_process.append(input_path)
    elif input_path.is_dir():
        # Directory input
        files_to_process.extend(scan_directory(input_path, recursive=recursive))
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return result

    result.total_discovered = len(files_to_process)

    # Process each file
    for file_path in files_to_process:
        if validate:
            validation = validate_dicom_file(
                file_path,
                check_pixel_data=check_pixel_data,
                check_dimensions=check_dimensions,
            )
            if validation.is_valid:
                result.add_valid(file_path)
                logger.debug(f"Valid DICOM file: {file_path}")
            else:
                result.add_skipped(file_path, validation.error_message or "Unknown error")
                logger.warning(f"Skipped invalid file: {file_path} - {validation.error_message}")
        else:
            # No validation, assume all files are valid
            result.add_valid(file_path)

    logger.info(
        f"Discovery complete: {result.total_valid} valid, "
        f"{result.total_skipped} skipped out of {result.total_discovered} files"
    )

    return result


class InputHandler:
    """
    High-level interface for handling DICOM input discovery and validation.

    This class provides a convenient API for discovering and validating
    DICOM files from various input sources (single file or directory).

    Example:
        >>> handler = InputHandler(validate=True, recursive=True)
        >>> result = handler.discover("/path/to/dicoms")
        >>> for dicom_path in result.valid_files:
        ...     process(dicom_path)
    """

    def __init__(
        self,
        validate: bool = True,
        recursive: bool = True,
        check_pixel_data: bool = True,
        check_dimensions: bool = True,
        extensions: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize the InputHandler.

        Args:
            validate: Whether to validate DICOM files.
            recursive: Whether to scan directories recursively.
            check_pixel_data: Whether to check for PixelData during validation.
            check_dimensions: Whether to check for Rows/Columns during validation.
            extensions: Custom set of file extensions to recognize as DICOM.
        """
        self.validate = validate
        self.recursive = recursive
        self.check_pixel_data = check_pixel_data
        self.check_dimensions = check_dimensions
        self.extensions = extensions or DICOM_EXTENSIONS

    def discover(self, input_path: Path | str) -> DiscoveryResult:
        """
        Discover DICOM files from the given input path.

        Args:
            input_path: Path to a single DICOM file or directory.

        Returns:
            DiscoveryResult containing lists of valid and skipped files.
        """
        return discover_dicom_files(
            input_path=input_path,
            recursive=self.recursive,
            validate=self.validate,
            check_pixel_data=self.check_pixel_data,
            check_dimensions=self.check_dimensions,
        )

    def get_valid_files(self, input_path: Path | str) -> List[Path]:
        """
        Get list of valid DICOM file paths.

        Convenience method that returns only the valid files list.

        Args:
            input_path: Path to a single DICOM file or directory.

        Returns:
            List of absolute paths to valid DICOM files.
        """
        result = self.discover(input_path)
        return result.valid_files

    def validate_single(self, file_path: Path | str) -> ValidationResult:
        """
        Validate a single DICOM file.

        Args:
            file_path: Path to the DICOM file.

        Returns:
            ValidationResult with validation status and any error message.
        """
        return validate_dicom_file(
            file_path=Path(file_path),
            check_pixel_data=self.check_pixel_data,
            check_dimensions=self.check_dimensions,
        )

    def __repr__(self) -> str:
        return (
            f"InputHandler(validate={self.validate}, recursive={self.recursive}, "
            f"check_pixel_data={self.check_pixel_data}, "
            f"check_dimensions={self.check_dimensions})"
        )
