"""
Tests for the Input Handler module.

These tests verify:
- Single DICOM file discovery
- Directory scanning (recursive and non-recursive)
- DICOM validation (PixelData, Rows, Columns)
- Graceful handling of invalid files
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cad_preprocess.input_handler import (
    DICOM_EXTENSIONS,
    DiscoveryResult,
    InputHandler,
    ValidationResult,
    discover_dicom_files,
    is_dicom_extension,
    scan_directory,
    validate_dicom_file,
)


class TestIsdicmExtension:
    """Tests for is_dicom_extension function."""

    def test_dcm_extension(self):
        """Test .dcm extension is recognized."""
        assert is_dicom_extension(Path("image.dcm")) is True

    def test_dicom_extension(self):
        """Test .dicom extension is recognized."""
        assert is_dicom_extension(Path("image.dicom")) is True

    def test_uppercase_extension(self):
        """Test uppercase extensions are recognized."""
        assert is_dicom_extension(Path("image.DCM")) is True
        assert is_dicom_extension(Path("image.DICOM")) is True

    def test_mixed_case_extension(self):
        """Test mixed case extensions are recognized."""
        assert is_dicom_extension(Path("image.Dcm")) is True

    def test_non_dicom_extension(self):
        """Test non-DICOM extensions are rejected."""
        assert is_dicom_extension(Path("image.png")) is False
        assert is_dicom_extension(Path("image.jpg")) is False
        assert is_dicom_extension(Path("image.nii")) is False

    def test_no_extension(self):
        """Test files without extension are rejected."""
        assert is_dicom_extension(Path("image")) is False


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result_is_truthy(self):
        """Test that valid results are truthy."""
        result = ValidationResult(is_valid=True, file_path=Path("test.dcm"))
        assert result
        assert bool(result) is True

    def test_invalid_result_is_falsy(self):
        """Test that invalid results are falsy."""
        result = ValidationResult(
            is_valid=False, file_path=Path("test.dcm"), error_message="Test error"
        )
        assert not result
        assert bool(result) is False


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_add_valid(self):
        """Test adding valid files."""
        result = DiscoveryResult()
        result.add_valid(Path("test1.dcm"))
        result.add_valid(Path("test2.dcm"))

        assert len(result.valid_files) == 2
        assert result.total_valid == 2

    def test_add_skipped(self):
        """Test adding skipped files."""
        result = DiscoveryResult()
        result.add_skipped(Path("bad1.dcm"), "Missing PixelData")
        result.add_skipped(Path("bad2.dcm"), "Invalid DICOM")

        assert len(result.skipped_files) == 2
        assert result.total_skipped == 2
        assert result.skipped_files[0][1] == "Missing PixelData"


class TestValidateDicomFile:
    """Tests for validate_dicom_file function."""

    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        result = validate_dicom_file(Path("/nonexistent/path/file.dcm"))
        assert result.is_valid is False
        assert "does not exist" in result.error_message

    def test_directory_path(self, tmp_path):
        """Test validation of directory path."""
        result = validate_dicom_file(tmp_path)
        assert result.is_valid is False
        assert "not a file" in result.error_message

    @patch("cad_preprocess.input_handler.pydicom.dcmread")
    def test_valid_dicom_file(self, mock_dcmread, tmp_path):
        """Test validation of valid DICOM file."""
        # Create a mock DICOM dataset
        mock_ds = MagicMock()
        mock_ds.PixelData = b"pixel_data"
        mock_ds.Rows = 512
        mock_ds.Columns = 512
        mock_dcmread.return_value = mock_ds

        # Create a temporary file
        dicom_file = tmp_path / "test.dcm"
        dicom_file.touch()

        result = validate_dicom_file(dicom_file)
        assert result.is_valid is True
        assert result.error_message is None

    @patch("cad_preprocess.input_handler.pydicom.dcmread")
    def test_missing_pixel_data(self, mock_dcmread, tmp_path):
        """Test validation fails when PixelData is missing."""
        mock_ds = MagicMock(spec=[])  # No attributes
        mock_dcmread.return_value = mock_ds

        dicom_file = tmp_path / "test.dcm"
        dicom_file.touch()

        result = validate_dicom_file(dicom_file, check_pixel_data=True)
        assert result.is_valid is False
        assert "PixelData" in result.error_message

    @patch("cad_preprocess.input_handler.pydicom.dcmread")
    def test_missing_rows(self, mock_dcmread, tmp_path):
        """Test validation fails when Rows is missing."""
        mock_ds = MagicMock()
        mock_ds.PixelData = b"pixel_data"
        del mock_ds.Rows  # Remove Rows attribute
        mock_dcmread.return_value = mock_ds

        dicom_file = tmp_path / "test.dcm"
        dicom_file.touch()

        result = validate_dicom_file(dicom_file)
        assert result.is_valid is False
        assert "Rows" in result.error_message

    @patch("cad_preprocess.input_handler.pydicom.dcmread")
    def test_missing_columns(self, mock_dcmread, tmp_path):
        """Test validation fails when Columns is missing."""
        mock_ds = MagicMock()
        mock_ds.PixelData = b"pixel_data"
        mock_ds.Rows = 512
        del mock_ds.Columns  # Remove Columns attribute
        mock_dcmread.return_value = mock_ds

        dicom_file = tmp_path / "test.dcm"
        dicom_file.touch()

        result = validate_dicom_file(dicom_file)
        assert result.is_valid is False
        assert "Columns" in result.error_message

    @patch("cad_preprocess.input_handler.pydicom.dcmread")
    def test_skip_pixel_data_check(self, mock_dcmread, tmp_path):
        """Test validation passes when pixel data check is disabled."""
        mock_ds = MagicMock()
        mock_ds.Rows = 512
        mock_ds.Columns = 512
        # No PixelData
        del mock_ds.PixelData
        mock_dcmread.return_value = mock_ds

        dicom_file = tmp_path / "test.dcm"
        dicom_file.touch()

        result = validate_dicom_file(dicom_file, check_pixel_data=False)
        assert result.is_valid is True


class TestScanDirectory:
    """Tests for scan_directory function."""

    def test_scan_flat_directory(self, tmp_path):
        """Test scanning a flat directory."""
        # Create test files
        (tmp_path / "image1.dcm").touch()
        (tmp_path / "image2.dcm").touch()
        (tmp_path / "image3.dicom").touch()
        (tmp_path / "readme.txt").touch()  # Should be ignored

        files = list(scan_directory(tmp_path, recursive=False))
        assert len(files) == 3
        assert all(f.suffix.lower() in DICOM_EXTENSIONS for f in files)

    def test_scan_recursive(self, tmp_path):
        """Test recursive directory scanning."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "image1.dcm").touch()
        (subdir / "image2.dcm").touch()

        files = list(scan_directory(tmp_path, recursive=True))
        assert len(files) == 2

    def test_scan_non_recursive(self, tmp_path):
        """Test non-recursive directory scanning."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "image1.dcm").touch()
        (subdir / "image2.dcm").touch()

        files = list(scan_directory(tmp_path, recursive=False))
        assert len(files) == 1

    def test_scan_nonexistent_directory(self, tmp_path):
        """Test scanning non-existent directory."""
        files = list(scan_directory(tmp_path / "nonexistent"))
        assert len(files) == 0

    def test_scan_file_path(self, tmp_path):
        """Test scanning a file path (not directory)."""
        file_path = tmp_path / "test.dcm"
        file_path.touch()
        files = list(scan_directory(file_path))
        assert len(files) == 0


class TestDiscoverDicomFiles:
    """Tests for discover_dicom_files function."""

    @patch("cad_preprocess.input_handler.validate_dicom_file")
    def test_discover_single_valid_file(self, mock_validate, tmp_path):
        """Test discovering a single valid DICOM file."""
        dicom_file = tmp_path / "test.dcm"
        dicom_file.touch()

        mock_validate.return_value = ValidationResult(is_valid=True, file_path=dicom_file)

        result = discover_dicom_files(dicom_file)
        assert result.total_discovered == 1
        assert result.total_valid == 1
        assert result.total_skipped == 0
        assert dicom_file in result.valid_files

    @patch("cad_preprocess.input_handler.validate_dicom_file")
    def test_discover_single_invalid_file(self, mock_validate, tmp_path):
        """Test discovering a single invalid DICOM file."""
        dicom_file = tmp_path / "test.dcm"
        dicom_file.touch()

        mock_validate.return_value = ValidationResult(
            is_valid=False, file_path=dicom_file, error_message="Invalid"
        )

        result = discover_dicom_files(dicom_file)
        assert result.total_discovered == 1
        assert result.total_valid == 0
        assert result.total_skipped == 1

    @patch("cad_preprocess.input_handler.validate_dicom_file")
    def test_discover_directory(self, mock_validate, tmp_path):
        """Test discovering DICOM files from directory."""
        (tmp_path / "image1.dcm").touch()
        (tmp_path / "image2.dcm").touch()

        mock_validate.return_value = ValidationResult(is_valid=True, file_path=tmp_path)

        result = discover_dicom_files(tmp_path)
        assert result.total_discovered == 2
        assert result.total_valid == 2

    def test_discover_without_validation(self, tmp_path):
        """Test discovery without validation."""
        (tmp_path / "image1.dcm").touch()
        (tmp_path / "image2.dcm").touch()

        result = discover_dicom_files(tmp_path, validate=False)
        assert result.total_discovered == 2
        assert result.total_valid == 2
        assert result.total_skipped == 0

    def test_discover_nonexistent_path(self):
        """Test discovering from non-existent path."""
        result = discover_dicom_files(Path("/nonexistent/path"))
        assert result.total_discovered == 0
        assert result.total_valid == 0


class TestInputHandler:
    """Tests for InputHandler class."""

    def test_init_defaults(self):
        """Test default initialization."""
        handler = InputHandler()
        assert handler.validate is True
        assert handler.recursive is True
        assert handler.check_pixel_data is True
        assert handler.check_dimensions is True

    def test_init_custom(self):
        """Test custom initialization."""
        handler = InputHandler(
            validate=False,
            recursive=False,
            check_pixel_data=False,
            check_dimensions=False,
        )
        assert handler.validate is False
        assert handler.recursive is False
        assert handler.check_pixel_data is False
        assert handler.check_dimensions is False

    @patch("cad_preprocess.input_handler.discover_dicom_files")
    def test_discover(self, mock_discover, tmp_path):
        """Test discover method."""
        mock_result = DiscoveryResult()
        mock_result.add_valid(tmp_path / "test.dcm")
        mock_discover.return_value = mock_result

        handler = InputHandler()
        result = handler.discover(tmp_path)

        assert result == mock_result
        mock_discover.assert_called_once()

    @patch("cad_preprocess.input_handler.discover_dicom_files")
    def test_get_valid_files(self, mock_discover, tmp_path):
        """Test get_valid_files convenience method."""
        test_file = tmp_path / "test.dcm"
        mock_result = DiscoveryResult()
        mock_result.add_valid(test_file)
        mock_discover.return_value = mock_result

        handler = InputHandler()
        files = handler.get_valid_files(tmp_path)

        assert len(files) == 1
        assert test_file in files

    @patch("cad_preprocess.input_handler.validate_dicom_file")
    def test_validate_single(self, mock_validate, tmp_path):
        """Test validate_single method."""
        test_file = tmp_path / "test.dcm"
        mock_validate.return_value = ValidationResult(is_valid=True, file_path=test_file)

        handler = InputHandler()
        result = handler.validate_single(test_file)

        assert result.is_valid is True
        mock_validate.assert_called_once()

    def test_repr(self):
        """Test string representation."""
        handler = InputHandler()
        repr_str = repr(handler)
        assert "InputHandler" in repr_str
        assert "validate=True" in repr_str
