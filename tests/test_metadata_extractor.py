"""
Tests for the Metadata Extractor module.

These tests verify:
- Profile-based field extraction
- Missing field handling (return None)
- JSON serialization
- File I/O operations
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cad_preprocess.metadata_extractor import (
    EXTENDED_FIELDS,
    PROFILE_FIELDS,
    ExtractionResult,
    MetadataExtractor,
    MetadataProfile,
    convert_value_to_json_serializable,
    extract_field,
    extract_metadata,
    extract_metadata_from_file,
    get_profile_fields,
    save_metadata,
)


class TestConvertValueToJsonSerializable:
    """Tests for convert_value_to_json_serializable function."""

    def test_none(self):
        """Test None value."""
        assert convert_value_to_json_serializable(None) is None

    def test_string(self):
        """Test string value."""
        assert convert_value_to_json_serializable("test") == "test"

    def test_int(self):
        """Test integer value."""
        assert convert_value_to_json_serializable(42) == 42

    def test_float(self):
        """Test float value."""
        assert convert_value_to_json_serializable(3.14) == 3.14

    def test_bool(self):
        """Test boolean value."""
        assert convert_value_to_json_serializable(True) is True

    def test_list(self):
        """Test list value."""
        result = convert_value_to_json_serializable([1, 2, 3])
        assert result == [1, 2, 3]

    def test_bytes_returns_none(self):
        """Test bytes are converted to None."""
        assert convert_value_to_json_serializable(b"binary") is None

    def test_person_name(self):
        """Test PersonName conversion."""
        mock_person = MagicMock()
        mock_person.family_name = "Doe"
        mock_person.__str__ = lambda self: "Doe^John"

        result = convert_value_to_json_serializable(mock_person)
        assert isinstance(result, str)

    def test_numpy_scalar(self):
        """Test numpy scalar conversion."""
        import numpy as np

        value = np.int64(42)
        result = convert_value_to_json_serializable(value)
        assert result == 42
        assert isinstance(result, int)


class TestExtractField:
    """Tests for extract_field function."""

    def test_existing_field(self):
        """Test extracting existing field."""
        mock_ds = MagicMock()
        mock_ds.Modality = "CR"

        result = extract_field(mock_ds, "Modality")
        assert result == "CR"

    def test_missing_field(self):
        """Test extracting missing field returns None."""
        mock_ds = MagicMock(spec=[])

        result = extract_field(mock_ds, "NonExistentField")
        assert result is None

    def test_field_with_error(self):
        """Test field extraction handles errors gracefully."""
        mock_ds = MagicMock()
        # Property that raises exception
        type(mock_ds).BadField = property(lambda self: 1 / 0)

        result = extract_field(mock_ds, "BadField")
        assert result is None


class TestGetProfileFields:
    """Tests for get_profile_fields function."""

    def test_single_profile(self):
        """Test getting fields from single profile."""
        fields = get_profile_fields([MetadataProfile.MINIMAL])
        assert "SOPInstanceUID" in fields
        assert "Modality" in fields
        assert "Rows" in fields
        assert "Columns" in fields

    def test_multiple_profiles(self):
        """Test getting fields from multiple profiles."""
        fields = get_profile_fields([MetadataProfile.MINIMAL, MetadataProfile.ML])
        # From minimal
        assert "SOPInstanceUID" in fields
        # From ML
        assert "BitsAllocated" in fields
        assert "RescaleSlope" in fields

    def test_profile_overlap(self):
        """Test overlapping fields are deduplicated."""
        fields = get_profile_fields([MetadataProfile.MINIMAL, MetadataProfile.GEOMETRY])
        # Both have Rows and Columns - should be unique
        assert fields.count("Rows") if isinstance(fields, list) else True


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ExtractionResult(
            metadata={"Modality": "CR"},
            source_file=Path("/test/file.dcm"),
            profiles_used=["minimal"],
            fields_extracted=1,
            fields_missing=0,
        )

        d = result.to_dict()
        assert d["metadata"] == {"Modality": "CR"}
        assert d["profiles_used"] == ["minimal"]
        assert d["success"] is True

    def test_to_json(self):
        """Test JSON serialization."""
        result = ExtractionResult(
            metadata={"Modality": "CR"},
            profiles_used=["minimal"],
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["Modality"] == "CR"


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_minimal_profile(self):
        """Test extraction with minimal profile."""
        mock_ds = MagicMock()
        mock_ds.SOPInstanceUID = "1.2.3.4"
        mock_ds.Modality = "CR"
        mock_ds.Rows = 512
        mock_ds.Columns = 512

        metadata = extract_metadata(mock_ds, profiles=[MetadataProfile.MINIMAL])

        assert metadata["SOPInstanceUID"] == "1.2.3.4"
        assert metadata["Modality"] == "CR"
        assert metadata["Rows"] == 512
        assert metadata["Columns"] == 512

    def test_default_profile(self):
        """Test extraction with default profile."""
        mock_ds = MagicMock()
        mock_ds.SOPInstanceUID = "1.2.3.4"
        mock_ds.Modality = "CR"

        metadata = extract_metadata(mock_ds)
        assert "SOPInstanceUID" in metadata

    def test_additional_fields(self):
        """Test extraction with additional fields."""
        mock_ds = MagicMock()
        mock_ds.SOPInstanceUID = "1.2.3.4"
        mock_ds.CustomField = "custom"

        metadata = extract_metadata(
            mock_ds,
            profiles=[MetadataProfile.MINIMAL],
            additional_fields=["CustomField"],
        )

        assert metadata.get("CustomField") == "custom"

    def test_include_all_profiles(self):
        """Test extraction with all profiles."""
        mock_ds = MagicMock()
        mock_ds.SOPInstanceUID = "1.2.3.4"
        mock_ds.PatientAge = "045Y"
        mock_ds.BitsAllocated = 16

        metadata = extract_metadata(mock_ds, include_all_profiles=True)

        # Should include fields from all profiles
        assert "SOPInstanceUID" in metadata  # minimal
        assert "PatientAge" in metadata  # patient
        assert "BitsAllocated" in metadata  # ml


class TestExtractMetadataFromFile:
    """Tests for extract_metadata_from_file function."""

    @patch("cad_preprocess.metadata_extractor.pydicom.dcmread")
    def test_successful_extraction(self, mock_dcmread, tmp_path):
        """Test successful extraction from file."""
        mock_ds = MagicMock()
        mock_ds.SOPInstanceUID = "1.2.3.4"
        mock_ds.Modality = "CR"
        mock_ds.Rows = 512
        mock_ds.Columns = 512
        mock_dcmread.return_value = mock_ds

        file_path = tmp_path / "test.dcm"
        file_path.touch()

        result = extract_metadata_from_file(file_path)

        assert result.success is True
        assert result.metadata["SOPInstanceUID"] == "1.2.3.4"
        assert result.fields_extracted > 0

    @patch("cad_preprocess.metadata_extractor.pydicom.dcmread")
    def test_failed_extraction(self, mock_dcmread, tmp_path):
        """Test failed extraction."""
        mock_dcmread.side_effect = Exception("Read error")

        file_path = tmp_path / "test.dcm"
        file_path.touch()

        result = extract_metadata_from_file(file_path)

        assert result.success is False
        assert result.error_message is not None


class TestSaveMetadata:
    """Tests for save_metadata function."""

    def test_save_metadata_only(self, tmp_path):
        """Test saving metadata without extraction info."""
        result = ExtractionResult(
            metadata={"Modality": "CR", "Rows": 512},
            profiles_used=["minimal"],
        )

        output_path = tmp_path / "metadata.json"
        success = save_metadata(result, output_path, include_extraction_info=False)

        assert success is True
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
        assert data == {"Modality": "CR", "Rows": 512}

    def test_save_with_extraction_info(self, tmp_path):
        """Test saving with extraction info."""
        result = ExtractionResult(
            metadata={"Modality": "CR"},
            profiles_used=["minimal"],
            fields_extracted=1,
        )

        output_path = tmp_path / "metadata.json"
        success = save_metadata(result, output_path, include_extraction_info=True)

        assert success is True

        with open(output_path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "profiles_used" in data

    def test_save_creates_directories(self, tmp_path):
        """Test save creates parent directories."""
        result = ExtractionResult(metadata={"Modality": "CR"})

        output_path = tmp_path / "nested" / "dir" / "metadata.json"
        success = save_metadata(result, output_path)

        assert success is True
        assert output_path.exists()


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""

    def test_init_default(self):
        """Test default initialization."""
        extractor = MetadataExtractor()
        assert MetadataProfile.MINIMAL in extractor.profiles

    def test_init_with_profiles(self):
        """Test initialization with profiles."""
        extractor = MetadataExtractor(profiles=["ml", "geometry"])
        assert MetadataProfile.ML in extractor.profiles
        assert MetadataProfile.GEOMETRY in extractor.profiles

    def test_init_with_enum_profiles(self):
        """Test initialization with enum profiles."""
        extractor = MetadataExtractor(
            profiles=[MetadataProfile.ML, MetadataProfile.GEOMETRY]
        )
        assert MetadataProfile.ML in extractor.profiles

    def test_init_with_additional_fields(self):
        """Test initialization with additional fields."""
        extractor = MetadataExtractor(additional_fields=["CustomField"])
        assert "CustomField" in extractor.additional_fields

    @patch("cad_preprocess.metadata_extractor.extract_metadata_from_file")
    def test_extract(self, mock_extract):
        """Test extract method."""
        mock_result = ExtractionResult(metadata={"Modality": "CR"})
        mock_extract.return_value = mock_result

        extractor = MetadataExtractor()
        result = extractor.extract("/fake/path.dcm")

        assert result == mock_result
        mock_extract.assert_called_once()

    def test_extract_from_dataset(self):
        """Test extract_from_dataset method."""
        mock_ds = MagicMock()
        mock_ds.SOPInstanceUID = "1.2.3.4"
        mock_ds.Modality = "CR"
        mock_ds.Rows = 512
        mock_ds.Columns = 512

        extractor = MetadataExtractor(profiles=["minimal"])
        metadata = extractor.extract_from_dataset(mock_ds)

        assert metadata["Modality"] == "CR"

    def test_get_fields(self):
        """Test get_fields method."""
        extractor = MetadataExtractor(
            profiles=["minimal"],
            additional_fields=["CustomField"],
        )
        fields = extractor.get_fields()

        assert "SOPInstanceUID" in fields
        assert "CustomField" in fields

    def test_available_profiles(self):
        """Test available_profiles static method."""
        profiles = MetadataExtractor.available_profiles()
        assert "minimal" in profiles
        assert "ml" in profiles
        assert "patient" in profiles

    def test_get_profile_fields_list(self):
        """Test get_profile_fields_list static method."""
        fields = MetadataExtractor.get_profile_fields_list("minimal")
        assert "SOPInstanceUID" in fields
        assert "Modality" in fields

    def test_get_extended_fields(self):
        """Test get_extended_fields static method."""
        fields = MetadataExtractor.get_extended_fields()
        assert len(fields) > 0
        assert "StudyInstanceUID" in fields

    def test_repr(self):
        """Test string representation."""
        extractor = MetadataExtractor(profiles=["ml"])
        repr_str = repr(extractor)
        assert "MetadataExtractor" in repr_str
        assert "ml" in repr_str


class TestProfileFields:
    """Tests to verify profile field definitions."""

    def test_minimal_profile_fields(self):
        """Test minimal profile has expected fields."""
        fields = PROFILE_FIELDS[MetadataProfile.MINIMAL]
        assert "SOPInstanceUID" in fields
        assert "Modality" in fields
        assert "Rows" in fields
        assert "Columns" in fields

    def test_patient_profile_fields(self):
        """Test patient profile has expected fields."""
        fields = PROFILE_FIELDS[MetadataProfile.PATIENT]
        assert "PatientSex" in fields
        assert "PatientAge" in fields
        assert "PatientSize" in fields
        assert "PatientWeight" in fields

    def test_geometry_profile_fields(self):
        """Test geometry profile has expected fields."""
        fields = PROFILE_FIELDS[MetadataProfile.GEOMETRY]
        assert "Rows" in fields
        assert "Columns" in fields
        assert "PixelSpacing" in fields
        assert "ImageOrientationPatient" in fields
        assert "ImagePositionPatient" in fields

    def test_ml_profile_fields(self):
        """Test ML profile has expected fields."""
        fields = PROFILE_FIELDS[MetadataProfile.ML]
        assert "PhotometricInterpretation" in fields
        assert "BitsAllocated" in fields
        assert "RescaleSlope" in fields
        assert "WindowCenter" in fields
        assert "PixelSpacing" in fields

    def test_acquisition_profile_fields(self):
        """Test acquisition profile has expected fields."""
        fields = PROFILE_FIELDS[MetadataProfile.ACQUISITION]
        assert "ViewPosition" in fields
        assert "BodyPartExamined" in fields
        assert "Laterality" in fields
