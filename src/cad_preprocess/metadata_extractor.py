"""
Metadata Extractor Module for CAD Preprocess.

This module provides profile-based extraction of DICOM metadata.
Profiles allow selective extraction of relevant metadata fields
for different use cases.

Profiles:
- minimal: Basic identification (SOPInstanceUID, Modality, dimensions)
- patient: Patient demographics (sex, age, size, weight)
- geometry: Spatial information (spacing, orientation, position)
- ml: Machine learning relevant fields (bit depth, rescale, windowing)
- acquisition: Acquisition parameters (view, body part, laterality)

Missing fields are handled gracefully by returning None.
Output format is JSON-serializable dictionary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pydicom
from pydicom.dataset import Dataset

# Configure module logger
logger = logging.getLogger(__name__)


class MetadataProfile(str, Enum):
    """Available metadata extraction profiles."""

    MINIMAL = "minimal"
    PATIENT = "patient"
    GEOMETRY = "geometry"
    ML = "ml"
    ACQUISITION = "acquisition"


# Profile definitions mapping profile names to DICOM tag names
PROFILE_FIELDS: Dict[MetadataProfile, List[str]] = {
    MetadataProfile.MINIMAL: [
        "SOPInstanceUID",
        "Modality",
        "Rows",
        "Columns",
    ],
    MetadataProfile.PATIENT: [
        "PatientSex",
        "PatientAge",
        "PatientSize",
        "PatientWeight",
    ],
    MetadataProfile.GEOMETRY: [
        "Rows",
        "Columns",
        "PixelSpacing",
        "ImageOrientationPatient",
        "ImagePositionPatient",
    ],
    MetadataProfile.ML: [
        "PhotometricInterpretation",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "RescaleSlope",
        "RescaleIntercept",
        "WindowCenter",
        "WindowWidth",
        "PixelSpacing",
    ],
    MetadataProfile.ACQUISITION: [
        "ViewPosition",
        "BodyPartExamined",
        "Laterality",
    ],
}

# Extended fields that can be requested in addition to profiles
EXTENDED_FIELDS: List[str] = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Manufacturer",
    "ManufacturerModelName",
    "InstitutionName",
    "StationName",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AccessionNumber",
    "StudyDescription",
    "SeriesDescription",
    "LossyImageCompression",
    "BurnedInAnnotation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "NumberOfFrames",
    "SliceThickness",
    "SpacingBetweenSlices",
    "KVP",
    "ExposureTime",
    "XRayTubeCurrent",
    "Exposure",
    "FilterType",
    "ConvolutionKernel",
    "PatientPosition",
    "ImageType",
    "SOPClassUID",
    "TransferSyntaxUID",
]


def convert_value_to_json_serializable(value: Any) -> Any:
    """
    Convert a DICOM value to a JSON-serializable format.

    Handles:
    - MultiValue sequences -> list
    - PersonName -> string
    - bytes -> None (binary data not serializable)
    - numpy types -> Python native types
    - Other types -> string representation

    Args:
        value: DICOM attribute value.

    Returns:
        JSON-serializable value.
    """
    if value is None:
        return None

    # Handle pydicom MultiValue (list-like)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [convert_value_to_json_serializable(v) for v in value]
        except (TypeError, ValueError):
            return str(value)

    # Handle PersonName
    if hasattr(value, "family_name"):
        return str(value)

    # Handle bytes (not JSON serializable)
    if isinstance(value, bytes):
        return None

    # Handle numpy types
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    # Handle basic types
    if isinstance(value, (int, float, str, bool)):
        return value

    # Fallback to string representation
    try:
        return str(value)
    except Exception:
        return None


def extract_field(ds: Dataset, field_name: str) -> Any:
    """
    Extract a single field from a DICOM dataset.

    Args:
        ds: pydicom Dataset object.
        field_name: Name of the DICOM attribute to extract.

    Returns:
        Extracted value (JSON-serializable) or None if not present.
    """
    try:
        if hasattr(ds, field_name):
            value = getattr(ds, field_name)
            return convert_value_to_json_serializable(value)
        return None
    except Exception as e:
        logger.debug(f"Could not extract field {field_name}: {e}")
        return None


def get_profile_fields(profiles: List[MetadataProfile]) -> Set[str]:
    """
    Get the union of all fields from specified profiles.

    Args:
        profiles: List of metadata profiles.

    Returns:
        Set of field names.
    """
    fields: Set[str] = set()
    for profile in profiles:
        if profile in PROFILE_FIELDS:
            fields.update(PROFILE_FIELDS[profile])
    return fields


@dataclass
class ExtractionResult:
    """Result of metadata extraction."""

    metadata: Dict[str, Any]
    source_file: Optional[Path] = None
    profiles_used: List[str] = field(default_factory=list)
    fields_extracted: int = 0
    fields_missing: int = 0
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata,
            "source_file": str(self.source_file) if self.source_file else None,
            "profiles_used": self.profiles_used,
            "fields_extracted": self.fields_extracted,
            "fields_missing": self.fields_missing,
            "success": self.success,
            "error_message": self.error_message,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def extract_metadata(
    ds: Dataset,
    profiles: Optional[List[MetadataProfile]] = None,
    additional_fields: Optional[List[str]] = None,
    include_all_profiles: bool = False,
) -> Dict[str, Any]:
    """
    Extract metadata from a DICOM dataset based on profiles.

    Args:
        ds: pydicom Dataset object.
        profiles: List of profiles to extract. Defaults to [MINIMAL].
        additional_fields: Additional field names to extract.
        include_all_profiles: If True, include all available profiles.

    Returns:
        Dictionary of extracted metadata.
    """
    # Determine which profiles to use
    if include_all_profiles:
        profiles = list(MetadataProfile)
    elif profiles is None:
        profiles = [MetadataProfile.MINIMAL]

    # Get all fields from profiles
    fields = get_profile_fields(profiles)

    # Add any additional fields
    if additional_fields:
        fields.update(additional_fields)

    # Extract each field
    metadata: Dict[str, Any] = {}
    for field_name in sorted(fields):
        metadata[field_name] = extract_field(ds, field_name)

    return metadata


def extract_metadata_from_file(
    file_path: Union[Path, str],
    profiles: Optional[List[MetadataProfile]] = None,
    additional_fields: Optional[List[str]] = None,
    include_all_profiles: bool = False,
) -> ExtractionResult:
    """
    Extract metadata from a DICOM file.

    Args:
        file_path: Path to the DICOM file.
        profiles: List of profiles to extract.
        additional_fields: Additional field names to extract.
        include_all_profiles: If True, include all available profiles.

    Returns:
        ExtractionResult with extracted metadata.
    """
    file_path = Path(file_path).resolve()

    # Determine profiles
    if include_all_profiles:
        profiles_to_use = list(MetadataProfile)
    else:
        profiles_to_use = profiles or [MetadataProfile.MINIMAL]

    try:
        # Load DICOM (stop before pixels for efficiency)
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)

        # Extract metadata
        metadata = extract_metadata(
            ds,
            profiles=profiles_to_use,
            additional_fields=additional_fields,
            include_all_profiles=include_all_profiles,
        )

        # Count extracted vs missing
        fields_extracted = sum(1 for v in metadata.values() if v is not None)
        fields_missing = sum(1 for v in metadata.values() if v is None)

        return ExtractionResult(
            metadata=metadata,
            source_file=file_path,
            profiles_used=[p.value for p in profiles_to_use],
            fields_extracted=fields_extracted,
            fields_missing=fields_missing,
            success=True,
        )

    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {type(e).__name__}: {e}")
        return ExtractionResult(
            metadata={},
            source_file=file_path,
            profiles_used=[p.value for p in profiles_to_use],
            success=False,
            error_message=f"{type(e).__name__}: {e}",
        )


def save_metadata(
    result: ExtractionResult,
    output_path: Union[Path, str],
    include_extraction_info: bool = False,
) -> bool:
    """
    Save extracted metadata to a JSON file.

    Args:
        result: ExtractionResult to save.
        output_path: Path for output JSON file.
        include_extraction_info: If True, include extraction metadata.

    Returns:
        True if save was successful.
    """
    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if include_extraction_info:
            data = result.to_dict()
        else:
            data = result.metadata

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved metadata to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save metadata to {output_path}: {type(e).__name__}: {e}")
        return False


class MetadataExtractor:
    """
    Profile-based metadata extractor for DICOM files.

    This class provides a convenient API for extracting metadata from
    DICOM files using predefined profiles. Each profile contains a set
    of relevant fields for a specific use case.

    Profiles:
    - minimal: SOPInstanceUID, Modality, Rows, Columns
    - patient: PatientSex, PatientAge, PatientSize, PatientWeight
    - geometry: Rows, Columns, PixelSpacing, orientation, position
    - ml: Bit depth, rescale parameters, windowing, pixel spacing
    - acquisition: ViewPosition, BodyPartExamined, Laterality

    Missing fields are returned as None (graceful skip).

    Example:
        >>> extractor = MetadataExtractor(profiles=["ml", "geometry"])
        >>> result = extractor.extract("/path/to/image.dcm")
        >>> print(result.metadata)
    """

    def __init__(
        self,
        profiles: Optional[List[Union[MetadataProfile, str]]] = None,
        additional_fields: Optional[List[str]] = None,
        include_all_profiles: bool = False,
    ) -> None:
        """
        Initialize the metadata extractor.

        Args:
            profiles: List of profiles to use. Can be MetadataProfile enum
                     or string values. Defaults to ["minimal"].
            additional_fields: Additional DICOM fields to extract.
            include_all_profiles: If True, extract all available profiles.
        """
        self.include_all_profiles = include_all_profiles

        # Convert string profiles to enum
        if profiles is None:
            self.profiles = [MetadataProfile.MINIMAL]
        else:
            self.profiles = []
            for p in profiles:
                if isinstance(p, str):
                    self.profiles.append(MetadataProfile(p))
                else:
                    self.profiles.append(p)

        self.additional_fields = additional_fields or []

    def extract(self, input_path: Union[Path, str]) -> ExtractionResult:
        """
        Extract metadata from a DICOM file.

        Args:
            input_path: Path to the DICOM file.

        Returns:
            ExtractionResult with extracted metadata.
        """
        return extract_metadata_from_file(
            file_path=input_path,
            profiles=self.profiles,
            additional_fields=self.additional_fields,
            include_all_profiles=self.include_all_profiles,
        )

    def extract_from_dataset(self, ds: Dataset) -> Dict[str, Any]:
        """
        Extract metadata from a pydicom Dataset.

        Args:
            ds: pydicom Dataset object.

        Returns:
            Dictionary of extracted metadata.
        """
        return extract_metadata(
            ds,
            profiles=self.profiles,
            additional_fields=self.additional_fields,
            include_all_profiles=self.include_all_profiles,
        )

    def save(
        self,
        result: ExtractionResult,
        output_path: Union[Path, str],
        include_extraction_info: bool = False,
    ) -> bool:
        """
        Save extraction result to JSON file.

        Args:
            result: ExtractionResult to save.
            output_path: Path for output JSON file.
            include_extraction_info: Include extraction metadata in output.

        Returns:
            True if save was successful.
        """
        return save_metadata(result, output_path, include_extraction_info)

    def get_fields(self) -> Set[str]:
        """
        Get all fields that will be extracted.

        Returns:
            Set of field names.
        """
        fields = get_profile_fields(self.profiles)
        fields.update(self.additional_fields)
        return fields

    @staticmethod
    def available_profiles() -> List[str]:
        """
        Get list of available profile names.

        Returns:
            List of profile name strings.
        """
        return [p.value for p in MetadataProfile]

    @staticmethod
    def get_profile_fields_list(profile: Union[MetadataProfile, str]) -> List[str]:
        """
        Get fields for a specific profile.

        Args:
            profile: Profile name or enum.

        Returns:
            List of field names in the profile.
        """
        if isinstance(profile, str):
            profile = MetadataProfile(profile)
        return PROFILE_FIELDS.get(profile, [])

    @staticmethod
    def get_extended_fields() -> List[str]:
        """
        Get list of all extended fields available for extraction.

        Returns:
            List of extended field names.
        """
        return EXTENDED_FIELDS.copy()

    def __repr__(self) -> str:
        profiles_str = [p.value for p in self.profiles]
        return (
            f"MetadataExtractor(profiles={profiles_str}, "
            f"additional_fields={self.additional_fields})"
        )
