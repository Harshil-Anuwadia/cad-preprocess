"""
DICOM Diagnostics Module.

This module provides tools to analyze DICOM files, check their health,
identify missing metadata, and detect corruption in pixel data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

import pydicom
from pydicom.errors import InvalidDicomError


@dataclass
class Issue:
    severity: str  # "ERROR", "WARNING", "INFO"
    description: str
    suggestion: str


@dataclass
class FileDiagnosis:
    file_path: Path
    is_dicom: bool = False
    health_status: str = "UNKNOWN"  # "HEALTHY", "WARNINGS", "CORRUPTED"
    metadata: Dict[str, str] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)


class DicomAnalyzer:
    """Analyzer for identifying issues in DICOM files."""

    def analyze(self, path: Path) -> FileDiagnosis:
        """Analyze a single DICOM file."""
        diagnosis = FileDiagnosis(file_path=path)

        if not path.exists():
            diagnosis.issues.append(
                Issue("ERROR", "File does not exist.", "Check the file path.")
            )
            diagnosis.health_status = "ERROR"
            return diagnosis

        if not path.is_file():
            diagnosis.issues.append(
                Issue("ERROR", "Path is not a file.", "Provide a path to a valid file.")
            )
            diagnosis.health_status = "ERROR"
            return diagnosis

        # 1. Try to read metadata only
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            diagnosis.is_dicom = True
        except InvalidDicomError:
            diagnosis.issues.append(
                Issue(
                    "ERROR",
                    "Not a valid DICOM file (missing 'DICM' marker).",
                    "Ensure the file is a standard DICOM. If it lacks a preamble, it might be unreadable.",
                )
            )
            diagnosis.health_status = "CORRUPTED"
            return diagnosis
        except Exception as e:
            diagnosis.issues.append(
                Issue(
                    "ERROR",
                    f"Failed to parse metadata: {e}",
                    "File might be severely corrupted or have an unsupported format.",
                )
            )
            diagnosis.health_status = "CORRUPTED"
            return diagnosis

        # Extract basic info
        ts_uid = getattr(ds.file_meta, "TransferSyntaxUID", None)
        diagnosis.metadata["PatientID"] = str(getattr(ds, "PatientID", "UNKNOWN"))
        diagnosis.metadata["Modality"] = str(getattr(ds, "Modality", "UNKNOWN"))
        diagnosis.metadata["TransferSyntax"] = str(ts_uid.name) if ts_uid else "UNKNOWN"
        diagnosis.metadata["TransferSyntaxUID"] = str(ts_uid) if ts_uid else "UNKNOWN"
        diagnosis.metadata["Rows"] = str(getattr(ds, "Rows", "UNKNOWN"))
        diagnosis.metadata["Columns"] = str(getattr(ds, "Columns", "UNKNOWN"))
        diagnosis.metadata["PhotometricInterpretation"] = str(
            getattr(ds, "PhotometricInterpretation", "UNKNOWN")
        )

        # Multiframe check
        num_frames = getattr(ds, "NumberOfFrames", 1)
        try:
            num_frames = int(num_frames)
        except (ValueError, TypeError):
            num_frames = 1
            
        if num_frames > 1:
            diagnosis.issues.append(
                Issue(
                    "WARNING",
                    f"File contains multiple frames ({num_frames}).",
                    "Ensure your preprocessing pipeline is configured to handle multi-frame DICOMs.",
                )
            )

        # 2. Try to extract pixel data (full read)
        try:
            ds_full = pydicom.dcmread(str(path))
            
            if "PixelData" not in ds_full:
                diagnosis.issues.append(
                    Issue(
                        "ERROR",
                        "Missing PixelData tag.",
                        "This file contains no image data. It might be a structured report (SR) or presentation state (PR).",
                    )
                )
                diagnosis.health_status = "CORRUPTED"
                return diagnosis

            try:
                pixels = ds_full.pixel_array
                diagnosis.metadata["PixelShape"] = str(pixels.shape)
                diagnosis.metadata["PixelDtype"] = str(pixels.dtype)
            except Exception as e:
                error_msg = str(e)
                if "No available image handler" in error_msg or "supported Transfer Syntax" in error_msg:
                    diagnosis.issues.append(
                        Issue(
                            "ERROR",
                            f"Unsupported compression: {diagnosis.metadata['TransferSyntax']}",
                            "Install decompression plugins: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm",
                        )
                    )
                elif "bytes of pixel data is less than expected" in error_msg:
                    diagnosis.issues.append(
                        Issue(
                            "ERROR",
                            f"Pixel data corrupted or truncated: {error_msg}",
                            "The file is corrupted. Re-download or re-export the file from the source system.",
                        )
                    )
                else:
                    diagnosis.issues.append(
                        Issue(
                            "ERROR",
                            f"Failed to decode pixel data: {e}",
                            "File might be corrupted or using an undocumented proprietary format.",
                        )
                    )
        except Exception as e:
            diagnosis.issues.append(
                Issue(
                    "ERROR",
                    f"Failed to read file completely: {e}",
                    "File is corrupted at the binary level.",
                )
            )

        # 3. Warning checks (Windowing)
        if "WindowCenter" not in ds or "WindowWidth" not in ds:
            diagnosis.issues.append(
                Issue(
                    "WARNING",
                    "Missing DICOM Windowing tags (WindowCenter/WindowWidth).",
                    "Use `--window-center`/`--window-width` manually, or use `--normalization min_max`.",
                )
            )

        # Determine overall health
        if any(i.severity == "ERROR" for i in diagnosis.issues):
            diagnosis.health_status = "CORRUPTED"
        elif any(i.severity == "WARNING" for i in diagnosis.issues):
            diagnosis.health_status = "WARNINGS"
        else:
            diagnosis.health_status = "HEALTHY"

        return diagnosis
