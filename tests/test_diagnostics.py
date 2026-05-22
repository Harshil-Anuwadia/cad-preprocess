import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian

from cad_preprocess.diagnostics import DicomAnalyzer, Issue, FileDiagnosis

@pytest.fixture
def dummy_dicom_path(tmp_path):
    # Create a minimal valid DICOM file
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = '1.2.3'
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = '1.2.3.4'

    ds = FileDataset(str(tmp_path / "test.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "123456"
    ds.Modality = "CT"
    ds.Rows = 512
    ds.Columns = 512
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    # We won't add PixelData to simulate an error for now, 
    # but let's just make it a basic test structure.
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    ds.save_as(str(tmp_path / "test.dcm"))
    return tmp_path / "test.dcm"


def test_analyze_non_existent_file():
    analyzer = DicomAnalyzer()
    diag = analyzer.analyze(Path("does_not_exist.dcm"))
    assert diag.health_status == "ERROR"
    assert len(diag.issues) == 1
    assert diag.issues[0].description == "File does not exist."

def test_analyze_not_a_file(tmp_path):
    analyzer = DicomAnalyzer()
    diag = analyzer.analyze(tmp_path)
    assert diag.health_status == "ERROR"
    assert len(diag.issues) == 1
    assert diag.issues[0].description == "Path is not a file."

def test_analyze_invalid_dicom(tmp_path):
    bad_file = tmp_path / "bad.dcm"
    bad_file.write_text("This is not a DICOM file")
    
    analyzer = DicomAnalyzer()
    diag = analyzer.analyze(bad_file)
    
    assert diag.health_status == "CORRUPTED"
    assert "Not a valid DICOM file" in diag.issues[0].description
    assert diag.is_dicom is False

def test_analyze_missing_pixel_data(dummy_dicom_path):
    # Our dummy dicom doesn't have PixelData
    analyzer = DicomAnalyzer()
    diag = analyzer.analyze(dummy_dicom_path)
    
    assert diag.is_dicom is True
    assert diag.health_status == "CORRUPTED"
    assert diag.metadata["PatientID"] == "123456"
    
    error_issues = [i for i in diag.issues if i.severity == "ERROR"]
    assert len(error_issues) > 0
    assert "Missing PixelData tag" in error_issues[0].description

@patch('cad_preprocess.diagnostics.pydicom.dcmread')
def test_analyze_healthy(mock_dcmread, tmp_path):
    # Mock the return values for stop_before_pixels=True and full read
    mock_ds = MagicMock()
    mock_ds.file_meta.TransferSyntaxUID.name = "Implicit VR Little Endian"
    mock_ds.PatientID = "123"
    mock_ds.Modality = "CR"
    mock_ds.Rows = 100
    mock_ds.Columns = 100
    mock_ds.PhotometricInterpretation = "MONOCHROME2"
    mock_ds.__contains__.return_value = True # Simulate containing PixelData, WindowCenter, WindowWidth
    mock_ds.WindowCenter = 50
    mock_ds.WindowWidth = 100
    
    import numpy as np
    mock_ds.pixel_array = np.zeros((100, 100), dtype=np.uint16)
    
    mock_dcmread.return_value = mock_ds
    
    test_file = tmp_path / "healthy.dcm"
    test_file.touch()
    
    analyzer = DicomAnalyzer()
    diag = analyzer.analyze(test_file)
    
    assert diag.health_status == "HEALTHY"
    assert len(diag.issues) == 0
    assert diag.metadata["PixelShape"] == "(100, 100)"

