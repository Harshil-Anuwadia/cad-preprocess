"""Tests for the integration module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cad_preprocess.integration import (
    BatchResult,
    CADPreprocessor,
    PreprocessResult,
    ProcessingManifest,
    get_config_hash,
    preprocess,
    verify_preprocessing_consistency,
)
from cad_preprocess.config import Config


class TestProcessingManifest:
    """Tests for ProcessingManifest class."""

    def test_manifest_creation(self) -> None:
        """Test manifest is created with defaults."""
        manifest = ProcessingManifest()

        assert manifest.manifest_id != ""
        assert manifest.created_at is not None
        assert manifest.processed_files == []

    def test_add_file(self) -> None:
        """Test adding file to manifest."""
        manifest = ProcessingManifest()
        manifest.add_file(
            input_file=Path("/input/test.dcm"),
            output_file=Path("/output/test.png"),
            metadata_file=Path("/output/test.json"),
            success=True,
        )

        assert len(manifest.processed_files) == 1
        assert manifest.processed_files[0]["success"] is True
        assert "test.dcm" in manifest.processed_files[0]["input"]

    def test_add_failed_file(self) -> None:
        """Test adding failed file to manifest."""
        manifest = ProcessingManifest()
        manifest.add_file(
            input_file=Path("/input/bad.dcm"),
            output_file=None,
            metadata_file=None,
            success=False,
            error_message="Invalid DICOM",
        )

        assert len(manifest.processed_files) == 1
        assert manifest.processed_files[0]["success"] is False
        assert manifest.processed_files[0]["error"] == "Invalid DICOM"

    def test_to_dict(self) -> None:
        """Test manifest serialization."""
        manifest = ProcessingManifest(
            config_hash="abc123",
            input_path="/input",
            output_path="/output",
            version="1.0.0",
        )

        data = manifest.to_dict()

        assert data["config_hash"] == "abc123"
        assert data["input_path"] == "/input"
        assert data["output_path"] == "/output"
        assert data["version"] == "1.0.0"
        assert "manifest_id" in data
        assert "created_at" in data

    def test_save_and_load(self, tmp_path) -> None:
        """Test manifest save and load."""
        manifest = ProcessingManifest(
            config_hash="test_hash",
            input_path="/input",
            output_path=str(tmp_path),
        )
        manifest.add_file(
            input_file=Path("/input/test.dcm"),
            output_file=Path(tmp_path / "test.png"),
            metadata_file=Path(tmp_path / "test.json"),
            success=True,
        )

        # Save
        manifest_path = manifest.save(tmp_path)
        assert manifest_path.exists()

        # Load
        loaded = ProcessingManifest.load(manifest_path)
        assert loaded.config_hash == "test_hash"
        assert len(loaded.processed_files) == 1


class TestPreprocessResult:
    """Tests for PreprocessResult class."""

    def test_successful_result(self) -> None:
        """Test successful preprocessing result."""
        image = np.zeros((512, 512), dtype=np.uint8)
        result = PreprocessResult(
            success=True,
            image=image,
            metadata={"SOPInstanceUID": "1.2.3"},
            output_image_path=Path("/output/test.png"),
        )

        assert result.success
        assert result.image is not None
        assert result.metadata["SOPInstanceUID"] == "1.2.3"

    def test_failed_result(self) -> None:
        """Test failed preprocessing result."""
        result = PreprocessResult(
            success=False,
            error_message="Invalid DICOM file",
        )

        assert not result.success
        assert result.image is None
        assert result.error_message == "Invalid DICOM file"

    def test_to_dict(self) -> None:
        """Test result serialization."""
        image = np.zeros((256, 256), dtype=np.uint8)
        result = PreprocessResult(
            success=True,
            image=image,
            metadata={"key": "value"},
            original_path=Path("/input/test.dcm"),
            processing_time=1.5,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["has_image"] is True
        assert data["image_shape"] == [256, 256]
        assert data["processing_time"] == 1.5


class TestBatchResult:
    """Tests for BatchResult class."""

    def test_batch_counts(self) -> None:
        """Test batch result counting."""
        results = [
            PreprocessResult(success=True, image=np.zeros((10, 10))),
            PreprocessResult(success=True, image=np.zeros((10, 10))),
            PreprocessResult(success=False, error_message="Error"),
        ]
        batch = BatchResult(success=True, results=results)

        assert batch.total_processed == 2
        assert batch.total_failed == 1

    def test_summary(self) -> None:
        """Test batch summary generation."""
        batch = BatchResult(
            success=True,
            results=[PreprocessResult(success=True)],
            config_hash="abcdef1234567890",
        )

        summary = batch.summary()

        assert "BATCH PROCESSING SUMMARY" in summary
        assert "Successful:" in summary
        assert "abcdef12" in summary  # First 16 chars of hash


class TestCADPreprocessor:
    """Tests for CADPreprocessor class."""

    def test_from_config_default(self) -> None:
        """Test creating preprocessor with default config."""
        processor = CADPreprocessor.from_config()

        assert processor.config is not None
        assert processor.config_hash != ""

    def test_from_config_file(self, tmp_path) -> None:
        """Test creating preprocessor from config file."""
        # Create config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
preprocessing:
  normalization: min_max
  windowing:
    strategy: use_dicom_window
  resizing:
    target_height: 256
    target_width: 256
""")

        processor = CADPreprocessor.from_config(config_path)

        assert processor.config.preprocessing.normalization == "min_max"
        assert processor.config.preprocessing.resizing.target_height == 256

    def test_config_hash_reproducibility(self) -> None:
        """Test that config hash is reproducible."""
        config = Config.default()

        proc1 = CADPreprocessor(config)
        proc2 = CADPreprocessor(config)

        assert proc1.config_hash == proc2.config_hash

    def test_config_hash_changes_with_config(self) -> None:
        """Test that config hash changes with different config."""
        config1 = Config.default()
        config2 = Config.default()
        config2.preprocessing.normalization = "z_score"

        proc1 = CADPreprocessor(config1)
        proc2 = CADPreprocessor(config2)

        assert proc1.config_hash != proc2.config_hash

    def test_verify_config_hash(self) -> None:
        """Test config hash verification."""
        processor = CADPreprocessor.from_config()
        expected_hash = processor.config_hash

        assert processor.verify_config_hash(expected_hash)
        assert not processor.verify_config_hash("wrong_hash")

    @patch("cad_preprocess.integration.PreprocessingEngine")
    @patch("cad_preprocess.integration.MetadataExtractor")
    def test_process_file_success(
        self,
        mock_extractor,
        mock_engine,
    ) -> None:
        """Test successful file processing."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.image = np.zeros((512, 512), dtype=np.uint8)
        mock_engine.return_value.process.return_value = mock_result

        mock_meta = MagicMock()
        mock_meta.success = True
        mock_meta.metadata = {"SOPInstanceUID": "1.2.3"}
        mock_extractor.return_value.extract.return_value = mock_meta

        processor = CADPreprocessor.from_config(write_outputs=False)
        result = processor.process_file(Path("/fake/test.dcm"))

        assert result.success
        assert result.image is not None
        assert result.metadata["SOPInstanceUID"] == "1.2.3"

    @patch("cad_preprocess.integration.PreprocessingEngine")
    def test_process_file_failure(self, mock_engine) -> None:
        """Test failed file processing."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Invalid DICOM"
        mock_engine.return_value.process.return_value = mock_result

        processor = CADPreprocessor.from_config(write_outputs=False)
        result = processor.process_file(Path("/fake/bad.dcm"))

        assert not result.success
        assert result.error_message == "Invalid DICOM"

    @patch("cad_preprocess.integration.InputHandler")
    @patch("cad_preprocess.integration.PreprocessingEngine")
    @patch("cad_preprocess.integration.MetadataExtractor")
    def test_process_directory(
        self,
        mock_extractor,
        mock_engine,
        mock_handler,
        tmp_path,
    ) -> None:
        """Test directory processing."""
        # Setup discovery mock
        mock_discovery = MagicMock()
        mock_discovery.total_discovered = 2
        mock_discovery.total_valid = 2
        mock_discovery.total_skipped = 0
        mock_discovery.valid_files = [
            Path("/fake/file1.dcm"),
            Path("/fake/file2.dcm"),
        ]
        mock_handler.return_value.discover.return_value = mock_discovery

        # Setup processing mock
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.image = np.zeros((512, 512), dtype=np.uint8)
        mock_engine.return_value.process.return_value = mock_result

        mock_meta = MagicMock()
        mock_meta.success = True
        mock_meta.metadata = {"SOPInstanceUID": "1.2.3"}
        mock_extractor.return_value.extract.return_value = mock_meta

        processor = CADPreprocessor.from_config(
            output_dir=tmp_path,
            write_outputs=False,
            create_manifest=False,
        )
        batch = processor.process_directory(tmp_path)

        assert batch.success
        assert len(batch.results) == 2
        assert batch.total_processed == 2

    @patch("cad_preprocess.integration.InputHandler")
    @patch("cad_preprocess.integration.PreprocessingEngine")
    @patch("cad_preprocess.integration.MetadataExtractor")
    def test_iterate_files(
        self,
        mock_extractor,
        mock_engine,
        mock_handler,
        tmp_path,
    ) -> None:
        """Test file iteration."""
        mock_discovery = MagicMock()
        mock_discovery.valid_files = [Path("/fake/file1.dcm")]
        mock_handler.return_value.discover.return_value = mock_discovery

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.image = np.zeros((512, 512))
        mock_engine.return_value.process.return_value = mock_result

        mock_meta = MagicMock()
        mock_meta.success = True
        mock_meta.metadata = {}
        mock_extractor.return_value.extract.return_value = mock_meta

        processor = CADPreprocessor.from_config(write_outputs=False)
        results = list(processor.iterate_files(tmp_path))

        assert len(results) == 1
        assert results[0].success


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_config_hash_from_path(self, tmp_path) -> None:
        """Test get_config_hash with file path."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
preprocessing:
  normalization: min_max
""")

        hash1 = get_config_hash(config_path)
        hash2 = get_config_hash(config_path)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_get_config_hash_from_config(self) -> None:
        """Test get_config_hash with Config object."""
        config = Config.default()
        hash_value = get_config_hash(config)

        assert len(hash_value) == 64

    def test_verify_preprocessing_consistency_same(self, tmp_path) -> None:
        """Test consistency verification with same configs."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
preprocessing:
  normalization: min_max
""")

        assert verify_preprocessing_consistency(config_path, config_path)

    def test_verify_preprocessing_consistency_different(self, tmp_path) -> None:
        """Test consistency verification with different configs."""
        config1 = tmp_path / "config1.yaml"
        config1.write_text("""
preprocessing:
  normalization: min_max
""")

        config2 = tmp_path / "config2.yaml"
        config2.write_text("""
preprocessing:
  normalization: z_score
""")

        assert not verify_preprocessing_consistency(config1, config2)

    @patch("cad_preprocess.integration.CADPreprocessor")
    def test_preprocess_single_file(self, mock_processor_class, tmp_path) -> None:
        """Test preprocess() with single file."""
        mock_processor = MagicMock()
        mock_result = PreprocessResult(success=True)
        mock_processor.process_file.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        # Create dummy file
        test_file = tmp_path / "test.dcm"
        test_file.touch()

        result = preprocess(test_file, write_outputs=False)

        assert result.success
        mock_processor.process_file.assert_called_once()

    @patch("cad_preprocess.integration.CADPreprocessor")
    def test_preprocess_directory(self, mock_processor_class, tmp_path) -> None:
        """Test preprocess() with directory."""
        mock_processor = MagicMock()
        mock_batch = BatchResult(success=True, results=[])
        mock_processor.process_directory.return_value = mock_batch
        mock_processor_class.return_value = mock_processor

        result = preprocess(tmp_path, write_outputs=False)

        assert result.success
        mock_processor.process_directory.assert_called_once()

    def test_preprocess_nonexistent_path(self) -> None:
        """Test preprocess() with non-existent path."""
        result = preprocess("/nonexistent/path", write_outputs=False)

        assert not result.success
        assert "does not exist" in result.error_message


class TestIntegrationGuarantees:
    """Tests for integration guarantees."""

    def test_identical_preprocessing_guarantee(self, tmp_path) -> None:
        """Test that same config produces identical hashes."""
        # Simulate training config
        train_config = tmp_path / "train_config.yaml"
        train_config.write_text("""
preprocessing:
  normalization: min_max
  windowing:
    strategy: fixed_window
    window_center: 40
    window_width: 400
  resizing:
    target_height: 512
    target_width: 512
""")

        # Simulate inference config (exact same)
        inference_config = tmp_path / "inference_config.yaml"
        inference_config.write_text("""
preprocessing:
  normalization: min_max
  windowing:
    strategy: fixed_window
    window_center: 40
    window_width: 400
  resizing:
    target_height: 512
    target_width: 512
""")

        train_hash = get_config_hash(train_config)
        inference_hash = get_config_hash(inference_config)

        assert train_hash == inference_hash, "Training and inference must use identical preprocessing"

    def test_manifest_traceability(self, tmp_path) -> None:
        """Test that manifest provides traceability."""
        manifest = ProcessingManifest(
            config_hash="abc123",
            input_path="/data/train",
            output_path="/data/preprocessed",
            version="1.0.0",
        )

        # Add files
        for i in range(3):
            manifest.add_file(
                input_file=Path(f"/data/train/file_{i}.dcm"),
                output_file=Path(f"/data/preprocessed/file_{i}.png"),
                metadata_file=Path(f"/data/preprocessed/file_{i}.json"),
                success=True,
            )

        # Save and verify
        manifest_path = manifest.save(tmp_path)

        with open(manifest_path) as f:
            data = json.load(f)

        # Verify traceability fields
        assert data["config_hash"] == "abc123"
        assert data["version"] == "1.0.0"
        assert len(data["processed_files"]) == 3

        # Each file should be traceable
        for file_record in data["processed_files"]:
            assert "input" in file_record
            assert "output" in file_record
            assert "timestamp" in file_record

    def test_reproducible_configuration(self) -> None:
        """Test that configuration is reproducible."""
        config1 = Config.default()
        config1.preprocessing.normalization = "min_max"
        config1.preprocessing.resizing.target_height = 256

        config2 = Config.default()
        config2.preprocessing.normalization = "min_max"
        config2.preprocessing.resizing.target_height = 256

        proc1 = CADPreprocessor(config1, write_outputs=False)
        proc2 = CADPreprocessor(config2, write_outputs=False)

        # Same config should produce same hash
        assert proc1.config_hash == proc2.config_hash

        # Modifying config should change hash
        config2.preprocessing.normalization = "z_score"
        proc3 = CADPreprocessor(config2, write_outputs=False)

        assert proc1.config_hash != proc3.config_hash
