"""
Tests for the Output Writer module.

These tests verify:
- Directory structure creation
- Image writing with naming policies
- Metadata JSON writing
- Overwrite policy handling
- Batch write operations
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from cad_preprocess.output_writer import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_METADATA_DIR,
    BatchWriteResult,
    NamingPolicy,
    OutputConfig,
    OutputWriter,
    OverwritePolicy,
    WriteResult,
    generate_filename,
    resolve_output_path,
    write_image,
    write_metadata,
)


class TestNamingPolicy:
    """Tests for NamingPolicy enum."""

    def test_sop_instance_uid(self):
        """Test SOP instance UID policy."""
        assert NamingPolicy.SOP_INSTANCE_UID.value == "sop_instance_uid"

    def test_original_filename(self):
        """Test original filename policy."""
        assert NamingPolicy.ORIGINAL_FILENAME.value == "original_filename"

    def test_sequential(self):
        """Test sequential policy."""
        assert NamingPolicy.SEQUENTIAL.value == "sequential"


class TestOverwritePolicy:
    """Tests for OverwritePolicy enum."""

    def test_skip(self):
        """Test skip policy."""
        assert OverwritePolicy.SKIP.value == "skip"

    def test_overwrite(self):
        """Test overwrite policy."""
        assert OverwritePolicy.OVERWRITE.value == "overwrite"

    def test_rename(self):
        """Test rename policy."""
        assert OverwritePolicy.RENAME.value == "rename"

    def test_error(self):
        """Test error policy."""
        assert OverwritePolicy.ERROR.value == "error"


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_config(self, tmp_path):
        """Test default configuration."""
        config = OutputConfig(output_root=tmp_path)
        assert config.images_subdir == DEFAULT_IMAGES_DIR
        assert config.metadata_subdir == DEFAULT_METADATA_DIR
        assert config.logs_subdir == DEFAULT_LOGS_DIR
        assert config.naming_policy == NamingPolicy.SOP_INSTANCE_UID
        assert config.overwrite_policy == OverwritePolicy.SKIP

    def test_directory_properties(self, tmp_path):
        """Test directory path properties."""
        config = OutputConfig(output_root=tmp_path)
        assert config.images_dir == tmp_path / DEFAULT_IMAGES_DIR
        assert config.metadata_dir == tmp_path / DEFAULT_METADATA_DIR
        assert config.logs_dir == tmp_path / DEFAULT_LOGS_DIR

    def test_string_policy_conversion(self, tmp_path):
        """Test string policy conversion."""
        config = OutputConfig(
            output_root=tmp_path,
            naming_policy="sequential",
            overwrite_policy="overwrite",
        )
        assert config.naming_policy == NamingPolicy.SEQUENTIAL
        assert config.overwrite_policy == OverwritePolicy.OVERWRITE


class TestWriteResult:
    """Tests for WriteResult dataclass."""

    def test_success_is_truthy(self):
        """Test successful result is truthy."""
        result = WriteResult(success=True, file_path=Path("/test"))
        assert result
        assert bool(result) is True

    def test_failure_is_falsy(self):
        """Test failed result is falsy."""
        result = WriteResult(success=False, error_message="Error")
        assert not result
        assert bool(result) is False


class TestBatchWriteResult:
    """Tests for BatchWriteResult dataclass."""

    def test_add_success(self):
        """Test adding successful result."""
        batch = BatchWriteResult()
        batch.add(WriteResult(success=True, action="written"))
        batch.add(WriteResult(success=True, action="written"))

        assert batch.total == 2
        assert batch.written == 2
        assert batch.skipped == 0
        assert batch.errors == 0

    def test_add_skipped(self):
        """Test adding skipped result."""
        batch = BatchWriteResult()
        batch.add(WriteResult(success=True, action="skipped"))

        assert batch.total == 1
        assert batch.written == 0
        assert batch.skipped == 1

    def test_add_error(self):
        """Test adding error result."""
        batch = BatchWriteResult()
        batch.add(WriteResult(success=False, action="error"))

        assert batch.total == 1
        assert batch.written == 0
        assert batch.errors == 1


class TestGenerateFilename:
    """Tests for generate_filename function."""

    def test_sop_instance_uid_policy(self):
        """Test filename with SOP instance UID."""
        filename = generate_filename(
            sop_instance_uid="1.2.3.4.5",
            naming_policy=NamingPolicy.SOP_INSTANCE_UID,
        )
        assert filename == "1.2.3.4.5"

    def test_sop_fallback_to_original(self):
        """Test SOP policy falls back to original filename."""
        filename = generate_filename(
            sop_instance_uid=None,
            original_path=Path("/path/to/image.dcm"),
            naming_policy=NamingPolicy.SOP_INSTANCE_UID,
        )
        assert filename == "image"

    def test_original_filename_policy(self):
        """Test filename with original filename policy."""
        filename = generate_filename(
            original_path=Path("/path/to/scan_001.dcm"),
            naming_policy=NamingPolicy.ORIGINAL_FILENAME,
        )
        assert filename == "scan_001"

    def test_sequential_policy(self):
        """Test filename with sequential policy."""
        filename = generate_filename(
            sequence_number=42,
            naming_policy=NamingPolicy.SEQUENTIAL,
        )
        assert filename == "image_000042"


class TestResolveOutputPath:
    """Tests for resolve_output_path function."""

    def test_new_file(self, tmp_path):
        """Test path resolution for new file."""
        path, action = resolve_output_path(
            output_dir=tmp_path,
            filename="test",
            extension="png",
            overwrite_policy=OverwritePolicy.SKIP,
        )
        assert path == tmp_path / "test.png"
        assert action == "write"

    def test_skip_existing(self, tmp_path):
        """Test skip policy for existing file."""
        existing = tmp_path / "test.png"
        existing.touch()

        path, action = resolve_output_path(
            output_dir=tmp_path,
            filename="test",
            extension="png",
            overwrite_policy=OverwritePolicy.SKIP,
        )
        assert path == existing
        assert action == "skip"

    def test_overwrite_existing(self, tmp_path):
        """Test overwrite policy for existing file."""
        existing = tmp_path / "test.png"
        existing.touch()

        path, action = resolve_output_path(
            output_dir=tmp_path,
            filename="test",
            extension="png",
            overwrite_policy=OverwritePolicy.OVERWRITE,
        )
        assert path == existing
        assert action == "overwrite"

    def test_rename_existing(self, tmp_path):
        """Test rename policy for existing file."""
        existing = tmp_path / "test.png"
        existing.touch()

        path, action = resolve_output_path(
            output_dir=tmp_path,
            filename="test",
            extension="png",
            overwrite_policy=OverwritePolicy.RENAME,
        )
        assert path == tmp_path / "test_1.png"
        assert action == "rename"

    def test_error_existing(self, tmp_path):
        """Test error policy for existing file."""
        existing = tmp_path / "test.png"
        existing.touch()

        with pytest.raises(FileExistsError):
            resolve_output_path(
                output_dir=tmp_path,
                filename="test",
                extension="png",
                overwrite_policy=OverwritePolicy.ERROR,
            )

    def test_extension_with_dot(self, tmp_path):
        """Test extension handling with dot."""
        path, _ = resolve_output_path(
            output_dir=tmp_path,
            filename="test",
            extension=".png",
            overwrite_policy=OverwritePolicy.SKIP,
        )
        assert path.suffix == ".png"


class TestWriteImage:
    """Tests for write_image function."""

    def test_write_uint8(self, tmp_path):
        """Test writing uint8 image."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        output_path = tmp_path / "test.png"

        write_image(image, output_path)

        assert output_path.exists()
        loaded = np.array(Image.open(output_path))
        assert loaded.shape == (100, 100)

    def test_write_float_0_1(self, tmp_path):
        """Test writing float [0,1] image."""
        image = np.random.rand(100, 100)
        output_path = tmp_path / "test.png"

        write_image(image, output_path)

        assert output_path.exists()

    def test_creates_directories(self, tmp_path):
        """Test that parent directories are created."""
        image = np.zeros((100, 100), dtype=np.uint8)
        output_path = tmp_path / "nested" / "dir" / "test.png"

        write_image(image, output_path)

        assert output_path.exists()


class TestWriteMetadata:
    """Tests for write_metadata function."""

    def test_write_metadata(self, tmp_path):
        """Test writing metadata."""
        metadata = {"Modality": "CR", "Rows": 512}
        output_path = tmp_path / "test.json"

        write_metadata(metadata, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded == metadata

    def test_creates_directories(self, tmp_path):
        """Test that parent directories are created."""
        metadata = {"test": "data"}
        output_path = tmp_path / "nested" / "dir" / "test.json"

        write_metadata(metadata, output_path)

        assert output_path.exists()


class TestOutputWriter:
    """Tests for OutputWriter class."""

    def test_init_creates_directories(self, tmp_path):
        """Test initialization creates directories."""
        output_root = tmp_path / "output"
        writer = OutputWriter(output_root)

        assert (output_root / DEFAULT_IMAGES_DIR).exists()
        assert (output_root / DEFAULT_METADATA_DIR).exists()
        assert (output_root / DEFAULT_LOGS_DIR).exists()

    def test_init_with_config(self, tmp_path):
        """Test initialization with config."""
        config = OutputConfig(
            output_root=tmp_path,
            overwrite_policy=OverwritePolicy.OVERWRITE,
        )
        writer = OutputWriter(tmp_path, config=config)

        assert writer.config.overwrite_policy == OverwritePolicy.OVERWRITE

    def test_write_image_success(self, tmp_path):
        """Test successful image write."""
        writer = OutputWriter(tmp_path)
        image = np.zeros((100, 100), dtype=np.uint8)

        result = writer.write_image(image, sop_instance_uid="1.2.3.4.5")

        assert result.success
        assert result.action == "written"
        assert result.file_path.exists()
        assert result.file_path.stem == "1.2.3.4.5"

    def test_write_image_skip_existing(self, tmp_path):
        """Test image write skips existing."""
        writer = OutputWriter(tmp_path, overwrite_policy=OverwritePolicy.SKIP)
        image = np.zeros((100, 100), dtype=np.uint8)

        # Write first time
        result1 = writer.write_image(image, sop_instance_uid="1.2.3.4.5")
        assert result1.action == "written"

        # Second write should skip
        result2 = writer.write_image(image, sop_instance_uid="1.2.3.4.5")
        assert result2.action == "skipped"

    def test_write_metadata_success(self, tmp_path):
        """Test successful metadata write."""
        writer = OutputWriter(tmp_path)
        metadata = {"Modality": "CR", "Rows": 512}

        result = writer.write_metadata(metadata, sop_instance_uid="1.2.3.4.5")

        assert result.success
        assert result.file_path.exists()
        assert result.file_path.suffix == ".json"

    def test_write_metadata_uses_sop_from_metadata(self, tmp_path):
        """Test metadata write uses SOPInstanceUID from metadata."""
        writer = OutputWriter(tmp_path)
        metadata = {"SOPInstanceUID": "9.8.7.6.5", "Modality": "CR"}

        result = writer.write_metadata(metadata)

        assert result.success
        assert "9.8.7.6.5" in result.file_path.stem

    def test_write_processing_result(self, tmp_path):
        """Test writing both image and metadata."""
        writer = OutputWriter(tmp_path)
        image = np.zeros((100, 100), dtype=np.uint8)
        metadata = {"SOPInstanceUID": "1.2.3.4.5", "Modality": "CR"}

        img_result, meta_result = writer.write_processing_result(
            image, metadata, sop_instance_uid="1.2.3.4.5"
        )

        assert img_result.success
        assert meta_result.success
        # Both should have same base filename
        assert img_result.file_path.stem == meta_result.file_path.stem

    def test_write_processing_log(self, tmp_path):
        """Test writing processing log."""
        writer = OutputWriter(tmp_path)
        log_data = {"processed": 10, "errors": 1}

        result = writer.write_processing_log(log_data)

        assert result.success
        assert result.file_path.exists()
        assert "processing_log" in result.file_path.name

    def test_get_image_path(self, tmp_path):
        """Test get_image_path method."""
        writer = OutputWriter(tmp_path)
        path = writer.get_image_path("1.2.3.4.5")

        assert path == tmp_path / DEFAULT_IMAGES_DIR / "1.2.3.4.5.png"

    def test_get_metadata_path(self, tmp_path):
        """Test get_metadata_path method."""
        writer = OutputWriter(tmp_path)
        path = writer.get_metadata_path("1.2.3.4.5")

        assert path == tmp_path / DEFAULT_METADATA_DIR / "1.2.3.4.5.json"

    def test_exists_false(self, tmp_path):
        """Test exists returns False for missing files."""
        writer = OutputWriter(tmp_path)
        assert writer.exists("nonexistent") is False

    def test_exists_true(self, tmp_path):
        """Test exists returns True for existing files."""
        writer = OutputWriter(tmp_path)
        image = np.zeros((100, 100), dtype=np.uint8)
        metadata = {"Modality": "CR"}

        writer.write_image(image, sop_instance_uid="1.2.3.4.5")
        writer.write_metadata(metadata, sop_instance_uid="1.2.3.4.5")

        assert writer.exists("1.2.3.4.5") is True

    def test_get_stats(self, tmp_path):
        """Test get_stats method."""
        writer = OutputWriter(tmp_path)
        image = np.zeros((100, 100), dtype=np.uint8)

        writer.write_image(image, sop_instance_uid="1.2.3.4.5")
        writer.write_metadata({"Modality": "CR"}, sop_instance_uid="1.2.3.4.5")

        stats = writer.get_stats()
        assert stats["images_count"] == 1
        assert stats["metadata_count"] == 1

    def test_clean_requires_confirm(self, tmp_path):
        """Test clean requires confirmation."""
        writer = OutputWriter(tmp_path)
        result = writer.clean(confirm=False)
        assert result is False

    def test_clean_with_confirm(self, tmp_path):
        """Test clean with confirmation."""
        writer = OutputWriter(tmp_path)
        image = np.zeros((100, 100), dtype=np.uint8)
        writer.write_image(image, sop_instance_uid="1.2.3.4.5")

        result = writer.clean(confirm=True)

        assert result is True
        assert writer.get_stats()["images_count"] == 0

    def test_repr(self, tmp_path):
        """Test string representation."""
        writer = OutputWriter(tmp_path)
        repr_str = repr(writer)
        assert "OutputWriter" in repr_str
        assert "sop_instance_uid" in repr_str
