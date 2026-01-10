"""Tests for the API Interface module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cad_preprocess.api import (
    PreprocessingResult,
    _compute_config_hash,
    _config_from_dict,
    _load_configuration,
    preprocess,
)
from cad_preprocess.config import Config


class TestPreprocessingResult:
    """Tests for PreprocessingResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = PreprocessingResult()
        assert result.processed_count == 0
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert result.error_summary == []
        assert result.processed_files == []
        assert result.skipped_files == []
        assert result.config_hash == ""
        assert result.output_dir is None

    def test_total_files(self):
        """Test total_files property."""
        result = PreprocessingResult(
            processed_count=10,
            skipped_count=3,
            error_count=2,
        )
        assert result.total_files == 15

    def test_success_rate(self):
        """Test success_rate property."""
        result = PreprocessingResult(
            processed_count=80,
            skipped_count=10,
            error_count=10,
        )
        assert result.success_rate == 80.0

    def test_success_rate_zero_files(self):
        """Test success_rate with no files."""
        result = PreprocessingResult()
        assert result.success_rate == 0.0

    def test_errors_property(self):
        """Test errors property."""
        result = PreprocessingResult(error_count=0)
        assert result.errors is False

        result = PreprocessingResult(error_count=1)
        assert result.errors is True

    def test_repr(self):
        """Test string representation."""
        result = PreprocessingResult(
            processed_count=5,
            skipped_count=2,
            error_count=1,
        )
        repr_str = repr(result)
        assert "processed=5" in repr_str
        assert "skipped=2" in repr_str
        assert "errors=1" in repr_str

    def test_summary(self):
        """Test summary generation."""
        result = PreprocessingResult(
            processed_count=10,
            skipped_count=2,
            error_count=1,
            config_hash="abc123def456",
            output_dir="/output",
            error_summary=[
                {"file": "test.dcm", "message": "Test error"}
            ],
        )
        summary = result.summary()
        assert "Total files:" in summary
        assert "Processed:" in summary
        assert "10" in summary
        assert "ERRORS:" in summary
        assert "test.dcm" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = PreprocessingResult(
            processed_count=5,
            skipped_count=2,
            error_count=1,
            config_hash="abc123",
            output_dir="/output",
        )
        d = result.to_dict()
        assert d["processed_count"] == 5
        assert d["skipped_count"] == 2
        assert d["error_count"] == 1
        assert d["total_files"] == 8
        assert d["success_rate"] == pytest.approx(62.5)


class TestConfigLoading:
    """Tests for configuration loading functions."""

    def test_load_configuration_none(self):
        """Test loading default configuration."""
        cfg = _load_configuration(None)
        assert isinstance(cfg, Config)

    def test_load_configuration_dict(self):
        """Test loading configuration from dict."""
        config_dict = {
            "preprocessing": {
                "windowing": {
                    "strategy": "fixed_window",
                    "window_center": 40,
                    "window_width": 400,
                }
            }
        }
        cfg = _load_configuration(config_dict)
        assert isinstance(cfg, Config)
        assert cfg.preprocessing.windowing.strategy == "fixed_window"
        assert cfg.preprocessing.windowing.window_center == 40
        assert cfg.preprocessing.windowing.window_width == 400

    def test_config_from_dict_full(self):
        """Test creating config from full dictionary."""
        config_dict = {
            "preprocessing": {
                "windowing": {
                    "strategy": "fixed_window",
                    "window_center": 50,
                    "window_width": 350,
                },
                "normalization": "min_max",
                "resizing": {
                    "target_height": 256,
                    "target_width": 256,
                    "preserve_aspect_ratio": False,
                },
                "output_dtype": "float32",
            },
            "metadata": {
                "profiles": ["minimal", "ml"],
                "additional_fields": ["Modality"],
            },
            "output": {
                "naming_policy": "original_filename",
                "format": "png",
            },
        }
        cfg = _config_from_dict(config_dict)
        
        assert cfg.preprocessing.windowing.strategy == "fixed_window"
        assert cfg.preprocessing.windowing.window_center == 50
        assert cfg.preprocessing.resizing.target_height == 256
        assert cfg.preprocessing.resizing.preserve_aspect_ratio is False
        assert cfg.metadata.profiles == ["minimal", "ml"]
        assert cfg.output.naming_policy == "original_filename"

    def test_config_from_dict_partial(self):
        """Test creating config from partial dictionary."""
        config_dict = {
            "metadata": {
                "profiles": ["all"],
            }
        }
        cfg = _config_from_dict(config_dict)
        assert cfg.metadata.profiles == ["all"]
        # Other settings should use defaults
        assert cfg.preprocessing is not None


class TestConfigHash:
    """Tests for configuration hashing."""

    def test_compute_config_hash(self):
        """Test that config hash is computed."""
        cfg = Config.default()
        hash_val = _compute_config_hash(cfg)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 hex length

    def test_config_hash_deterministic(self):
        """Test that same config produces same hash."""
        cfg1 = Config.default()
        cfg2 = Config.default()
        hash1 = _compute_config_hash(cfg1)
        hash2 = _compute_config_hash(cfg2)
        assert hash1 == hash2

    def test_config_hash_changes_with_config(self):
        """Test that different configs produce different hashes."""
        cfg1 = Config.default()
        cfg2 = Config.default()
        cfg2.preprocessing.windowing.window_center = 100  # Different value
        
        hash1 = _compute_config_hash(cfg1)
        hash2 = _compute_config_hash(cfg2)
        assert hash1 != hash2


class TestPreprocessFunction:
    """Tests for the main preprocess function."""

    def test_preprocess_nonexistent_path(self):
        """Test preprocessing with nonexistent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = preprocess(
                "/nonexistent/path/to/dicoms",
                tmpdir,
            )
            # Should return result with no processed files
            assert result.processed_count == 0

    def test_preprocess_empty_directory(self):
        """Test preprocessing an empty directory."""
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                result = preprocess(input_dir, output_dir)
                assert result.processed_count == 0
                assert result.skipped_count == 0

    def test_preprocess_with_config_dict(self):
        """Test preprocessing with configuration dictionary."""
        config = {
            "preprocessing": {
                "windowing": {"strategy": "from_dicom"},
            },
            "output": {
                "format": "png",
            },
        }
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                result = preprocess(input_dir, output_dir, config=config)
                assert isinstance(result, PreprocessingResult)
                assert result.config_hash != ""

    @patch('cad_preprocess.api.PreprocessingEngine')
    @patch('cad_preprocess.api.InputHandler')
    def test_preprocess_mocked_success(self, mock_handler, mock_engine):
        """Test preprocessing with mocked components."""
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery.valid_files = []
        mock_discovery.invalid_files = []
        mock_handler.return_value.discover.return_value = mock_discovery
        
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                result = preprocess(input_dir, output_dir)
                assert isinstance(result, PreprocessingResult)
                assert result.output_dir == output_dir


class TestPreprocessIntegration:
    """Integration tests for preprocess function (require actual DICOM files)."""

    @pytest.mark.skip(reason="Requires actual DICOM files")
    def test_preprocess_real_dicom(self):
        """Test preprocessing actual DICOM files."""
        # This test would require actual DICOM files
        pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
