"""
Tests for the Configuration System module.

These tests verify:
- Configuration loading from YAML
- Configuration saving to YAML
- Default configuration values
- Configuration validation
- Settings conversion to module configs
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from cad_preprocess.config import (
    CONFIG_TEMPLATE,
    Config,
    InputSettings,
    LoggingSettings,
    LogLevel,
    MetadataSettings,
    OutputSettings,
    PreprocessingSettings,
    ResizingSettings,
    WindowingSettings,
    create_config_template,
    create_default_config_file,
    load_config,
)
from cad_preprocess.metadata_extractor import MetadataProfile
from cad_preprocess.output_writer import NamingPolicy, OverwritePolicy
from cad_preprocess.preprocessing_engine import (
    InterpolationMethod,
    NormalizationMethod,
    WindowingStrategy,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels(self):
        """Test all log levels are defined."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_to_logging_level(self):
        """Test conversion to logging module level."""
        import logging

        assert LogLevel.DEBUG.to_logging_level() == logging.DEBUG
        assert LogLevel.INFO.to_logging_level() == logging.INFO
        assert LogLevel.WARNING.to_logging_level() == logging.WARNING


class TestWindowingSettings:
    """Tests for WindowingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = WindowingSettings()
        assert settings.strategy == "use_dicom_window"
        assert settings.window_center is None
        assert settings.window_width is None

    def test_to_windowing_config(self):
        """Test conversion to WindowingConfig."""
        settings = WindowingSettings(
            strategy="fixed_window",
            window_center=40,
            window_width=400,
        )
        config = settings.to_windowing_config()
        assert config.strategy == WindowingStrategy.FIXED_WINDOW
        assert config.window_center == 40
        assert config.window_width == 400

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = WindowingSettings(strategy="use_dicom_window")
        d = settings.to_dict()
        assert d["strategy"] == "use_dicom_window"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"strategy": "fixed_window", "window_center": 50, "window_width": 350}
        settings = WindowingSettings.from_dict(data)
        assert settings.strategy == "fixed_window"
        assert settings.window_center == 50


class TestResizingSettings:
    """Tests for ResizingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ResizingSettings()
        assert settings.enabled is True
        assert settings.target_height == 1024
        assert settings.target_width == 1024
        assert settings.keep_aspect_ratio is True
        assert settings.interpolation == "bilinear"

    def test_to_resizing_config(self):
        """Test conversion to ResizingConfig."""
        settings = ResizingSettings(target_height=512, target_width=512)
        config = settings.to_resizing_config()
        assert config.target_size == (512, 512)
        assert config.interpolation == InterpolationMethod.BILINEAR

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"enabled": False, "target_height": 256}
        settings = ResizingSettings.from_dict(data)
        assert settings.enabled is False
        assert settings.target_height == 256


class TestPreprocessingSettings:
    """Tests for PreprocessingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = PreprocessingSettings()
        assert settings.normalization == "min_max"
        assert settings.output_dtype == "uint8"

    def test_to_preprocessing_config(self):
        """Test conversion to PreprocessingConfig."""
        settings = PreprocessingSettings(normalization="z_score")
        config = settings.to_preprocessing_config()
        assert config.normalization == NormalizationMethod.Z_SCORE

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "normalization": "z_score",
            "windowing": {"strategy": "use_dicom_window"},
        }
        settings = PreprocessingSettings.from_dict(data)
        assert settings.normalization == "z_score"


class TestMetadataSettings:
    """Tests for MetadataSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = MetadataSettings()
        assert settings.profiles == ["minimal"]
        assert settings.additional_fields == []
        assert settings.include_all_profiles is False

    def test_get_profiles(self):
        """Test get_profiles method."""
        settings = MetadataSettings(profiles=["minimal", "ml"])
        profiles = settings.get_profiles()
        assert MetadataProfile.MINIMAL in profiles
        assert MetadataProfile.ML in profiles

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"profiles": ["ml", "geometry"]}
        settings = MetadataSettings.from_dict(data)
        assert settings.profiles == ["ml", "geometry"]


class TestOutputSettings:
    """Tests for OutputSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = OutputSettings()
        assert settings.naming_policy == "sop_instance_uid"
        assert settings.overwrite_policy == "skip"
        assert settings.image_format == "png"

    def test_get_naming_policy(self):
        """Test get_naming_policy method."""
        settings = OutputSettings(naming_policy="sequential")
        assert settings.get_naming_policy() == NamingPolicy.SEQUENTIAL

    def test_get_overwrite_policy(self):
        """Test get_overwrite_policy method."""
        settings = OutputSettings(overwrite_policy="overwrite")
        assert settings.get_overwrite_policy() == OverwritePolicy.OVERWRITE


class TestInputSettings:
    """Tests for InputSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = InputSettings()
        assert settings.recursive is True
        assert settings.validate is True
        assert settings.check_pixel_data is True
        assert settings.check_dimensions is True

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"recursive": False, "validate": False}
        settings = InputSettings.from_dict(data)
        assert settings.recursive is False
        assert settings.validate is False


class TestLoggingSettings:
    """Tests for LoggingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = LoggingSettings()
        assert settings.level == "info"
        assert settings.log_to_file is True
        assert settings.log_to_console is True

    def test_get_log_level(self):
        """Test get_log_level method."""
        import logging

        settings = LoggingSettings(level="debug")
        assert settings.get_log_level() == logging.DEBUG


class TestConfig:
    """Tests for Config dataclass."""

    def test_default(self):
        """Test default configuration."""
        config = Config.default()
        assert config.preprocessing.normalization == "min_max"
        assert config.metadata.profiles == ["minimal"]
        assert config.output.overwrite_policy == "skip"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Config.default()
        d = config.to_dict()
        assert "preprocessing" in d
        assert "metadata" in d
        assert "output" in d
        assert "input" in d
        assert "logging" in d

    def test_to_yaml(self):
        """Test conversion to YAML."""
        config = Config.default()
        yaml_str = config.to_yaml()
        assert "preprocessing:" in yaml_str
        assert "normalization:" in yaml_str

    def test_from_yaml(self):
        """Test creation from YAML."""
        yaml_str = """
preprocessing:
  normalization: z_score
metadata:
  profiles:
    - ml
"""
        config = Config.from_yaml(yaml_str)
        assert config.preprocessing.normalization == "z_score"
        assert config.metadata.profiles == ["ml"]

    def test_save_and_load(self, tmp_path):
        """Test save and load round-trip."""
        config = Config.default()
        config.preprocessing.normalization = "z_score"

        file_path = tmp_path / "config.yaml"
        config.save(file_path)

        loaded = Config.load(file_path)
        assert loaded.preprocessing.normalization == "z_score"

    def test_load_nonexistent(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.load("/nonexistent/config.yaml")

    def test_validate_valid(self):
        """Test validation of valid config."""
        config = Config.default()
        errors = config.validate()
        assert len(errors) == 0
        assert config.is_valid()

    def test_validate_invalid_normalization(self):
        """Test validation catches invalid normalization."""
        config = Config.default()
        config.preprocessing.normalization = "invalid"
        errors = config.validate()
        assert len(errors) > 0
        assert "normalization" in errors[0].lower()

    def test_validate_fixed_window_missing_values(self):
        """Test validation catches missing fixed window values."""
        config = Config.default()
        config.preprocessing.windowing.strategy = "fixed_window"
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_invalid_profile(self):
        """Test validation catches invalid metadata profile."""
        config = Config.default()
        config.metadata.profiles = ["invalid_profile"]
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_invalid_naming_policy(self):
        """Test validation catches invalid naming policy."""
        config = Config.default()
        config.output.naming_policy = "invalid"
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_negative_target_size(self):
        """Test validation catches negative target size."""
        config = Config.default()
        config.preprocessing.resizing.target_height = -100
        errors = config.validate()
        assert len(errors) > 0


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_file(self, tmp_path):
        """Test loading from specified file."""
        config = Config.default()
        config.preprocessing.normalization = "z_score"
        file_path = tmp_path / "config.yaml"
        config.save(file_path)

        loaded = load_config(file_path)
        assert loaded.preprocessing.normalization == "z_score"

    def test_load_default(self):
        """Test loading default when no file."""
        config = load_config()
        assert config.preprocessing.normalization == "min_max"

    def test_load_from_env(self, tmp_path, monkeypatch):
        """Test loading from environment variable."""
        config = Config.default()
        config.preprocessing.normalization = "z_score"
        file_path = tmp_path / "config.yaml"
        config.save(file_path)

        monkeypatch.setenv("CAD_PREPROCESS_CONFIG", str(file_path))
        loaded = load_config()
        assert loaded.preprocessing.normalization == "z_score"


class TestConfigTemplate:
    """Tests for configuration template."""

    def test_template_is_valid_yaml(self):
        """Test that template is valid YAML."""
        data = yaml.safe_load(CONFIG_TEMPLATE)
        assert "preprocessing" in data
        assert "metadata" in data

    def test_create_config_template(self, tmp_path):
        """Test creating config template file."""
        file_path = tmp_path / "config.yaml"
        create_config_template(file_path)

        assert file_path.exists()
        with open(file_path) as f:
            content = f.read()
        assert "preprocessing:" in content

    def test_create_default_config_file(self, tmp_path):
        """Test creating default config file."""
        file_path = tmp_path / "config.yaml"
        create_default_config_file(file_path)

        assert file_path.exists()
        loaded = Config.load(file_path)
        assert loaded.is_valid()
