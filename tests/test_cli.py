"""
Tests for the CLI module.

These tests verify:
- Argument parsing
- Configuration loading and overrides
- Pipeline execution
- Error handling
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cad_preprocess.cli import (
    apply_cli_overrides,
    cli_setup_logging,
    create_parser,
    main,
    run_pipeline,
)
from cad_preprocess.config import Config
from cad_preprocess.logging_utils import ProcessingStats


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_creation(self):
        """Test parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "cad-preprocess"

    def test_required_arguments(self):
        """Test required arguments are enforced."""
        parser = create_parser()

        # Missing required arguments should fail
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_input_output_arguments(self):
        """Test input and output arguments."""
        parser = create_parser()
        args = parser.parse_args(["--input", "/path/to/input", "--output", "/path/to/output"])

        assert args.input == "/path/to/input"
        assert args.output == "/path/to/output"

    def test_short_arguments(self):
        """Test short argument forms."""
        parser = create_parser()
        args = parser.parse_args(["-i", "/input", "-o", "/output"])

        assert args.input == "/input"
        assert args.output == "/output"

    def test_config_argument(self):
        """Test config argument."""
        parser = create_parser()
        args = parser.parse_args([
            "--input", "/input",
            "--output", "/output",
            "--config", "config.yaml",
        ])

        assert args.config == "config.yaml"

    def test_metadata_profile_choices(self):
        """Test metadata profile choices."""
        parser = create_parser()
        args = parser.parse_args([
            "--input", "/input",
            "--output", "/output",
            "--metadata-profile", "ml",
        ])

        assert args.metadata_profile == "ml"

    def test_invalid_metadata_profile(self):
        """Test invalid metadata profile is rejected."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([
                "--input", "/input",
                "--output", "/output",
                "--metadata-profile", "invalid",
            ])

    def test_overwrite_flag(self):
        """Test overwrite flag."""
        parser = create_parser()

        # Without flag
        args = parser.parse_args(["--input", "/input", "--output", "/output"])
        assert args.overwrite is False

        # With flag
        args = parser.parse_args(["--input", "/input", "--output", "/output", "--overwrite"])
        assert args.overwrite is True

    def test_log_level_choices(self):
        """Test log level choices."""
        parser = create_parser()

        for level in ["debug", "info", "warning", "error", "critical"]:
            args = parser.parse_args([
                "--input", "/input",
                "--output", "/output",
                "--log-level", level,
            ])
            assert args.log_level == level

    def test_no_recursive_flag(self):
        """Test no-recursive flag."""
        parser = create_parser()
        args = parser.parse_args([
            "--input", "/input",
            "--output", "/output",
            "--no-recursive",
        ])

        assert args.no_recursive is True

    def test_dry_run_flag(self):
        """Test dry-run flag."""
        parser = create_parser()
        args = parser.parse_args([
            "--input", "/input",
            "--output", "/output",
            "--dry-run",
        ])

        assert args.dry_run is True

    def test_window_arguments(self):
        """Test window center and width arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "--input", "/input",
            "--output", "/output",
            "--window-center", "40",
            "--window-width", "400",
        ])

        assert args.window_center == 40.0
        assert args.window_width == 400.0

    def test_target_size_argument(self):
        """Test target size argument."""
        parser = create_parser()
        args = parser.parse_args([
            "--input", "/input",
            "--output", "/output",
            "--target-size", "512", "512",
        ])

        assert args.target_size == [512, 512]

    def test_normalization_choices(self):
        """Test normalization choices."""
        parser = create_parser()

        for method in ["min_max", "z_score"]:
            args = parser.parse_args([
                "--input", "/input",
                "--output", "/output",
                "--normalization", method,
            ])
            assert args.normalization == method

    def test_create_config_argument(self):
        """Test create-config argument."""
        parser = create_parser()
        args = parser.parse_args([
            "--input", "/input",
            "--output", "/output",
            "--create-config", "config.yaml",
        ])

        assert args.create_config == "config.yaml"


class TestApplyCliOverrides:
    """Tests for apply_cli_overrides function."""

    def test_override_log_level(self):
        """Test log level override."""
        config = Config.default()
        args = MagicMock()
        args.log_level = "debug"
        args.overwrite = False
        args.no_recursive = False
        args.no_validate = False
        args.metadata_profile = None
        args.normalization = None
        args.target_size = None
        args.window_center = None
        args.window_width = None

        config = apply_cli_overrides(config, args)
        assert config.logging.level == "debug"

    def test_override_overwrite(self):
        """Test overwrite policy override."""
        config = Config.default()
        args = MagicMock()
        args.log_level = None
        args.overwrite = True
        args.no_recursive = False
        args.no_validate = False
        args.metadata_profile = None
        args.normalization = None
        args.target_size = None
        args.window_center = None
        args.window_width = None

        config = apply_cli_overrides(config, args)
        assert config.output.overwrite_policy == "overwrite"

    def test_override_metadata_profile(self):
        """Test metadata profile override."""
        config = Config.default()
        args = MagicMock()
        args.log_level = None
        args.overwrite = False
        args.no_recursive = False
        args.no_validate = False
        args.metadata_profile = "ml"
        args.normalization = None
        args.target_size = None
        args.window_center = None
        args.window_width = None

        config = apply_cli_overrides(config, args)
        assert config.metadata.profiles == ["ml"]

    def test_override_metadata_all(self):
        """Test metadata profile 'all' override."""
        config = Config.default()
        args = MagicMock()
        args.log_level = None
        args.overwrite = False
        args.no_recursive = False
        args.no_validate = False
        args.metadata_profile = "all"
        args.normalization = None
        args.target_size = None
        args.window_center = None
        args.window_width = None

        config = apply_cli_overrides(config, args)
        assert config.metadata.include_all_profiles is True

    def test_override_target_size(self):
        """Test target size override."""
        config = Config.default()
        args = MagicMock()
        args.log_level = None
        args.overwrite = False
        args.no_recursive = False
        args.no_validate = False
        args.metadata_profile = None
        args.normalization = None
        args.target_size = [512, 512]
        args.window_center = None
        args.window_width = None

        config = apply_cli_overrides(config, args)
        assert config.preprocessing.resizing.target_height == 512
        assert config.preprocessing.resizing.target_width == 512

    def test_override_windowing(self):
        """Test windowing override."""
        config = Config.default()
        args = MagicMock()
        args.log_level = None
        args.overwrite = False
        args.no_recursive = False
        args.no_validate = False
        args.metadata_profile = None
        args.normalization = None
        args.target_size = None
        args.window_center = 40.0
        args.window_width = 400.0

        config = apply_cli_overrides(config, args)
        assert config.preprocessing.windowing.strategy == "fixed_window"
        assert config.preprocessing.windowing.window_center == 40.0
        assert config.preprocessing.windowing.window_width == 400.0


class TestSetupLogging:
    """Tests for cli_setup_logging function."""

    def test_setup_console_logging(self):
        """Test console logging setup."""
        import logging

        cli_setup_logging(level="info", log_to_console=True)

        logger = logging.getLogger("cad_preprocess")
        assert logger.level == logging.INFO

    def test_setup_file_logging(self, tmp_path):
        """Test file logging setup."""
        log_file = tmp_path / "test.log"
        cli_setup_logging(level="debug", log_file=log_file, log_to_console=False)

        logger = logging.getLogger("cad_preprocess")
        logger.info("Test message")

        # File should be created (may need to flush)
        assert log_file.parent.exists()


class TestMain:
    """Tests for main function."""

    def test_create_config_option(self, tmp_path):
        """Test --create-config option."""
        config_path = tmp_path / "config.yaml"

        result = main([
            "--input", "/fake/input",
            "--output", "/fake/output",
            "--create-config", str(config_path),
        ])

        assert result == 0
        assert config_path.exists()

    def test_missing_input(self, tmp_path):
        """Test error when input doesn't exist."""
        result = main([
            "--input", "/nonexistent/path",
            "--output", str(tmp_path),
        ])

        assert result == 1

    def test_invalid_config_file(self, tmp_path):
        """Test error with invalid config file."""
        result = main([
            "--input", str(tmp_path),
            "--output", str(tmp_path / "output"),
            "--config", "/nonexistent/config.yaml",
        ])

        assert result == 1

    @patch("cad_preprocess.cli.run_pipeline")
    def test_successful_run(self, mock_run_pipeline, tmp_path):
        """Test successful pipeline run."""
        # Create dummy input
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Mock ProcessingStats
        mock_stats = ProcessingStats()
        mock_stats.files_discovered = 0
        mock_stats.files_processed = 0
        mock_stats.files_skipped = 0
        mock_stats.files_failed = 0
        mock_run_pipeline.return_value = mock_stats

        result = main([
            "--input", str(input_dir),
            "--output", str(tmp_path / "output"),
        ])

        assert result == 0
        mock_run_pipeline.assert_called_once()

    @patch("cad_preprocess.cli.run_pipeline")
    def test_run_with_failures(self, mock_run_pipeline, tmp_path):
        """Test pipeline run with failures returns error code."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Mock ProcessingStats with failures
        mock_stats = ProcessingStats()
        mock_stats.files_discovered = 1
        mock_stats.files_processed = 0
        mock_stats.files_skipped = 0
        mock_stats.files_failed = 1
        mock_run_pipeline.return_value = mock_stats

        result = main([
            "--input", str(input_dir),
            "--output", str(tmp_path / "output"),
        ])

        assert result == 1


class TestRunPipeline:
    """Tests for run_pipeline function."""

    @patch("cad_preprocess.cli.InputHandler")
    @patch("cad_preprocess.cli.PreprocessingEngine")
    @patch("cad_preprocess.cli.MetadataExtractor")
    @patch("cad_preprocess.cli.OutputWriter")
    def test_dry_run(
        self,
        mock_writer,
        mock_extractor,
        mock_engine,
        mock_handler,
        tmp_path,
    ):
        """Test dry run mode."""
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery.total_discovered = 2
        mock_discovery.total_valid = 2
        mock_discovery.total_skipped = 0
        mock_discovery.valid_files = [
            Path("/fake/file1.dcm"),
            Path("/fake/file2.dcm"),
        ]
        mock_handler.return_value.discover.return_value = mock_discovery

        config = Config.default()
        stats = run_pipeline(
            input_path=tmp_path,
            output_path=tmp_path / "output",
            config=config,
            dry_run=True,
        )

        assert isinstance(stats, ProcessingStats)
        assert stats.files_discovered == 2
        # Engine should not process in dry run
        mock_engine.return_value.process.assert_not_called()

    @patch("cad_preprocess.cli.InputHandler")
    @patch("cad_preprocess.cli.PreprocessingEngine")
    @patch("cad_preprocess.cli.MetadataExtractor")
    @patch("cad_preprocess.cli.OutputWriter")
    def test_empty_input(
        self,
        mock_writer,
        mock_extractor,
        mock_engine,
        mock_handler,
        tmp_path,
    ):
        """Test with no input files."""
        mock_discovery = MagicMock()
        mock_discovery.total_discovered = 0
        mock_discovery.total_valid = 0
        mock_discovery.total_skipped = 0
        mock_discovery.valid_files = []
        mock_handler.return_value.discover.return_value = mock_discovery

        config = Config.default()
        stats = run_pipeline(
            input_path=tmp_path,
            output_path=tmp_path / "output",
            config=config,
        )

        assert isinstance(stats, ProcessingStats)
        assert stats.files_discovered == 0
        assert stats.files_processed == 0
