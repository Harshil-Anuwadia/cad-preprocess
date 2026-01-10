"""
Command Line Interface for CAD Preprocess.

This module provides a CLI for running the DICOM preprocessing pipeline
without writing code.

Usage:
    cad-preprocess --input <path> --output <path> [options]

Examples:
    # Process single file
    cad-preprocess --input image.dcm --output ./output

    # Process directory
    cad-preprocess --input ./dicoms --output ./output --config config.yaml

    # With metadata profile
    cad-preprocess --input ./dicoms --output ./output --metadata-profile ml

    # Override overwrite policy
    cad-preprocess --input ./dicoms --output ./output --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from cad_preprocess import __version__
from cad_preprocess.config import Config, load_config
from cad_preprocess.input_handler import InputHandler
from cad_preprocess.logging_utils import (
    ErrorRecord,
    LogLevel,
    ProcessingLogger,
    ProcessingStage,
    ProcessingStats,
    setup_logging,
)
from cad_preprocess.metadata_extractor import MetadataExtractor, MetadataProfile
from cad_preprocess.output_writer import OutputConfig, OutputWriter, OverwritePolicy
from cad_preprocess.preprocessing_engine import PreprocessingEngine

# Configure module logger
logger = ProcessingLogger(__name__)


def cli_setup_logging(
    level: str = "info",
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
) -> None:
    """
    Configure logging for the CLI.

    Args:
        level: Log level (debug, info, warning, error, critical).
        log_file: Optional path to log file.
        log_to_console: Whether to log to console.
    """
    # Use centralized logging setup from logging_utils
    log_level = LogLevel(level.lower())
    setup_logging(
        level=log_level,
        log_file=log_file,
        log_to_console=log_to_console,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cad-preprocess",
        description="Preprocess DICOM images for CAD systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single DICOM file
  cad-preprocess --input image.dcm --output ./output

  # Process a directory of DICOM files
  cad-preprocess --input ./dicoms --output ./output

  # Use a configuration file
  cad-preprocess --input ./dicoms --output ./output --config config.yaml

  # Specify metadata profile
  cad-preprocess --input ./dicoms --output ./output --metadata-profile ml

  # Enable overwrite mode
  cad-preprocess --input ./dicoms --output ./output --overwrite

  # Set log level
  cad-preprocess --input ./dicoms --output ./output --log-level debug

For more information, see: https://github.com/cad-preprocess/cad-preprocess
        """,
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input DICOM file or directory containing DICOM files",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for processed images and metadata",
    )

    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--metadata-profile", "-m",
        type=str,
        choices=["minimal", "patient", "geometry", "ml", "acquisition", "all"],
        default=None,
        help="Metadata extraction profile (overrides config file)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files (default: skip existing)",
    )

    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default=None,
        help="Logging level (overrides config file)",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        default=False,
        help="Do not scan input directory recursively",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be processed without actually processing",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip DICOM validation (not recommended)",
    )

    # Advanced options
    parser.add_argument(
        "--window-center",
        type=float,
        default=None,
        help="Fixed window center value (requires --window-width)",
    )

    parser.add_argument(
        "--window-width",
        type=float,
        default=None,
        help="Fixed window width value (requires --window-center)",
    )

    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Target image size (height width), e.g., --target-size 512 512",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=["min_max", "z_score"],
        default=None,
        help="Normalization method (overrides config file)",
    )

    parser.add_argument(
        "--create-config",
        type=str,
        metavar="PATH",
        default=None,
        help="Create a default configuration file at the specified path and exit",
    )

    return parser


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """
    Apply CLI argument overrides to configuration.

    Args:
        config: Base configuration.
        args: Parsed CLI arguments.

    Returns:
        Updated configuration.
    """
    # Override log level
    if args.log_level:
        config.logging.level = args.log_level

    # Override overwrite policy
    if args.overwrite:
        config.output.overwrite_policy = "overwrite"

    # Override recursive scanning
    if args.no_recursive:
        config.input.recursive = False

    # Override validation
    if args.no_validate:
        config.input.validate = False

    # Override metadata profile
    if args.metadata_profile:
        if args.metadata_profile == "all":
            config.metadata.include_all_profiles = True
        else:
            config.metadata.profiles = [args.metadata_profile]

    # Override normalization
    if args.normalization:
        config.preprocessing.normalization = args.normalization

    # Override target size
    if args.target_size:
        config.preprocessing.resizing.target_height = args.target_size[0]
        config.preprocessing.resizing.target_width = args.target_size[1]

    # Override windowing
    if args.window_center is not None and args.window_width is not None:
        config.preprocessing.windowing.strategy = "fixed_window"
        config.preprocessing.windowing.window_center = args.window_center
        config.preprocessing.windowing.window_width = args.window_width
    elif args.window_center is not None or args.window_width is not None:
        logger.warning("Both --window-center and --window-width must be specified for fixed windowing")

    return config


def run_pipeline(
    input_path: Path,
    output_path: Path,
    config: Config,
    dry_run: bool = False,
) -> ProcessingStats:
    """
    Run the complete preprocessing pipeline.

    Uses fail-safe error handling: logs errors and continues processing.
    Never crashes on single file failure.

    Args:
        input_path: Input file or directory.
        output_path: Output directory.
        config: Processing configuration.
        dry_run: If True, only show what would be processed.

    Returns:
        ProcessingStats with processing statistics.
    """
    # Use processing logger with structured tracking
    proc_logger = ProcessingLogger("cad_preprocess.pipeline")

    with proc_logger.processing_context("cli_batch") as stats:
        # Initialize components
        input_handler = InputHandler(
            validate=config.input.validate,
            recursive=config.input.recursive,
            check_pixel_data=config.input.check_pixel_data,
            check_dimensions=config.input.check_dimensions,
        )

        preprocessing_engine = PreprocessingEngine(
            config=config.preprocessing.to_preprocessing_config()
        )

        metadata_extractor = MetadataExtractor(
            profiles=config.metadata.profiles,
            additional_fields=config.metadata.additional_fields,
            include_all_profiles=config.metadata.include_all_profiles,
        )

        output_config = OutputConfig(
            output_root=output_path,
            naming_policy=config.output.get_naming_policy(),
            overwrite_policy=config.output.get_overwrite_policy(),
            image_format=config.output.image_format,
            images_subdir=config.output.images_subdir,
            metadata_subdir=config.output.metadata_subdir,
            logs_subdir=config.output.logs_subdir,
        )
        output_writer = OutputWriter(output_path, config=output_config)

        # Discover input files
        proc_logger.info(f"Discovering DICOM files in: {input_path}")
        discovery_result = input_handler.discover(input_path)
        stats.files_discovered = discovery_result.total_discovered
        stats.files_valid = discovery_result.total_valid

        proc_logger.info(
            f"Found {discovery_result.total_valid} valid DICOM files "
            f"({discovery_result.total_skipped} skipped)"
        )

        if dry_run:
            proc_logger.info("Dry run mode - no files will be processed")
            print("\nFiles that would be processed:")
            for file_path in discovery_result.valid_files:
                print(f"  {file_path}")
            print(f"\nTotal: {discovery_result.total_valid} files")
            return stats

        # Log batch start
        proc_logger.log_batch_start(discovery_result.total_valid)

        # Process each file with fail-safe error handling
        for i, file_path in enumerate(discovery_result.valid_files, 1):
            proc_logger.log_file_start(file_path, i, discovery_result.total_valid)

            # Use file context for automatic error handling
            with proc_logger.file_context(
                file_path, stats, ProcessingStage.PREPROCESSING
            ):
                # Preprocess image
                result = preprocessing_engine.process(file_path)

                if not result.success:
                    proc_logger.log_file_error(
                        file_path,
                        Exception(result.error_message),
                        ProcessingStage.PREPROCESSING,
                    )
                    stats.files_failed += 1
                    stats.errors.append(
                        ErrorRecord(
                            file_path=file_path,
                            stage=ProcessingStage.PREPROCESSING,
                            error_type="PreprocessingError",
                            error_message=result.error_message or "Unknown error",
                        )
                    )
                    continue

                # Extract metadata
                metadata_result = metadata_extractor.extract(file_path)
                if not metadata_result.success:
                    proc_logger.warning(
                        f"Metadata extraction failed: {metadata_result.error_message}"
                    )

                # Get SOPInstanceUID for naming
                sop_uid = metadata_result.metadata.get("SOPInstanceUID")

                # Write outputs
                img_result = output_writer.write_image(
                    result.image,
                    sop_instance_uid=sop_uid,
                    original_path=file_path,
                )

                meta_result = output_writer.write_metadata(
                    metadata_result.metadata,
                    sop_instance_uid=sop_uid,
                    original_path=file_path,
                )

                if img_result.action == "skipped":
                    proc_logger.log_file_skipped(file_path, "already exists")
                    stats.files_skipped += 1
                elif img_result.success:
                    proc_logger.log_file_success(file_path)
                    stats.files_processed += 1
                else:
                    proc_logger.log_file_error(
                        file_path,
                        Exception(img_result.error_message or "Write failed"),
                        ProcessingStage.OUTPUT_WRITING,
                    )
                    stats.files_failed += 1

        # Write processing log with stats
        stats_dict = stats.to_dict()
        stats_dict["input_path"] = str(input_path)
        stats_dict["output_path"] = str(output_path)
        output_writer.write_batch_summary(stats_dict)

    return stats


def print_summary(stats: ProcessingStats) -> None:
    """Print processing summary to console."""
    print(stats.summary())


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle --create-config
    if args.create_config:
        from cad_preprocess.config import create_config_template
        create_config_template(args.create_config)
        print(f"Created configuration template: {args.create_config}")
        return 0

    # Validate input/output paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    # Setup logging
    log_file = None
    if config.logging.log_to_file:
        log_file = output_path / config.output.logs_subdir / config.logging.log_filename

    cli_setup_logging(
        level=config.logging.level,
        log_file=log_file,
        log_to_console=config.logging.log_to_console,
    )

    # Print banner
    logger.info(f"CAD Preprocess v{__version__}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Run pipeline
    try:
        stats = run_pipeline(
            input_path=input_path,
            output_path=output_path,
            config=config,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1

    # Print summary
    print_summary(stats)

    # Return appropriate exit code
    if stats.files_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
