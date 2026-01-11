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
        description="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CAD-PREPROCESS — Medical DICOM Image Preprocessing Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A production-ready preprocessing toolkit for Computer-Aided Detection and
  Diagnosis (CAD) systems. Standardizes DICOM images for training, inference,
  and clinical integration workflows.

  CAPABILITIES
  ─────────────────────────────────────────────────────────────────────────────
  • Automatic decompression (JPEG Lossless, JPEG 2000, RLE)
  • Intensity windowing with DICOM VOI LUT support
  • Flexible normalization (min-max, z-score)
  • Batch processing with parallel execution
  • Structured metadata extraction (JSON/CSV output)
  • Configurable via YAML or command-line arguments
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  BASIC OPERATIONS
  ─────────────────────────────────────────────────────────────────────────────
  Process a directory:
    $ cad-preprocess -i /data/dicoms -o /data/processed

  Process a single file:
    $ cad-preprocess -i scan.dcm -o ./output

  Preview without processing:
    $ cad-preprocess -i /data/dicoms -o ./output --dry-run

  CONFIGURATION
  ─────────────────────────────────────────────────────────────────────────────
  Use a configuration file:
    $ cad-preprocess -i ./dicoms -o ./output --config pipeline.yaml

  Generate a configuration template:
    $ cad-preprocess --create-config config.yaml

  METADATA EXTRACTION
  ─────────────────────────────────────────────────────────────────────────────
  Machine learning profile (recommended for training):
    $ cad-preprocess -i ./dicoms -o ./output --metadata-profile ml

  Full metadata extraction:
    $ cad-preprocess -i ./dicoms -o ./output --metadata-profile all

  IMAGE PROCESSING
  ─────────────────────────────────────────────────────────────────────────────
  Resize to standard dimensions:
    $ cad-preprocess -i ./dicoms -o ./output --target-size 512 512

  CT soft tissue windowing:
    $ cad-preprocess -i ./dicoms -o ./output --window-center 40 --window-width 400

  CT lung windowing:
    $ cad-preprocess -i ./dicoms -o ./output --window-center -600 --window-width 1500

  Min-max normalization:
    $ cad-preprocess -i ./dicoms -o ./output --normalization min_max

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RELATED TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DICOM Explorer (Interactive GUI):
    $ cad-preprocess-explorer

  Python API:
    >>> from cad_preprocess import preprocess, CADPreprocessor
    >>> result = preprocess('input.dcm', 'output/')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Documentation: https://github.com/Harshil-Anuwadia/cad-preprocess
  License: MIT | Copyright (c) 2024-2026 Harshil Anuwadia
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """,
    )

    # Version
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # =========================================================================
    # REQUIRED ARGUMENTS
    # =========================================================================
    required_group = parser.add_argument_group(
        'Required Arguments'
    )

    required_group.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        metavar="PATH",
        help="Input DICOM file or directory containing DICOM files",
    )

    required_group.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory for processed images and metadata",
    )

    # =========================================================================
    # CONFIGURATION OPTIONS
    # =========================================================================
    config_group = parser.add_argument_group(
        'Configuration'
    )

    config_group.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        metavar="FILE",
        help="YAML configuration file for pipeline settings",
    )

    config_group.add_argument(
        "--create-config",
        type=str,
        metavar="PATH",
        default=None,
        help="Generate a template configuration file and exit",
    )

    # =========================================================================
    # METADATA OPTIONS
    # =========================================================================
    metadata_group = parser.add_argument_group(
        'Metadata Extraction'
    )

    metadata_group.add_argument(
        "--metadata-profile", "-m",
        type=str,
        choices=["minimal", "patient", "geometry", "ml", "acquisition", "all"],
        default=None,
        metavar="PROFILE",
        help="Extraction profile: minimal, patient, geometry, ml, acquisition, all",
    )

    # =========================================================================
    # IMAGE PROCESSING OPTIONS
    # =========================================================================
    processing_group = parser.add_argument_group(
        'Image Processing'
    )

    processing_group.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Resize images to HxW pixels (e.g., --target-size 512 512)",
    )

    processing_group.add_argument(
        "--window-center",
        type=float,
        default=None,
        metavar="WC",
        help="Window center for intensity mapping (use with --window-width)",
    )

    processing_group.add_argument(
        "--window-width",
        type=float,
        default=None,
        metavar="WW",
        help="Window width for intensity mapping (use with --window-center)",
    )

    processing_group.add_argument(
        "--normalization",
        type=str,
        choices=["min_max", "z_score"],
        default=None,
        metavar="METHOD",
        help="Normalization: min_max (0-1) or z_score (standardized)",
    )

    # =========================================================================
    # PROCESSING OPTIONS
    # =========================================================================
    behavior_group = parser.add_argument_group(
        'Processing Options'
    )

    behavior_group.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files (default: skip)",
    )

    behavior_group.add_argument(
        "--no-recursive",
        action="store_true",
        default=False,
        help="Process only top-level directory (no subdirectories)",
    )

    behavior_group.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip DICOM validation (not recommended for production)",
    )

    behavior_group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show files to be processed without executing",
    )

    # =========================================================================
    # LOGGING OPTIONS
    # =========================================================================
    logging_group = parser.add_argument_group(
        'Logging'
    )

    logging_group.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default=None,
        metavar="LEVEL",
        help="Log verbosity: debug, info, warning, error, critical",
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
