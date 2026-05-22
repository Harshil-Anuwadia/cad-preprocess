#!/usr/bin/env python3
# python-argcomplete-check
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
import os
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
        default=None,
        metavar="PATH",
        help="Input DICOM file or directory (can also be provided as first positional argument)",
    )

    required_group.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Output directory (can also be provided as second positional argument)",
    )

    # Positional arguments as fallbacks
    parser.add_argument(
        "pos_input",
        nargs="?",
        metavar="INPUT",
        help=argparse.SUPPRESS,  # Hide from help to keep it clean, but allow usage
    )
    parser.add_argument(
        "pos_output",
        nargs="?",
        metavar="OUTPUT",
        help=argparse.SUPPRESS,
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

    window_group = processing_group.add_mutually_exclusive_group()

    # To support both window_center and window_width being required together, 
    # we can't easily use purely mutually exclusive groups for the pair vs other options.
    # Instead, we will keep them as regular arguments and enforce in apply_cli_overrides.
    # The original implementation had them as separate arguments. Let's improve the logic in apply_cli_overrides instead.

    processing_group.add_argument(
        "--window-center",
        type=float,
        default=None,
        metavar="WC",
        help="Window center for intensity mapping (must use with --window-width)",
    )

    processing_group.add_argument(
        "--window-width",
        type=float,
        default=None,
        metavar="WW",
        help="Window width for intensity mapping (must use with --window-center)",
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

    behavior_group.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help="Number of worker processes for parallel processing (default: CPU count)",
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
        import sys
        print("Error: Both --window-center and --window-width must be specified together.", file=sys.stderr)
        sys.exit(1)

    return config


def run_pipeline(
    input_path: Path,
    output_path: Path,
    config: Config,
    dry_run: bool = False,
    num_workers: Optional[int] = None,
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
        num_workers: Number of worker processes.

    Returns:
        ProcessingStats with processing statistics.
    """
    from cad_preprocess.integration import CADPreprocessor
    
    # Use CADPreprocessor for batch processing as it now supports parallelism
    processor = CADPreprocessor(
        config=config,
        output_dir=output_path,
        num_workers=num_workers
    )
    
    if dry_run:
        # We still need to discover for dry-run
        from cad_preprocess.input_handler import InputHandler
        handler = InputHandler(
            validate=config.input.validate,
            recursive=config.input.recursive,
        )
        discovery = handler.discover(input_path)
        print(f"\nDry run: Found {discovery.total_valid} files to process.")
        return ProcessingStats()

    batch_result = processor.process_directory(input_path, output_path)
    return batch_result.stats or ProcessingStats()


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

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args(argv)

    # Handle fallbacks for input and output
    input_val = args.input or args.pos_input
    output_val = args.output or args.pos_output

    if args.create_config:
        from cad_preprocess.config import create_config_template
        create_config_template(args.create_config)
        print(f"Created configuration template: {args.create_config}")
        return 0

    # Ensure required arguments are present if not creating config
    if not input_val or not output_val:
        # If we are in argcomplete mode, exit quietly instead of printing error
        # to avoid breaking shell completion UI.
        if "_ARGCOMPLETE" in os.environ:
            sys.exit(0)
            
        parser.error("the following arguments are required: --input/-i, --output/-o (or positional INPUT OUTPUT)")

    # Validate input/output paths
    input_path = Path(input_val).resolve()
    output_path = Path(output_val).resolve()

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    # Check output directory permissions
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        # Test write permission
        test_file = output_path / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        print(f"Error: Output directory is not writable: {output_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
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
            num_workers=args.workers,
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
