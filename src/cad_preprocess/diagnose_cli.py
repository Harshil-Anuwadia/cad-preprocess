"""
CLI entry point for the DICOM diagnostics tool.
"""

import argparse
import os
import sys
from pathlib import Path

from cad_preprocess.diagnostics import DicomAnalyzer

# ANSI Colors for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def is_dicom_file(filepath: Path) -> bool:
    """Quickly check if a file has the DICOM magic bytes."""
    if not filepath.is_file():
        return False
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="CAD-Preprocess DICOM Diagnostics Tool. Analyzes DICOM files for corruption, missing metadata, and decompression errors."
    )
    parser.add_argument("input", type=str, help="Path to DICOM file or directory")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show detailed reports for HEALTHY files as well (default: only show issues)",
    )
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    if not args.input:
        if "_ARGCOMPLETE" in os.environ:
            sys.exit(0)
        parser.error("the following arguments are required: input")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"{RED}Error: Path {input_path} does not exist.{RESET}")
        sys.exit(1)

    analyzer = DicomAnalyzer()

    files_to_check = []
    if input_path.is_file():
        files_to_check = [input_path]
    else:
        print(f"{BOLD}Scanning directory for DICOM files...{RESET}")
        # Find files with .dcm/.dicom extensions or valid magic bytes
        for p in input_path.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                if p.suffix.lower() in [".dcm", ".dicom"] or is_dicom_file(p):
                    files_to_check.append(p)

    if not files_to_check:
        print(f"{YELLOW}No DICOM files found in the specified path.{RESET}")
        sys.exit(0)

    print(f"{BOLD}Analyzing {len(files_to_check)} files...{RESET}\n")

    healthy_count = 0
    warning_count = 0
    error_count = 0

    for file_path in files_to_check:
        diag = analyzer.analyze(file_path)

        if diag.health_status == "HEALTHY":
            healthy_count += 1
        elif diag.health_status == "WARNINGS":
            warning_count += 1
        else:
            error_count += 1

        if diag.health_status == "HEALTHY":
            color = GREEN
        elif diag.health_status == "WARNINGS":
            color = YELLOW
        else:
            color = RED

        # Print detailed report if it's a single file, if there are issues, or if --all is used
        if len(files_to_check) == 1 or diag.health_status != "HEALTHY" or args.all:
            print(f"{BOLD}File:{RESET} {CYAN}{file_path}{RESET}")
            print(f"{BOLD}Status:{RESET} {color}{diag.health_status}{RESET}")

            if diag.is_dicom:
                print(f"{BOLD}Metadata:{RESET}")
                for k, v in diag.metadata.items():
                    print(f"  - {k}: {v}")

            if diag.issues:
                print(f"{BOLD}Issues & Fixes:{RESET}")
                for issue in diag.issues:
                    icolor = RED if issue.severity == "ERROR" else YELLOW
                    print(f"  [{icolor}{issue.severity}{RESET}] {issue.description}")
                    print(f"          {BOLD}Fix:{RESET} {issue.suggestion}")
            print("-" * 50)

    print(f"\n{BOLD}Diagnostic Summary:{RESET}")
    print(f"  Total Files Checked: {len(files_to_check)}")
    print(f"  {GREEN}Healthy:{RESET}            {healthy_count}")
    print(f"  {YELLOW}With Warnings:{RESET}      {warning_count}")
    print(f"  {RED}Corrupted/Errors:{RESET}   {error_count}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
