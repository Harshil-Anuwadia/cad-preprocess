"""
Standalone Build Script for CAD Preprocess.

This script uses PyInstaller to bundle the application into a single
executable (or directory) for Windows and Linux.

Usage:
    python build_standalone.py --onefile
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

def run_command(cmd, msg=None):
    if msg:
        print(f"--> {msg}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # Detect platform
    system = platform.system().lower()
    print(f"Building for: {system}")

    # Build directory
    dist_dir = Path("dist_standalone")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    # Base PyInstaller command
    # We build the Explorer (GUI) as the main entry point for the standalone app
    # but the CLI is also included in the package.
    
    # Common hidden imports for pydicom handlers
    hidden_imports = [
        "pydicom.encoders.native",
        "pydicom.encoders.pylibjpeg",
        "pydicom.encoders.gdcm",
        "pylibjpeg",
        "pylibjpeg_libjpeg",
        "openjpeg",
        "pandas",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
    ]

    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--name", f"cad-preprocess-{system}",
    ]

    # Onefile vs Onedir
    if "--onefile" in sys.argv:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # GUI vs Console
    # On Windows, we might want to hide the console for the explorer
    if system == "windows":
        cmd.append("--windowed")
    
    # Add hidden imports
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # Collect data files (if any needed, e.g. icons)
    # cmd.extend(["--add-data", "src/cad_preprocess/assets;assets"])

    # Entry point
    # We'll create a wrapper script that can launch either CLI or GUI
    wrapper_script = Path("standalone_wrapper.py")
    with open(wrapper_script, "w") as f:
        f.write("""
import sys
from cad_preprocess.cli import main as cli_main
from cad_preprocess.explorer import main as gui_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Shift arguments
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        sys.exit(cli_main())
    else:
        sys.exit(gui_main())
""")

    cmd.append(str(wrapper_script))

    # Run PyInstaller
    run_command(cmd, "Running PyInstaller...")

    print(f"\nDone! Executable built in: {dist_dir}")
    print(f"You can find it in the 'dist' folder.")

if __name__ == "__main__":
    # Ensure dependencies are installed
    # run_command([sys.executable, "-m", "pip", "install", ".[gui]", "pyinstaller"], "Installing dependencies...")
    main()
