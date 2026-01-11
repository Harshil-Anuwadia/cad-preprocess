#!/bin/bash
# ==========================================================================
# CAD-PREPROCESS COMPLETE BUNDLE BUILD SCRIPT
# ==========================================================================
# Creates a fully self-contained .deb package with ALL dependencies
# including JPEG lossless decompression support for medical DICOM files
# ==========================================================================

set -e

PACKAGE_NAME="cad-preprocess"
VERSION="0.1.0"
ARCH="all"

echo ""
echo "=========================================================="
echo "  CAD-PREPROCESS COMPLETE BUNDLE BUILD"
echo "=========================================================="
echo ""

# Clean previous builds
rm -rf build_deb
mkdir -p build_deb

# Create package directory structure
PKG_DIR="build_deb/${PACKAGE_NAME}_${VERSION}_${ARCH}"
mkdir -p "${PKG_DIR}/DEBIAN"
mkdir -p "${PKG_DIR}/opt/cad-preprocess/lib"
mkdir -p "${PKG_DIR}/usr/bin"
mkdir -p "${PKG_DIR}/usr/lib/python3/dist-packages"

# Create a virtual environment and install all dependencies
echo "[1/6] Creating virtual environment..."
python3 -m venv build_deb/venv

echo "[2/6] Installing ALL dependencies (this may take a while)..."
build_deb/venv/bin/pip install --upgrade pip wheel setuptools -q

# Core dependencies
echo "       - Core packages (numpy, pillow, scipy, pandas, python-dateutil)..."
build_deb/venv/bin/pip install numpy pillow scipy pandas python-dateutil -q

# DICOM handling with ALL decompression plugins
echo "       - DICOM packages with decompression support..."
build_deb/venv/bin/pip install pydicom -q
build_deb/venv/bin/pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg -q || true
build_deb/venv/bin/pip install python-gdcm -q || true

# Image processing
echo "       - Image processing (scikit-image)..."
build_deb/venv/bin/pip install scikit-image -q

# Configuration and utilities  
echo "       - Utilities (PyYAML, click, tzdata)..."
build_deb/venv/bin/pip install PyYAML click tzdata -q

# Additional image format support
echo "       - Image format support (imageio, tifffile)..."
build_deb/venv/bin/pip install imageio tifffile -q

# GUI support
echo "       - GUI support (PyQt6)..."
build_deb/venv/bin/pip install PyQt6 -q

echo "[3/6] Copying bundled libraries..."
# Copy site-packages (all dependencies)
cp -r build_deb/venv/lib/python*/site-packages/* "${PKG_DIR}/opt/cad-preprocess/lib/"

# Copy our module
echo "[4/6] Copying cad_preprocess module..."
cp -r src/cad_preprocess "${PKG_DIR}/opt/cad-preprocess/lib/"

# Cleanup unnecessary files
echo "[5/6] Cleaning up unnecessary files..."
find "${PKG_DIR}/opt/cad-preprocess/lib" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PKG_DIR}/opt/cad-preprocess/lib" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "${PKG_DIR}/opt/cad-preprocess/lib" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
find "${PKG_DIR}/opt/cad-preprocess/lib" -name "*.pyc" -delete 2>/dev/null || true

# Remove pip/setuptools/wheel (not needed at runtime)
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/pip" 2>/dev/null || true
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/setuptools" 2>/dev/null || true
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/wheel" 2>/dev/null || true
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/_distutils_hack" 2>/dev/null || true
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/pkg_resources" 2>/dev/null || true
rm -f "${PKG_DIR}/opt/cad-preprocess/lib/pip-"*".dist-info" 2>/dev/null || true
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/pip-"* 2>/dev/null || true
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/setuptools-"* 2>/dev/null || true
rm -rf "${PKG_DIR}/opt/cad-preprocess/lib/wheel-"* 2>/dev/null || true

# Create the CLI wrapper script
cat > "${PKG_DIR}/usr/bin/cad-preprocess" << 'ENDSCRIPT'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/cad-preprocess/lib')

from cad_preprocess.cli import main

if __name__ == "__main__":
    sys.exit(main())
ENDSCRIPT
chmod 755 "${PKG_DIR}/usr/bin/cad-preprocess"

# Create the DICOM Explorer GUI wrapper script
cat > "${PKG_DIR}/usr/bin/cad-preprocess-explorer" << 'ENDSCRIPT'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/cad-preprocess/lib')

from cad_preprocess.explorer import main

if __name__ == "__main__":
    sys.exit(main())
ENDSCRIPT
chmod 755 "${PKG_DIR}/usr/bin/cad-preprocess-explorer"

# Create a .pth file so Python can find the module when importing
cat > "${PKG_DIR}/usr/lib/python3/dist-packages/cad_preprocess.pth" << 'ENDPTH'
/opt/cad-preprocess/lib
ENDPTH

# Create control file
cat > "${PKG_DIR}/DEBIAN/control" << ENDCONTROL
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: python
Priority: optional
Architecture: ${ARCH}
Depends: python3 (>= 3.9)
Installed-Size: $(du -sk "${PKG_DIR}" | cut -f1)
Maintainer: Harshil Anuwadia <harshil@example.com>
Homepage: https://github.com/Harshil-Anuwadia/cad-preprocess
Description: DICOM preprocessing library for CAD systems (fully bundled)
 CAD Preprocess is a Python library for standardized DICOM image
 preprocessing in Computer-Aided Detection/Diagnosis systems.
 .
 This package includes ALL dependencies bundled:
  - numpy, scipy, pillow, pandas, python-dateutil (data processing)
  - pydicom with JPEG decompression plugins
  - python-gdcm (additional DICOM codec support)
  - scikit-image (advanced image operations)
  - PyYAML, click (configuration and CLI)
  - PyQt6 (DICOM Explorer GUI)
 .
 Supports compressed DICOM formats:
  - JPEG Lossless (Process 14)
  - JPEG 2000
  - Other standard compressions
 .
 Usage:
  CLI: cad-preprocess -i ./dicoms -o ./output
  Explorer: cad-preprocess-explorer
  Python: from cad_preprocess import preprocess
ENDCONTROL

# Create postinst
cat > "${PKG_DIR}/DEBIAN/postinst" << 'ENDPOST'
#!/bin/bash
set -e
chmod 644 /usr/lib/python3/dist-packages/cad_preprocess.pth 2>/dev/null || true
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        CAD-Preprocess installed successfully!                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Usage:"
echo ""
echo "  CLI (Preprocessing):"
echo "    cad-preprocess -i ./dicoms -o ./output"
echo "    cad-preprocess -i ./dicoms -o ./output --config config.yaml"
echo "    cad-preprocess --help"
echo ""
echo "  DICOM Explorer (GUI):"
echo "    cad-preprocess-explorer"
echo ""
echo "  Python API:"
echo "    from cad_preprocess import preprocess"
echo "    from cad_preprocess import CADPreprocessor"
echo ""
exit 0
ENDPOST
chmod 755 "${PKG_DIR}/DEBIAN/postinst"

# Create prerm
cat > "${PKG_DIR}/DEBIAN/prerm" << 'ENDPRERM'
#!/bin/bash
set -e
rm -rf /opt/cad-preprocess/lib/__pycache__ 2>/dev/null || true
exit 0
ENDPRERM
chmod 755 "${PKG_DIR}/DEBIAN/prerm"

# Create postrm (cleanup on uninstall)
cat > "${PKG_DIR}/DEBIAN/postrm" << 'ENDPOSTRM'
#!/bin/bash
set -e
if [ "$1" = "purge" ] || [ "$1" = "remove" ]; then
    rm -rf /opt/cad-preprocess 2>/dev/null || true
    rm -f /usr/lib/python3/dist-packages/cad_preprocess.pth 2>/dev/null || true
fi
exit 0
ENDPOSTRM
chmod 755 "${PKG_DIR}/DEBIAN/postrm"

echo "[6/6] Building .deb package..."
dpkg-deb --build --root-owner-group "${PKG_DIR}"

# Move to current directory
mv "build_deb/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb" .

# Show package info
SIZE=$(du -h "${PACKAGE_NAME}_${VERSION}_${ARCH}.deb" | cut -f1)
INSTALLED_SIZE=$(du -sh "${PKG_DIR}" | cut -f1)

echo ""
echo "=========================================================="
echo "  BUILD COMPLETE!"
echo "=========================================================="
echo ""
echo "  Package:        ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo "  Package Size:   ${SIZE}"
echo "  Installed Size: ${INSTALLED_SIZE}"
echo ""
echo "  Included packages:"
echo "    - numpy, scipy, pillow, pandas, python-dateutil"
echo "    - pydicom + JPEG decompression plugins"
echo "    - python-gdcm (additional DICOM support)"
echo "    - scikit-image"
echo "    - PyYAML, click, tzdata"
echo "    - imageio, tifffile"
echo "    - PyQt6 (GUI - for DICOM Explorer)"
echo ""
echo "=========================================================="
echo "  INSTALLATION"
echo "=========================================================="
echo ""
echo "  sudo dpkg -i ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo "=========================================================="
echo "  USAGE AFTER INSTALLATION"
echo "=========================================================="
echo ""
echo "  CLI (Preprocessing):"
echo "    cad-preprocess -i ./dicoms -o ./output"
echo "    cad-preprocess -i ./dicoms -o ./output --config config.yaml"
echo "    cad-preprocess -i ./dicoms -o ./output --metadata-profile ml"
echo "    cad-preprocess -i ./dicoms -o ./output --overwrite --log-level debug"
echo "    cad-preprocess --help"
echo ""
echo "  DICOM Explorer (GUI):"
echo "    cad-preprocess-explorer"
echo ""
echo "  Python API:"
echo "    from cad_preprocess import preprocess"
echo "    result = preprocess('input.dcm', 'output/')"
echo ""
echo "    from cad_preprocess import CADPreprocessor"
echo "    preprocessor = CADPreprocessor('config.yaml')"
echo "    result = preprocessor.process_batch(['file1.dcm', 'file2.dcm'])"
echo ""
echo "=========================================================="
