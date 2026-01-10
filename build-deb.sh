#!/bin/bash
#
# Build FULLY SELF-CONTAINED .deb package for cad-preprocess
# One click install - no internet required!
#
# Usage: ./build-deb.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║     CAD Preprocess - Self-Contained DEB Builder      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Package info
PACKAGE_NAME="cad-preprocess"
VERSION="0.1.0"
ARCH="amd64"
MAINTAINER="CAD Preprocess Team <support@cadpreprocess.org>"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_deb"
PACKAGE_DIR="${BUILD_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}"
VENV_DIR="${BUILD_DIR}/venv"

# Cleanup
echo -e "${YELLOW}[1/7] Cleaning previous builds...${NC}"
rm -rf "${BUILD_DIR}"
rm -f "${SCRIPT_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
mkdir -p "${BUILD_DIR}"

# Check Python
echo -e "${YELLOW}[2/7] Checking Python...${NC}"
PYTHON_BIN=$(which python3)
if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}Error: python3 not found!${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✓ Found Python ${PYTHON_VERSION}${NC}"

# Create virtual environment and install dependencies
echo -e "${YELLOW}[3/7] Installing dependencies (this may take a minute)...${NC}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install all dependencies into venv
pip install --quiet --upgrade pip
pip install --quiet pydicom>=2.3.0 numpy>=1.21.0 Pillow>=9.0.0 PyYAML>=6.0 scikit-image>=0.19.0

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create package structure
echo -e "${YELLOW}[4/7] Creating package structure...${NC}"

mkdir -p "${PACKAGE_DIR}/DEBIAN"
mkdir -p "${PACKAGE_DIR}/opt/cad-preprocess/lib"
mkdir -p "${PACKAGE_DIR}/opt/cad-preprocess/bin"
mkdir -p "${PACKAGE_DIR}/usr/bin"
mkdir -p "${PACKAGE_DIR}/usr/share/applications"
mkdir -p "${PACKAGE_DIR}/usr/share/doc/cad-preprocess"
mkdir -p "${PACKAGE_DIR}/etc/cad-preprocess"

# Copy Python site-packages (all dependencies)
echo -e "${YELLOW}[5/7] Bundling Python packages...${NC}"
cp -r "${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages"/* "${PACKAGE_DIR}/opt/cad-preprocess/lib/"

# Copy our package
cp -r "${SCRIPT_DIR}/src/cad_preprocess" "${PACKAGE_DIR}/opt/cad-preprocess/lib/"

# Copy config
if [ -f "${SCRIPT_DIR}/config.example.yaml" ]; then
    cp "${SCRIPT_DIR}/config.example.yaml" "${PACKAGE_DIR}/etc/cad-preprocess/"
fi

# Copy docs
if [ -f "${SCRIPT_DIR}/README.md" ]; then
    cp "${SCRIPT_DIR}/README.md" "${PACKAGE_DIR}/usr/share/doc/cad-preprocess/"
fi

# Deactivate venv
deactivate

# Create the main executable
echo -e "${YELLOW}[6/7] Creating executable...${NC}"

cat > "${PACKAGE_DIR}/opt/cad-preprocess/bin/cad-preprocess" << 'SCRIPT'
#!/usr/bin/env python3
"""CAD Preprocess - DICOM Preprocessing for CAD Systems."""
import sys
import os

# Add bundled packages to path
lib_path = '/opt/cad-preprocess/lib'
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Import and run
from cad_preprocess.cli import main

if __name__ == '__main__':
    sys.exit(main())
SCRIPT

chmod +x "${PACKAGE_DIR}/opt/cad-preprocess/bin/cad-preprocess"

# Create symlink in /usr/bin
ln -sf /opt/cad-preprocess/bin/cad-preprocess "${PACKAGE_DIR}/usr/bin/cad-preprocess"

# Create desktop entry (for GUI file managers)
cat > "${PACKAGE_DIR}/usr/share/applications/cad-preprocess.desktop" << EOF
[Desktop Entry]
Name=CAD Preprocess
Comment=DICOM Preprocessing for CAD Systems
Exec=cad-preprocess --help
Terminal=true
Type=Application
Categories=Science;Medical;
EOF

# Create DEBIAN control file
INSTALLED_SIZE=$(du -sk "${PACKAGE_DIR}" | cut -f1)

cat > "${PACKAGE_DIR}/DEBIAN/control" << EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Installed-Size: ${INSTALLED_SIZE}
Depends: python3 (>= 3.9)
Maintainer: ${MAINTAINER}
Description: DICOM preprocessing tool for CAD systems
 CAD Preprocess is a complete, self-contained preprocessing tool
 for medical DICOM images. Used in CAD (Computer-Aided Detection)
 systems for ML training, inference, and visualization.
 .
 USAGE: cad-preprocess -i <input> -o <output>
 .
 Features:
  - DICOM file discovery and validation
  - Deterministic image preprocessing
  - Metadata extraction (ML, patient, geometry profiles)
  - PNG output with JSON metadata
  - CLI and Python API
 .
 All dependencies are bundled - no internet required!
Homepage: https://github.com/cad-preprocess/cad-preprocess
EOF

# Post-install script
cat > "${PACKAGE_DIR}/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   CAD Preprocess installed successfully!             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Usage:"
echo "    cad-preprocess --help"
echo "    cad-preprocess -i /path/to/dicoms -o /path/to/output"
echo ""
echo "  Example config: /etc/cad-preprocess/config.example.yaml"
echo ""

exit 0
EOF
chmod +x "${PACKAGE_DIR}/DEBIAN/postinst"

# Pre-remove script
cat > "${PACKAGE_DIR}/DEBIAN/prerm" << 'EOF'
#!/bin/bash
echo "Removing CAD Preprocess..."
exit 0
EOF
chmod +x "${PACKAGE_DIR}/DEBIAN/prerm"

# Post-remove script (cleanup)
cat > "${PACKAGE_DIR}/DEBIAN/postrm" << 'EOF'
#!/bin/bash
if [ "$1" = "purge" ]; then
    rm -rf /opt/cad-preprocess
    rm -rf /etc/cad-preprocess
fi
exit 0
EOF
chmod +x "${PACKAGE_DIR}/DEBIAN/postrm"

# Build the .deb
echo -e "${YELLOW}[7/7] Building .deb package...${NC}"
cd "${BUILD_DIR}"
dpkg-deb --build --root-owner-group "${PACKAGE_NAME}_${VERSION}_${ARCH}"

# Move to project root
mv "${PACKAGE_NAME}_${VERSION}_${ARCH}.deb" "${SCRIPT_DIR}/"

# Cleanup
rm -rf "${BUILD_DIR}"

# Done!
DEB_FILE="${SCRIPT_DIR}/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
DEB_SIZE=$(du -h "${DEB_FILE}" | cut -f1)

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              BUILD SUCCESSFUL!                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Package: ${CYAN}${PACKAGE_NAME}_${VERSION}_${ARCH}.deb${NC}"
echo -e "  Size:    ${CYAN}${DEB_SIZE}${NC}"
echo ""
echo -e "  ${YELLOW}To Install:${NC}"
echo -e "    sudo dpkg -i ${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo -e "  ${YELLOW}Or double-click the .deb file in file manager!${NC}"
echo ""
echo -e "  ${YELLOW}After install, just type:${NC}"
echo -e "    cad-preprocess --help"
echo ""
echo -e "  ${YELLOW}To Uninstall:${NC}"
echo -e "    sudo dpkg -r ${PACKAGE_NAME}"
echo ""
