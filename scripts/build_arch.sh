#!/bin/bash
# ==========================================================================
# CAD-PREPROCESS COMPLETE BUNDLE BUILD SCRIPT FOR ARCH LINUX
# ==========================================================================
# Creates a fully self-contained Arch Linux package (.pkg.tar.zst)
# with ALL dependencies bundled in /opt/cad-preprocess/lib
# ==========================================================================

set -euo pipefail

PACKAGE_NAME="cad-preprocess"
VERSION="0.1.0"
RELEASE="1"

echo ""
echo "=========================================================="
echo "  CAD-PREPROCESS ARCH LINUX BUNDLE BUILD"
echo "=========================================================="
echo ""

# Clean previous builds safely
rm -rf build_arch
mkdir -p build_arch

# Create package directory structure
# Using 'root' instead of 'pkg' to avoid conflict with makepkg's internal pkg directory
PKG_ROOT="${PWD}/build_arch/root"
mkdir -p "${PKG_ROOT}/opt/cad-preprocess/lib"
mkdir -p "${PKG_ROOT}/usr/bin"

# Note: Arch typically uses the latest python. Let's find the current version.
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITEPACKAGES_DIR="${PKG_ROOT}/usr/lib/python${PYTHON_VERSION}/site-packages"
mkdir -p "${SITEPACKAGES_DIR}"

# Create a virtual environment and install all dependencies
echo "[1/6] Creating virtual environment..."
python3 -m venv build_arch/venv
echo "      - Virtual environment created."

# Create a wrapper function to run pip inside the virtual environment
function run_pip() {
    build_arch/venv/bin/pip "$@"
}

echo "[2/6] Installing ALL dependencies..."
echo "      - Upgrading pip, wheel, setuptools..."
run_pip install --upgrade pip wheel setuptools -q

echo "      - Installing core data science packages (numpy, pillow, scipy, pandas)..."
run_pip install numpy pillow scipy pandas python-dateutil -q

echo "      - Installing DICOM specific packages (pydicom, pylibjpeg, gdcm)..."
run_pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg -q || echo "      - Warning: Failed to install some pylibjpeg packages"

# Note: python-gdcm might not be available via pip on all platforms easily, 
# but we try to bundle what we can.
run_pip install python-gdcm -q || echo "      - Warning: python-gdcm not available via pip, skipping"

echo "      - Installing image processing and GUI packages (scikit-image, PyQt6)..."
run_pip install scikit-image PyYAML click tzdata imageio tifffile PyQt6 -q || echo "      - Warning: Failed to install some optional packages"

# Build and install the actual cad_preprocess package to ensure dependencies match pyproject.toml
echo "      - Installing cad-preprocess to resolve remaining dependencies..."
run_pip install . -q

echo "[3/6] Copying bundled libraries..."
cp -r "build_arch/venv/lib/python${PYTHON_VERSION}/site-packages/"* "${PKG_ROOT}/opt/cad-preprocess/lib/"
echo "      - Libraries copied to /opt/cad-preprocess/lib"

# Copy our module (this shouldn't be strictly necessary if we pip install it, but keeping for compatibility)
echo "[4/6] Copying cad_preprocess module..."
cp -r src/cad_preprocess "${PKG_ROOT}/opt/cad-preprocess/lib/"
echo "      - Source module copied."

# Cleanup
echo "[5/6] Cleaning up unnecessary files..."
find "${PKG_ROOT}/opt/cad-preprocess/lib" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf "${PKG_ROOT}/opt/cad-preprocess/lib/pip"* 2>/dev/null || true
rm -rf "${PKG_ROOT}/opt/cad-preprocess/lib/setuptools"* 2>/dev/null || true
echo "      - Staging area cleaned."

# Create CLI wrappers
cat > "${PKG_ROOT}/usr/bin/cad-preprocess" << 'ENDSCRIPT'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/cad-preprocess/lib')
from cad_preprocess.cli import main
if __name__ == "__main__":
    sys.exit(main())
ENDSCRIPT
chmod 755 "${PKG_ROOT}/usr/bin/cad-preprocess"

cat > "${PKG_ROOT}/usr/bin/cad-preprocess-explorer" << 'ENDSCRIPT'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/cad-preprocess/lib')
from cad_preprocess.explorer import main
if __name__ == "__main__":
    sys.exit(main())
ENDSCRIPT
chmod 755 "${PKG_ROOT}/usr/bin/cad-preprocess-explorer"

cat > "${PKG_ROOT}/usr/bin/cad-preprocess-diagnose" << 'ENDSCRIPT'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/cad-preprocess/lib')
from cad_preprocess.diagnose_cli import main
if __name__ == "__main__":
    sys.exit(main())
ENDSCRIPT
chmod 755 "${PKG_ROOT}/usr/bin/cad-preprocess-diagnose"

# Create .pth file
cat > "${SITEPACKAGES_DIR}/cad_preprocess.pth" << 'ENDPTH'
/opt/cad-preprocess/lib
ENDPTH

# Create PKGBUILD for the bundled version
cat > build_arch/PKGBUILD << ENDPKGBUILD
pkgname=${PACKAGE_NAME}-bundled
pkgver=${VERSION}
pkgrel=${RELEASE}
pkgdesc="DICOM preprocessing library for CAD systems (fully bundled)"
arch=('x86_64')
url="https://github.com/cad-preprocess/cad-preprocess"
license=('MIT')
depends=('python')
provides=('cad-preprocess')
conflicts=('cad-preprocess')

package() {
    # Copy from the absolute path of the staging root
    cp -r "${PKG_ROOT}/"* "\${pkgdir}/"
}
ENDPKGBUILD

echo "[6/6] Building Arch package with makepkg..."
pushd build_arch > /dev/null
makepkg -f --noconfirm
popd > /dev/null

# Move to current directory
mv build_arch/${PACKAGE_NAME}-bundled-${VERSION}-${RELEASE}-x86_64.pkg.tar.zst . || echo "Warning: Could not find generated package."

echo ""
echo "=========================================================="
echo "  ARCH BUILD COMPLETE!"
echo "=========================================================="
