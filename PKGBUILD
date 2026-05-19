# Maintainer: CAD Preprocess Team <cad-preprocess@example.com>
pkgname=cad-preprocess
pkgver=0.1.0
pkgrel=1
pkgdesc="A Python library for DICOM preprocessing in CAD systems"
arch=('any')
url="https://github.com/cad-preprocess/cad-preprocess"
license=('MIT')
depends=(
    'python>=3.9'
    'python-numpy'
    'python-pillow'
    'python-yaml'
    'python-pandas'
    'python-scikit-image'
    'python-pydicom'
    'python-dateutil'
)
optdepends=(
    'python-pyqt6: for the DICOM Explorer GUI'
    'python-pylibjpeg: for JPEG decompression support'
    'python-pylibjpeg-libjpeg: for JPEG decompression support'
    'python-pylibjpeg-openjpeg: for JPEG 2000 support'
    'gdcm: for additional DICOM codec support'
    'python-imageio: for additional image format support'
    'python-tifffile: for TIFF support'
)
makedepends=('python-setuptools' 'python-build' 'python-installer' 'python-wheel')
# To build from the local directory:
source=("${pkgname}-${pkgver}::git+file://$PWD")
sha256sums=('SKIP')

build() {
    cd "${pkgname}-${pkgver}"
    python -m build --wheel --no-isolation
}

package() {
    cd "${pkgname}-${pkgver}"
    python -m installer --destdir="${pkgdir}" dist/*.whl
}
