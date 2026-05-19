<h1 align="center">CAD Preprocess</h1>

<p align="center">
  <strong>Medical DICOM Image Preprocessing Pipeline for CAD Systems</strong>
</p>

<p align="center">
  A production-ready Python utility for standardizing DICOM preprocessing across<br>
  training, inference, and clinical integration workflows.
</p>

<p align="center">
  <a href="https://harshil-anuwadia.github.io/cad-preprocess/"><strong>Documentation</strong></a> ·
  <a href="https://harshil-anuwadia.github.io/cad-preprocess/docs/api/index.html">API Reference</a> ·
  <a href="https://harshil-anuwadia.github.io/cad-preprocess/docs/examples.html">Examples</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey?style=flat-square" alt="Platform">
</p>

---

## Why this exists

- **Reproducibility**: Configuration hashing ensures that your inference data matches your training data exactly.
- **Decompression**: Native support for JPEG Lossless (Process 14), JPEG 2000, and RLE via bundled codecs.
- **Intensity Mapping**: Consistent VOI LUT and windowing application (Soft Tissue, Lung, Bone, or Custom).
- **Metadata Integrity**: Profile-based extraction designed for ML feature engineering and clinical auditing.

---

## Installation

### Arch Linux (Native)
We provide a standard `PKGBUILD` and a bundled installer for Arch users.

```bash
# Standard installation
makepkg -si

# Or build a self-contained bundle (recommended for production)
./build_arch.sh
sudo pacman -U cad-preprocess-bundled-0.1.0-1-x86_64.pkg.tar.zst
```

### Debian / Ubuntu
Use the build script to create a self-contained `.deb` package.

```bash
./build_deb.sh
sudo dpkg -i cad-preprocess_0.1.0_all.deb
```

### Standard Python (pip)
```bash
pip install cad-preprocess
# For decompression support
pip install pylibjpeg pylibjpeg-libjpeg python-gdcm
```

---

## Usage

### Command Line Interface
The CLI is designed for batch processing of local datasets.

```bash
# Basic directory processing
cad-preprocess -i /data/scans -o /data/output

# Apply a specific CT window and resize
cad-preprocess -i ./input -o ./output --window-center 40 --window-width 400 --target-size 512 512

# Dry run to verify file discovery
cad-preprocess -i ./input -o ./output --dry-run
```

### Python API
Integrate the pipeline directly into your data loader or inference server.

```python
from cad_preprocess import CADPreprocessor

# Load settings from YAML
processor = CADPreprocessor.from_config("config.yaml")

# Process a single scan
result = processor.process("scan.dcm")
image_array = result.image  # Normalized NumPy array
metadata = result.metadata  # Extracted patient/geometry tags

# The configuration hash allows you to track pipeline versions
print(f"Pipeline signature: {processor.config_hash}")
```

### DICOM Explorer
A lightweight PyQt6 utility is included for interactive dataset browsing.

```bash
cad-preprocess-explorer
```

---

## Configuration

Standardize your pipeline using a YAML configuration. This ensures every researcher on the team uses the same parameters.

```yaml
preprocessing:
  windowing:
    strategy: fixed_window
    window_center: 40
    window_width: 400
  normalization: min_max
  resizing:
    target_height: 512
    target_width: 512
    preserve_aspect_ratio: true

metadata:
  profiles:
    - ml          # Orientation, Spacing, Thickness
    - patient     # De-identified demographics

output:
  format: png     # Or npy for direct NumPy storage
  naming_policy: sop_instance_uid
```

---

## Project Structure

- `cad_preprocess/`: Core logic (Engine, Metadata, Output).
- `cli.py`: Command-line tool.
- `explorer.py`: Interactive GUI browser.
- `PKGBUILD` / `build_arch.sh`: Arch Linux packaging.
- `debian/` / `build_deb.sh`: Debian/Ubuntu packaging.

## License

MIT License. See [LICENSE](LICENSE) for details.
